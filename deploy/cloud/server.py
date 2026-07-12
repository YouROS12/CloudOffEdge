"""Cloud server (Jetson Nano or any CUDA-capable host).

POST /classify  body=ClassifyRequest  -> ClassifyResponse
GET  /health    -> {"status": "ok", ...}

Pipeline per request:
  decode JPEG -> for each backbone:
                    load weights to device (sequential mode) OR reuse cached
                    extract feature, L2-normalize
                    free weights from VRAM (sequential mode)
                 concat features -> L2-normalize
                 QualityGate head -> softmax -> top-1

Sequential mode is the Nano-4GB default: backbones are loaded one at a
time, used, then deleted (and torch.cuda.empty_cache() called). Memory
peaks at one backbone at a time (~600 MB fp16) which fits.

Resident mode (sequential=False) keeps all three backbones loaded; faster
per-request (~3-5x) but needs ~6 GB VRAM headroom. Use on a desktop GPU.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import config as cfg_mod  # noqa: E402
from shared import heartbeat as hb_mod  # noqa: E402
from shared.image_io import decode_b64  # noqa: E402
from shared.protocol import ClassifyRequest, ClassifyResponse  # noqa: E402
from shared.stats import StatTracker  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cloud] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cloud")


# ---------------------------------------------------------------------------
# Backbone loaders. Each returns (extract_fn, feat_dim, preprocess).
# extract_fn takes a PIL image and returns a [1, D] L2-normalized tensor on
# `device`. Loaders are kept lightweight so they can be re-imported per
# request in sequential mode.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _eval_tf(size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_dinov2_l(device: torch.device, img_size: int = 336):
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14",
        pretrained=True, verbose=False,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    pre = _eval_tf(img_size)

    @torch.no_grad()
    def extract(img):
        x = pre(img).unsqueeze(0).to(device)
        feat = model(x).float()
        return F.normalize(feat, dim=-1)
    return model, extract, model.embed_dim, pre


def load_dinov3_l(device: torch.device, img_size: int = 336):
    import timm
    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=True, num_classes=0,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    pre = _eval_tf(img_size)

    @torch.no_grad()
    def extract(img):
        x = pre(img).unsqueeze(0).to(device)
        feat = model(x).float()
        return F.normalize(feat, dim=-1)
    return model, extract, model.num_features, pre


def load_clip_l(device: torch.device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai",
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def extract(img):
        x = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(x).float()
        return F.normalize(feat, dim=-1)
    # feat dim probe
    with torch.no_grad():
        d = int(model.encode_image(torch.zeros(1, 3, 224, 224, device=device)).shape[-1])
    return model, extract, d, preprocess


BACKBONE_LOADERS: dict[str, Callable] = {
    "dinov2_vitl14": lambda dev: load_dinov2_l(dev),
    "dinov3_vitl16": lambda dev: load_dinov3_l(dev),
    "clip_vitl14":   lambda dev: load_clip_l(dev),
}


# ---------------------------------------------------------------------------
# QualityGate head. Re-implemented here without importing paper1 to avoid
# pulling sklearn/timm chains we don't need on Nano. State dict shape
# matches paper1.src.classifiers.QualityGate.
# ---------------------------------------------------------------------------

class QualityGate(torch.nn.Module):
    def __init__(self, feat_dim: int, n_classes: int, n_prototypes: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.linear = torch.nn.Linear(feat_dim, n_classes, bias=True)
        self.prototypes = torch.nn.Parameter(
            torch.randn(n_classes, n_prototypes, feat_dim) * 0.01
        )
        self.gate_logits = torch.nn.Parameter(torch.zeros(n_classes))
        self.log_temp = torch.nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear path
        linear_logits = self.linear(F.normalize(x, dim=-1))
        # Prototype path: cosine similarity to nearest prototype per class
        x_n = F.normalize(x, dim=-1)                                  # [B, D]
        protos_n = F.normalize(self.prototypes, dim=-1)               # [C, K, D]
        sims = torch.einsum("bd,ckd->bck", x_n, protos_n)             # [B, C, K]
        proto_logits = sims.max(dim=-1).values / self.log_temp.exp()  # [B, C]
        gate = torch.sigmoid(self.gate_logits)                        # [C]
        return gate * linear_logits + (1.0 - gate) * proto_logits


def load_class_names(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"class_names file missing: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CloudState:
    def __init__(self, cfg: cfg_mod.DeployConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.cloud.device if torch.cuda.is_available()
                                   or cfg.cloud.device == "cpu" else "cpu")
        self.class_names = load_class_names(cfg.cloud.class_names_path)
        self.n_classes = len(self.class_names)

        # Cache for resident mode: backbone_name -> (extract_fn, feat_dim).
        # In sequential mode, we don't populate this and reload each request.
        self._resident: dict[str, Tuple[Callable, int]] = {}
        self._resident_models: list = []  # keep refs alive

        if not cfg.cloud.sequential:
            log.info("resident mode: pre-loading %d backbones...", len(cfg.cloud.backbones))
            for name in cfg.cloud.backbones:
                model, extract, dim, _ = BACKBONE_LOADERS[name](self.device)
                self._resident[name] = (extract, dim)
                self._resident_models.append(model)
            self.feat_dim = sum(d for _, d in self._resident.values())
        else:
            # Probe each backbone briefly to learn feat_dim, then free.
            log.info("sequential mode: probing %d backbones for feat_dim...", len(cfg.cloud.backbones))
            self.feat_dim = 0
            for name in cfg.cloud.backbones:
                model, _, dim, _ = BACKBONE_LOADERS[name](self.device)
                self.feat_dim += dim
                log.info("  %-18s D=%d", name, dim)
                del model
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Load QualityGate head
        ck = torch.load(cfg.cloud.head_ckpt_path, map_location=self.device, weights_only=False)
        sd = ck.get("state_dict", ck)
        n_proto = ck.get("result", {}).get("n_prototypes", 8)
        self.head = QualityGate(self.feat_dim, self.n_classes, n_prototypes=n_proto).to(self.device)
        self.head.load_state_dict(sd, strict=True)
        self.head.eval()

        # Stats + dashboard heartbeat
        self.stats = StatTracker(role="cloud")
        self.node_id = cfg.dashboard.node_id or hb_mod.default_node_id()
        self.heartbeat = hb_mod.maybe_start(
            cfg.dashboard.url,
            node_id=self.node_id,
            role="cloud",
            port=cfg.server.port,
            stats=self.stats,
            config={
                "device": str(self.device),
                "n_classes": self.n_classes,
                "feat_dim": self.feat_dim,
                "sequential": cfg.cloud.sequential,
                "backbones": ",".join(cfg.cloud.backbones),
            },
            interval_s=cfg.dashboard.interval_s,
        )

        log.info("cloud ready: device=%s feat_dim=%d n_classes=%d sequential=%s node_id=%s",
                 self.device, self.feat_dim, self.n_classes, cfg.cloud.sequential, self.node_id)

    def _extract_one(self, name: str, img) -> torch.Tensor:
        if not self.cfg.cloud.sequential:
            extract, _ = self._resident[name]
            return extract(img)
        # Sequential: load, extract, free
        model, extract, _, _ = BACKBONE_LOADERS[name](self.device)
        feat = extract(img)
        del model, extract
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return feat

    @torch.no_grad()
    def classify(self, img) -> Tuple[str, float, list[str], list[float], list[str]]:
        feats = []
        used = []
        for name in self.cfg.cloud.backbones:
            f = self._extract_one(name, img)
            feats.append(f)
            used.append(name)
        fused = F.normalize(torch.cat(feats, dim=-1), dim=-1)
        logits = self.head(fused.float())
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        import numpy as np
        order = np.argsort(-probs)
        top5_idx = order[:5]
        return (
            self.class_names[int(order[0])],
            float(probs[order[0]]),
            [self.class_names[i] for i in top5_idx],
            [float(probs[i]) for i in top5_idx],
            used,
        )


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

state: CloudState
app = FastAPI(title="plant-cloud", version="1.0")


@app.on_event("startup")
def _startup():
    global state
    cfg = cfg_mod.load()
    if cfg.role and cfg.role != "cloud":
        log.warning("role=%s but loading cloud server; ignoring", cfg.role)
    state = CloudState(cfg)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "node_id": state.node_id,
        "role": "cloud",
        "device": str(state.device),
        "n_classes": state.n_classes,
        "feat_dim": state.feat_dim,
        "backbones": state.cfg.cloud.backbones,
        "sequential": state.cfg.cloud.sequential,
    }


@app.get("/stats")
def stats_endpoint():
    return state.stats.snapshot().dict()


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    t0 = time.perf_counter()
    try:
        img = decode_b64(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"decode failed: {e}") from e

    label, conf, top5, top5_p, used = state.classify(img)
    dt_ms = (time.perf_counter() - t0) * 1000
    state.stats.record_cloud(latency_ms=dt_ms, label=label)

    log.info("rid=%s label=%s conf=%.3f lat=%.0fms in=%.1fKB",
             req.request_id or "-", label, conf, dt_ms, req.image_kb)
    return ClassifyResponse(
        label=label,
        confidence=conf,
        top5=top5,
        top5_probs=top5_p,
        latency_ms=dt_ms,
        backbones_used=used,
    )


def main() -> None:
    import uvicorn
    cfg = cfg_mod.load()
    uvicorn.run(
        "cloud.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
