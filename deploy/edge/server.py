"""Edge server (Raspberry Pi).

POST /predict  body=PredictRequest  -> PredictResponse
GET  /health   -> {"status": "ok", "arch": ..., "n_classes": ...}

Design:
  - On startup: load MobileNetV3-Large checkpoint to CPU. The Pi has no
    CUDA; running on 4 ARM cores at 224x224 takes ~50-150 ms/image
    depending on Pi model.
  - Per request: decode image -> run model -> compute margin/entropy.
    If margin < threshold (default 0.20), call cloud server with the same
    JPEG bytes; otherwise return edge prediction.
  - The cloud call is HTTP+JSON. On timeout / connection failure we fall
    back to the edge prediction (graceful degradation).
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import List

import httpx
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image

# Allow `from shared.X import ...` when this file is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import config as cfg_mod  # noqa: E402
from shared import heartbeat as hb_mod  # noqa: E402
from shared.image_io import decode_b64, encode_b64  # noqa: E402
from shared.protocol import (  # noqa: E402
    ClassifyRequest, ClassifyResponse, EdgePrediction,
    PredictRequest, PredictResponse,
)
from shared.routing import (  # noqa: E402
    entropy_from_probs, margin_from_probs, should_route,
)
from shared.stats import StatTracker  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [edge] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("edge")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(arch: str, n_classes: int) -> torch.nn.Module:
    """Pi-friendly: only torchvision is imported here."""
    import torch.nn as nn
    from torchvision.models import (
        MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
        EfficientNet_B0_Weights,
        mobilenet_v3_large, mobilenet_v3_small, efficientnet_b0,
    )
    if arch == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif arch == "mobilenet_v3_large":
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    elif arch == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown arch {arch}")
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, n_classes)
    return m


def load_class_names(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"class_names file missing: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EdgeState:
    def __init__(self, cfg: cfg_mod.DeployConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.edge.device)
        self.class_names = load_class_names(cfg.edge.class_names_path)
        if len(self.class_names) != cfg.edge.n_classes:
            log.warning("class_names count (%d) != cfg.edge.n_classes (%d); "
                        "using class_names length",
                        len(self.class_names), cfg.edge.n_classes)
            cfg.edge.n_classes = len(self.class_names)

        ck = torch.load(cfg.edge.ckpt_path, map_location="cpu", weights_only=False)
        arch_in_ckpt = ck.get("arch", cfg.edge.arch)
        self.model = build_model(arch_in_ckpt, cfg.edge.n_classes)
        sd = ck.get("state_dict", ck.get("model", ck))
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(self.device)

        # Preprocess: ImageNet stats, 224x224 center crop (matches train_edge_model.py)
        from torchvision import transforms as T
        s = cfg.edge.img_size
        self.preprocess = T.Compose([
            T.Resize(int(s * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(s),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # HTTP client to cloud (lazy: only opened if a request actually escalates)
        self._http: httpx.Client | None = None

        # Stats + dashboard heartbeat
        self.stats = StatTracker(role="edge")
        self.node_id = cfg.dashboard.node_id or hb_mod.default_node_id()
        self.heartbeat = hb_mod.maybe_start(
            cfg.dashboard.url,
            node_id=self.node_id,
            role="edge",
            port=cfg.server.port,
            stats=self.stats,
            config={
                "arch": arch_in_ckpt,
                "n_classes": cfg.edge.n_classes,
                "margin_threshold": cfg.edge.margin_threshold,
                "cloud_url": cfg.edge.cloud_url,
                "device": str(self.device),
            },
            interval_s=cfg.dashboard.interval_s,
        )

        log.info("loaded arch=%s n_classes=%d device=%s threshold=%.3f cloud=%s node_id=%s",
                 arch_in_ckpt, cfg.edge.n_classes, self.device,
                 cfg.edge.margin_threshold, cfg.edge.cloud_url, self.node_id)

    def http_client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                base_url=self.cfg.edge.cloud_url,
                timeout=self.cfg.edge.cloud_timeout_s,
            )
        return self._http

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple[EdgePrediction, np.ndarray]:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        logits = self.model(x).float()
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        order = np.argsort(-probs)
        top5_idx = order[:5]
        margin = margin_from_probs(probs)
        ent = entropy_from_probs(probs)
        return EdgePrediction(
            label=self.class_names[int(order[0])],
            confidence=float(probs[order[0]]),
            margin=margin,
            entropy=ent,
            top5=[self.class_names[i] for i in top5_idx],
            top5_probs=[float(probs[i]) for i in top5_idx],
        ), probs


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

state: EdgeState  # populated in lifespan

app = FastAPI(title="plant-edge", version="1.0")


@app.on_event("startup")
def _startup():
    global state
    cfg = cfg_mod.load()
    if cfg.role and cfg.role != "edge":
        log.warning("role=%s but loading edge server; ignoring role mismatch", cfg.role)
    state = EdgeState(cfg)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "node_id": state.node_id,
        "role": "edge",
        "arch": state.cfg.edge.arch,
        "n_classes": state.cfg.edge.n_classes,
        "threshold": state.cfg.edge.margin_threshold,
        "cloud_url": state.cfg.edge.cloud_url,
        "device": str(state.device),
    }


@app.get("/stats")
def stats_endpoint():
    return state.stats.snapshot().dict()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    rid = req.request_id or uuid.uuid4().hex[:8]

    try:
        img = decode_b64(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"decode failed: {e}") from e

    edge_pred, _ = state.predict(img)

    # Routing decision (force_route overrides)
    if req.force_route == "edge":
        route = False
    elif req.force_route == "cloud":
        route = True
    else:
        route = should_route(edge_pred.margin, state.cfg.edge.margin_threshold)

    cloud_pred: ClassifyResponse | None = None
    bw_kb = 0.0
    if route:
        # Re-encode to JPEG (size measured for accounting; we already have b64 in req,
        # but its size depends on caller's encoding choices, so we re-encode at our quality).
        b64, kb = encode_b64(img, quality=state.cfg.edge.jpeg_quality)
        bw_kb = kb
        cloud_req = ClassifyRequest(image_b64=b64, image_kb=kb, request_id=rid)
        try:
            r = state.http_client().post("/classify", json=cloud_req.dict())
            r.raise_for_status()
            cloud_pred = ClassifyResponse(**r.json())
            label = cloud_pred.label
            confidence = cloud_pred.confidence
            source = "cloud"
        except Exception as e:
            log.warning("rid=%s cloud call failed (%s); falling back to edge", rid, e)
            label = edge_pred.label
            confidence = edge_pred.confidence
            source = "edge"
            route = False
    else:
        label = edge_pred.label
        confidence = edge_pred.confidence
        source = "edge"

    dt_ms = (time.perf_counter() - t0) * 1000
    state.stats.record_edge(
        margin=edge_pred.margin, latency_ms=dt_ms, label=label,
        routed=route, bandwidth_kb=bw_kb,
    )
    log.info("rid=%s margin=%.3f route=%s source=%s label=%s lat=%.1fms bw=%.2fKB",
             rid, edge_pred.margin, route, source, label, dt_ms, bw_kb)
    return PredictResponse(
        label=label,
        confidence=confidence,
        source=source,
        routed=route,
        edge_prediction=edge_pred,
        cloud_prediction=cloud_pred,
        routing_signal=edge_pred.margin,
        threshold=state.cfg.edge.margin_threshold,
        latency_ms=dt_ms,
        bandwidth_kb=bw_kb,
    )


def main() -> None:
    import uvicorn
    cfg = cfg_mod.load()
    uvicorn.run(
        "edge.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
