"""Desktop side of the Jetson Nano TensorRT path.

JetPack 4 (Nano) only has Python 3.6 + TensorRT 8.2, which cannot run the
torch/timm/open_clip cloud server. Instead we export each backbone to ONNX
(opset 13, the newest TensorRT 8.2 parses) and run it with TensorRT on the
Nano; preprocessing and the QualityGate head run in numpy (trt_common.py).

This script, run once on a desktop with the training stack:
  1. exports dinov2_vitl14 / dinov3_vitl16 / clip_vitl14 -> <out>/<name>.onnx
     (fp32 weights; the Nano builds fp16 engines from them)
  2. checks ONNX Runtime output == PyTorch output (cosine similarity)
  3. checks numpy preprocess == torchvision preprocess, numpy head == torch head
  4. writes reference features for --n-ref test images (the Nano's parity gate)
  5. runs the full pipeline on --agree-n test images and compares with the
     offline cloud predictions (deploy pipeline == paper pipeline)

Usage:
  python jetson_deploy/cloud/export_onnx.py --image-dir <plantwild_test> \
      --offline-preds outputs/cloud_predictions_test.csv --out outputs/trt_export
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import trt_common as tc  # noqa: E402
from cloud.server import QualityGate, load_clip_l, load_dinov2_l, load_dinov3_l  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ORIG_SDPA = F.scaled_dot_product_attention


def manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_):
    """Explicit attention: exports to MatMul/Softmax at opset 13 (the fused
    SDPA op needs opset >= 14, which TensorRT 8.2 cannot parse)."""
    s = scale if scale is not None else q.shape[-1] ** -0.5
    attn = (q * s) @ k.transpose(-2, -1)
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool \
            else attn + attn_mask
    return attn.softmax(dim=-1) @ v


class Feat(torch.nn.Module):
    """[1,3,H,W] normalized image -> raw [1,D] feature (normalized later in numpy)."""

    def __init__(self, fn: torch.nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x).float()


def bake_dinov2_pos_embed(model: torch.nn.Module, size: int) -> None:
    """DINOv2 bicubically resizes its 37x37 position grid to the input grid
    on every forward; TensorRT 8.2 has no cubic Resize. The input size is
    fixed, so compute the interpolated embedding once (exactly, in PyTorch)
    and serve it as a constant."""
    n_tok = 1 + (size // model.patch_size) ** 2
    with torch.no_grad():
        pe = model.interpolate_pos_encoding(
            torch.zeros(1, n_tok, model.embed_dim, device=model.pos_embed.device), size, size)
    model.register_buffer("pos_embed_fixed", pe.clone())
    model.interpolate_pos_encoding = lambda x, w, h: model.pos_embed_fixed


def bake_dinov3_rope(model: torch.nn.Module, size: int) -> None:
    """timm builds DINOv3 with dynamic_img_size=True, so the rotary embedding
    is recomputed from the input shape on every forward, which exports as
    Shape/Gather arithmetic and If nodes. At a fixed input size it is a
    constant: compute it once and serve it as a buffer."""
    grid = size // model.patch_embed.patch_size[0]
    with torch.no_grad():
        rope = model.rope.get_embed(shape=(grid, grid))
    if isinstance(rope, torch.Tensor):
        model.register_buffer("rope_fixed", rope.clone())
        model.rope.get_embed = lambda shape=None: model.rope_fixed
    else:  # tuple of tensors (e.g. sin, cos)
        for i, t in enumerate(rope):
            model.register_buffer(f"rope_fixed_{i}", t.clone())
        n = len(rope)
        model.rope.get_embed = lambda shape=None: tuple(getattr(model, f"rope_fixed_{i}") for i in range(n))


# Ops TensorRT 8.2 rejects or handles poorly; none may survive simplification.
TRT82_UNSAFE = {"If", "Loop", "Mod", "NonZero"}


def simplify(onnx_path: Path) -> list[str]:
    """Constant-fold shape arithmetic and conditionals (static input shape)
    with onnx-simplifier; return any op types TensorRT 8.2 would reject."""
    import onnx
    import onnxsim
    model, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
    if not ok:
        raise RuntimeError(f"onnxsim could not validate {onnx_path.name}")
    onnx.save(model, str(onnx_path))
    bad = sorted({n.op_type for n in model.graph.node if n.op_type in TRT82_UNSAFE}
                 | {"Resize(cubic)" for n in model.graph.node if n.op_type == "Resize"
                    and any(a.name == "mode" and a.s == b"cubic" for a in n.attribute)})
    return bad


def build(name: str, device: torch.device) -> torch.nn.Module:
    if name == "dinov2_vitl14":
        model = load_dinov2_l(device)[0]
        bake_dinov2_pos_embed(model, 336)
    elif name == "dinov3_vitl16":
        model = load_dinov3_l(device)[0]
        bake_dinov3_rope(model, 336)
    elif name == "clip_vitl14":
        model = load_clip_l(device)[0].visual   # encode_image(x) == visual(x)
    else:
        raise KeyError(name)
    return Feat(model).eval()


class ConvLinear(torch.nn.Module):
    """nn.Linear expressed as a 1x1 convolution (identical math). TensorRT 8.2
    compiles transformer MatMul+constant-weight subgraphs into one Myelin
    node that held ~13 bytes/parameter in device memory on the Nano (969 MB
    for six ViT-L blocks); as convolutions the weights stay in TensorRT's own
    fp16 layers and only the weightless attention products go to Myelin."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.out_features = lin.out_features
        self.conv = torch.nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
        self.conv.weight.data.copy_(lin.weight.data[:, :, None, None])
        if lin.bias is not None:
            self.conv.bias.data.copy_(lin.bias.data)

    def forward(self, x):
        lead = x.shape[:-1]
        y = self.conv(x.reshape(1, -1, x.shape[-1]).transpose(1, 2).unsqueeze(-1))  # [1, C, M, 1]
        return y.squeeze(-1).transpose(1, 2).reshape(*lead, self.out_features)


class ExplicitMHA(torch.nn.Module):
    """Self-attention equivalent of nn.MultiheadAttention (as used by
    open_clip), with its projections as ConvLinear. nn.MultiheadAttention
    applies in_proj_weight via F.linear internally, so swapping Linear
    modules alone would leave those weights inside Myelin."""

    def __init__(self, mha: torch.nn.MultiheadAttention):
        super().__init__()
        d, self.h = mha.embed_dim, mha.num_heads
        self.batch_first = mha.batch_first
        qkv = torch.nn.Linear(d, 3 * d, bias=mha.in_proj_bias is not None)
        qkv.weight.data.copy_(mha.in_proj_weight.data)
        if mha.in_proj_bias is not None:
            qkv.bias.data.copy_(mha.in_proj_bias.data)
        out = torch.nn.Linear(d, d, bias=mha.out_proj.bias is not None)
        out.weight.data.copy_(mha.out_proj.weight.data)
        if mha.out_proj.bias is not None:
            out.bias.data.copy_(mha.out_proj.bias.data)
        self.qkv, self.out = ConvLinear(qkv), ConvLinear(out)

    def forward(self, q, k=None, v=None, need_weights=False, attn_mask=None):
        if attn_mask is not None:
            raise NotImplementedError("vision tower uses no attention mask")
        x = q if self.batch_first else q.transpose(0, 1)          # [B, N, D]
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = manual_sdpa(qkv[0], qkv[1], qkv[2])                     # [B, H, N, d/H]
        y = self.out(a.transpose(1, 2).reshape(b, n, d))
        return (y if self.batch_first else y.transpose(0, 1)), None


def linear_as_conv(module: torch.nn.Module) -> torch.nn.Module:
    """Recursively replace nn.Linear / nn.MultiheadAttention (see above)."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.MultiheadAttention):
            setattr(module, name, ExplicitMHA(child))
        elif isinstance(child, torch.nn.Linear):
            setattr(module, name, ConvLinear(child))
        else:
            linear_as_conv(child)
    return module


class Chunk(torch.nn.Module):
    """Blocks [s, e) of a ViT backbone, plus its stem if first and its
    pooling/projection tail if last. Chaining the K chunks of a backbone
    computes exactly the full forward; each chunk is small enough for the
    TensorRT builder on a 4 GB Nano (a whole ViT-L is not)."""

    def __init__(self, name: str, model: torch.nn.Module, s: int, e: int, first: bool, last: bool):
        super().__init__()
        self.name, self.m, self.s, self.e, self.first, self.last = name, model, s, e, first, last
        if name == "dinov3_vitl16" and getattr(model, "rope_mixed", False):
            raise NotImplementedError("depth-dependent rope not supported")

    def blocks(self):
        if self.name == "clip_vitl14":
            return self.m.transformer.resblocks
        return self.m.blocks

    def forward(self, x):
        m = self.m
        if self.first:
            if self.name == "dinov2_vitl14":
                x = m.prepare_tokens_with_masks(x)
            elif self.name == "dinov3_vitl16":
                x, _ = m._pos_embed(m.patch_embed(x))
                x = m.norm_pre(x)
            else:
                x = m._embeds(x)
        clip_lnd = self.name == "clip_vitl14" and not m.transformer.batch_first
        if clip_lnd:
            x = x.transpose(0, 1)
        rope = m.rope.get_embed() if self.name == "dinov3_vitl16" else None
        for blk in self.blocks()[self.s:self.e]:
            if self.name == "dinov3_vitl16":
                x = blk(x, rope=rope)
            elif self.name == "clip_vitl14":
                x = blk(x, attn_mask=None)
            else:
                x = blk(x)
        if clip_lnd:
            x = x.transpose(0, 1)
        if self.last:
            if self.name == "dinov2_vitl14":
                x = m.head(m.norm(x)[:, 0])
            elif self.name == "dinov3_vitl16":
                x = m.forward_head(m.norm(x))
            else:
                x, _ = m._pool(x)
                if m.proj is not None:
                    x = x @ m.proj
        return x.float()


def make_chunks(name: str, feat: Feat, k: int) -> list[Chunk]:
    model = feat.fn
    n = len(Chunk(name, model, 0, 0, False, False).blocks())
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [Chunk(name, model, bounds[i], bounds[i + 1], i == 0, i == k - 1).eval() for i in range(k)]


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (tc.l2_normalize(a) * tc.l2_normalize(b)).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True, help="<class>/<img> test split")
    ap.add_argument("--offline-preds", required=True, help="cloud_predictions_test.csv")
    ap.add_argument("--head", default=str(Path(__file__).resolve().parents[1]
                                          / "checkpoints" / "qualitygate_fusion_3L.pt"))
    ap.add_argument("--class-names", default=str(Path(__file__).resolve().parents[1]
                                                 / "checkpoints" / "class_names.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--n-ref", type=int, default=24)
    ap.add_argument("--agree-n", type=int, default=300)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_dir)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    ref_imgs = images[:: max(1, len(images) // args.n_ref)][: args.n_ref]
    agree_imgs = images[:: max(1, len(images) // args.agree_n)][: args.agree_n]
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {"backbones": {}}

    feats_ref, feats_agree = [], []
    for spec in tc.BACKBONES:
        name, size = spec["name"], spec["size"]
        onnx_path = out / f"{name}.onnx"
        print(f"\n=== {name} ({size}px) ===")

        # -- export (CPU, fp32, explicit attention) -------------------------
        if not (args.skip_export and onnx_path.exists()):
            m = build(name, torch.device("cpu"))
            F.scaled_dot_product_attention = manual_sdpa
            # nn.MultiheadAttention (open_clip) otherwise takes the fused
            # _native_multi_head_attention path, which has no ONNX export.
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                t0 = time.time()
                torch.onnx.export(
                    m, torch.zeros(1, 3, size, size), str(onnx_path),
                    input_names=["input"], output_names=["feat"],
                    opset_version=args.opset, do_constant_folding=True, dynamo=False,
                )
            finally:
                F.scaled_dot_product_attention = _ORIG_SDPA
                torch.backends.mha.set_fastpath_enabled(True)
            bad = simplify(onnx_path)
            n_nodes = len(__import__("onnx").load(str(onnx_path), load_external_data=False).graph.node)
            print(f"  exported + simplified {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.0f} MB, "
                  f"{n_nodes} nodes, {time.time() - t0:.0f}s) | TensorRT-8.2-unsafe ops: {bad or 'none'}")
            if bad:
                raise SystemExit(f"{name}: unsupported ops remain after simplification: {bad}")
            del m

        # -- reference + agreement features (GPU, original attention) -------
        m = build(name, gpu)
        x_ref = [tc.preprocess(Image.open(p).convert("RGB"), size, spec["mean"], spec["std"])
                 for p in ref_imgs]
        with torch.no_grad():
            fr = np.concatenate([m(torch.from_numpy(x).to(gpu)).cpu().numpy() for x in x_ref])
            fa = np.concatenate([
                m(torch.from_numpy(tc.preprocess(Image.open(p).convert("RGB"), size,
                                                 spec["mean"], spec["std"])).to(gpu)).cpu().numpy()
                for p in agree_imgs])
        feats_ref.append(fr)
        feats_agree.append(fa)
        np.save(out / f"ref_{name}.npy", fr)

        # -- numpy preprocess vs torchvision preprocess ---------------------
        if name == "clip_vitl14":
            import open_clip
            _, _, tv_pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        else:
            from cloud.server import _eval_tf
            tv_pre = _eval_tf(size)
        pre_diff = max(float(np.abs(tv_pre(Image.open(p).convert("RGB")).numpy()[None] - x).max())
                       for p, x in zip(ref_imgs[:6], x_ref[:6]))

        # -- ONNX Runtime vs PyTorch ----------------------------------------
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        fo = np.concatenate([sess.run(None, {"input": x})[0] for x in x_ref[:4]])
        c = cos(fo, fr[:4])
        print(f"  onnx vs torch cosine: min {c.min():.6f} | numpy-vs-torchvision preprocess "
              f"max|diff| {pre_diff:.2e}")
        report["backbones"][name] = {"onnx_cos_min": float(c.min()), "preprocess_max_diff": pre_diff,
                                     "onnx_mb": onnx_path.stat().st_size / 1e6, "size": size}
        del m, sess
        torch.cuda.empty_cache()

    # -- head: torch -> npz, check numpy port ------------------------------
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    np.savez(out / "qualitygate_head.npz",
             linear_weight=sd["linear.weight"].numpy(), linear_bias=sd["linear.bias"].numpy(),
             prototypes=sd["prototypes"].numpy(), gate_logits=sd["gate_logits"].numpy(),
             log_temp=sd["log_temp"].numpy())
    head_np = tc.QualityGateNP(str(out / "qualitygate_head.npz"))
    head_t = QualityGate(2816, sd["linear.weight"].shape[0],
                         n_prototypes=sd["prototypes"].shape[1])
    head_t.load_state_dict(sd)
    head_t.eval()
    fused_ref = tc.fuse([f for f in feats_ref])
    with torch.no_grad():
        lt = head_t(torch.from_numpy(fused_ref)).numpy()
    head_diff = float(np.abs(head_np(fused_ref) - lt).max())
    print(f"\nhead: numpy vs torch max|logit diff| {head_diff:.2e}")
    report["head_max_logit_diff"] = head_diff

    # -- references for the Nano --------------------------------------------
    classes = tc.load_class_names(args.class_names)
    p_ref = tc.softmax(head_np(fused_ref))
    with (out / "ref_images.json").open("w", encoding="utf-8") as fh:
        json.dump([{"image": str(p.relative_to(root)).replace("\\", "/"),
                    "pred": classes[int(pr.argmax())], "conf": float(pr.max())}
                   for p, pr in zip(ref_imgs, p_ref)], fh, indent=1)

    # -- deploy pipeline vs offline paper predictions ----------------------
    offline = {}
    with open(args.offline_preds, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pth = r["image_path"].replace("\\", "/")
            offline[(r["true_label"], pth.rsplit("/", 1)[-1])] = r["cloud_pred"]
    p_agree = tc.softmax(head_np(tc.fuse(feats_agree)))
    n = agree = correct = 0
    for p, pr in zip(agree_imgs, p_agree):
        key = (p.parent.name, p.name)
        if key not in offline:
            continue
        pred = classes[int(pr.argmax())]
        n += 1
        agree += pred == offline[key]
        correct += pred == p.parent.name
    report["offline_agreement"] = {"n": n, "agree": agree / max(n, 1), "acc": correct / max(n, 1)}
    print(f"deploy vs offline cloud predictions: {agree}/{n} agree ({agree / max(n, 1):.1%}), "
          f"acc {correct / max(n, 1):.1%} on this subset")
    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
