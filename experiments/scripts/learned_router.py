"""
Learned router v3 — rigorous: train on val (1820), evaluate on full test (3677).

Two router variants:
  - probs_only:  edge softmax (89-d) + 3 scalars
  - probs_plus_dinov2: above + DINOv2-L features (1024-d)

Pipeline:
  1. Load val edge logits + val cloud preds
  2. Build router train inputs+labels (useful_offload = cloud right & edge wrong)
  3. Train MLP, early-stop on val AUC measured via 5-fold CV inside val
     (so eval on test stays clean)
  4. Run trained router on full 3677-sample test, sweep threshold,
     compare against confidence/margin/entropy/oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = Path(__file__).resolve().parents[2] / "artifacts"
CACHE = Path(__import__("os").environ.get("PAPER1_CACHED_FEATURES",
                                         Path(__file__).resolve().parents[2] / "data" / "paper1_cached_features"))


def key(path: str) -> str:
    """class/filename: identifies an image whatever the dataset root."""
    return "/".join(str(path).replace("\\", "/").split("/")[-2:])


def build_features(probs: torch.Tensor, dinov2: torch.Tensor | None = None) -> torch.Tensor:
    top2 = probs.topk(2, dim=-1).values
    max_p = top2[:, :1]
    margin = top2[:, :1] - top2[:, 1:2]
    ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    scalars = torch.cat([max_p, margin, ent], dim=-1)
    if dinov2 is None:
        return torch.cat([probs, scalars], dim=-1)
    return torch.cat([probs, scalars, dinov2], dim=-1)


class RouterMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(X, y, *, device, epochs=100, lr=1e-3, batch_size=128,
              wd=1e-3, hidden=128, dropout=0.4, seed=1):
    torch.manual_seed(seed)
    in_dim = X.shape[-1]
    model = RouterMLP(in_dim, hidden=hidden, dropout=dropout).to(device)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pos_w = torch.tensor([neg / max(1.0, pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xd = X.to(device)
    yd = y.to(device).float()
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xd), device=device)
        for i in range(0, len(Xd), batch_size):
            b = idx[i:i + batch_size]
            loss = loss_fn(model(Xd[b]), yd[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def cv_best_epochs(X, y, *, device, candidate_epochs=(20, 40, 60, 100, 150),
                   hidden=128, dropout=0.4, n_splits=5, seed=1):
    """Pick the # of epochs that maximises out-of-fold AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_np = y.numpy()
    results = {}
    for ep in candidate_epochs:
        oof = np.zeros(len(y))
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y_np)):
            model = train_one(X[tr], y[tr], device=device, epochs=ep,
                              hidden=hidden, dropout=dropout, seed=seed + fold)
            model.eval()
            with torch.no_grad():
                oof[va] = torch.sigmoid(model(X[va].to(device))).cpu().numpy()
        auc = roc_auc_score(y_np, oof)
        results[ep] = auc
        print(f"  CV epochs={ep:3d}  oof_AUC={auc:.4f}")
    best = max(results, key=results.get)
    return best, results


def sweep(scores, edge_pred, cloud_pred, true,
          higher_is_uncertain=True, n_thresh=80,
          edge_lat=50.0, cloud_lat=250.0, send=100.0, ret=30.0, image_kb=250.0):
    lo, hi = float(scores.min()), float(scores.max())
    rows = []
    for t in np.linspace(lo, hi, n_thresh):
        offload = (scores > t) if higher_is_uncertain else (scores < t)
        final = np.where(offload, cloud_pred, edge_pred)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(true, final)),
            "offload_rate": float(offload.mean()),
            "average_latency_ms": float(np.where(offload, edge_lat + send + cloud_lat + ret, edge_lat).mean()),
            "bandwidth_kb_per_sample": float(offload.mean() * image_kb),
        })
    return pd.DataFrame(rows)


def at_offload(curve, t):
    return curve.loc[(curve["offload_rate"] - t).abs().idxmin()]


def get_dinov2_features(paths: list[str], split_cache_path: str, fallback_extract_split: str) -> torch.Tensor:
    """Return DINOv2 features in the order of `paths`.

    - For test: paper-1 has them cached (use plantwild__dinov2_vitl14.pt)
    - For val: we cached them in outputs/val_features_cache/dinov2_vitl14.pt
    """
    d = torch.load(split_cache_path, map_location="cpu", weights_only=False)
    if "features" in d:
        feats = d["features"].float()
        order_paths = d["image_paths"]
    else:
        # paper-1 cache: needs the loader-ordered paths to pair up
        feats = d[f"{fallback_extract_split}_features"].float()
        # paper-1 cache doesn't include paths; paths must be passed in
        order_paths = None
    feats = F.normalize(feats, dim=-1)
    if order_paths is not None and [key(p) for p in order_paths] != [key(p) for p in paths]:
        idx = {key(p): i for i, p in enumerate(order_paths)}
        feats = feats[[idx[key(p)] for p in paths]]
    return feats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/phase0_routing_signals")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load val (router train) ────────────────────────────────────────
    val_bundle = torch.load(OUT / "edge_predictions_val.logits.pt",
                            map_location="cpu", weights_only=False)
    val_logits = val_bundle["logits"]
    val_paths = val_bundle["image_paths"]
    val_class_names = list(val_bundle["class_names"])
    val_probs = F.softmax(val_logits, dim=-1)
    val_pred_id = val_logits.argmax(-1).numpy()
    val_edge_pred = np.array([val_class_names[i] for i in val_pred_id])

    val_cloud = pd.read_csv(OUT / "cloud_predictions_val.csv")
    val_cloud.index = [key(x) for x in val_cloud.image_path]
    val_cloud_pred = val_cloud.loc[[key(x) for x in val_paths]]["cloud_pred"].to_numpy()
    val_true = val_cloud.loc[[key(x) for x in val_paths]]["true_label"].to_numpy()
    val_edge_right = (val_edge_pred == val_true)
    val_cloud_right = (val_cloud_pred == val_true)
    val_y = (val_cloud_right & ~val_edge_right).astype(np.float32)

    print(f"[val] N={len(val_y)}  edge={val_edge_right.mean()*100:.2f}%  "
          f"cloud={val_cloud_right.mean()*100:.2f}%  prevalence={val_y.mean()*100:.2f}%")

    val_dino = get_dinov2_features(val_paths, OUT / "val_features_cache" / "dinov2_vitl14.pt", "")

    # ── Load test (router eval) ────────────────────────────────────────
    test_bundle = torch.load(OUT / "edge_predictions_test.logits.pt",
                             map_location="cpu", weights_only=False)
    test_logits = test_bundle["logits"]
    test_paths = test_bundle["image_paths"]
    test_class_names = list(test_bundle["class_names"])
    test_probs = F.softmax(test_logits, dim=-1)
    test_pred_id = test_logits.argmax(-1).numpy()
    test_edge_pred = np.array([test_class_names[i] for i in test_pred_id])
    test_cloud = pd.read_csv(OUT / "cloud_predictions_test.csv")
    test_cloud.index = [key(x) for x in test_cloud.image_path]
    test_cloud_pred = test_cloud.loc[[key(x) for x in test_paths]]["cloud_pred"].to_numpy()
    test_true = test_cloud.loc[[key(x) for x in test_paths]]["true_label"].to_numpy()
    test_edge_right = (test_edge_pred == test_true)
    test_cloud_right = (test_cloud_pred == test_true)
    test_cloud_acc = float(test_cloud_right.mean())
    test_edge_acc = float(test_edge_right.mean())
    target = 0.95 * test_cloud_acc

    # paper-1 cache holds test features in load_plantwild() order; align by re-running the loader
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_edge_model as base
    base._add_turboquant_root_to_path()
    from plantwild import load_plantwild
    _, test_items, _ = load_plantwild()
    loader_keys = [key(p) for p, _ in test_items]
    cache = torch.load(CACHE / "plantwild__dinov2_vitl14.pt", map_location="cpu", weights_only=False)
    test_dino = F.normalize(cache["test_features"].float(), dim=-1)
    if loader_keys != [key(p) for p in test_paths]:
        idx = {k: i for i, k in enumerate(loader_keys)}
        test_dino = test_dino[[idx[key(p)] for p in test_paths]]
    print(f"[test] N={len(test_y := (test_cloud_right & ~test_edge_right))}  "
          f"edge={test_edge_acc*100:.2f}%  cloud={test_cloud_acc*100:.2f}%  "
          f"target={target*100:.2f}%")

    # Build features
    Xv_p = build_features(val_probs)
    Xv_pd = build_features(val_probs, val_dino)
    Xt_p = build_features(test_probs)
    Xt_pd = build_features(test_probs, test_dino)
    yv = torch.tensor(val_y)

    print(f"\n[features] probs_only dim={Xv_p.shape[-1]}  "
          f"probs+dinov2 dim={Xv_pd.shape[-1]}")

    # ── CV-pick best epochs for each variant on val ────────────────────
    print("\n[CV] selecting epochs (probs_only)")
    best_ep_p, _ = cv_best_epochs(Xv_p, yv, device=device,
                                  candidate_epochs=(20, 40, 60, 100), hidden=128, dropout=0.4)
    print(f"  -> probs_only: epochs={best_ep_p}")
    print("\n[CV] selecting epochs (probs+dinov2)")
    best_ep_pd, _ = cv_best_epochs(Xv_pd, yv, device=device,
                                   candidate_epochs=(20, 40, 60, 100), hidden=128, dropout=0.4)
    print(f"  -> probs+dinov2: epochs={best_ep_pd}")

    # ── Final fit on all val, predict on test ──────────────────────────
    print("\n[fit-final] probs_only on full val")
    m_p = train_one(Xv_p, yv, device=device, epochs=best_ep_p, hidden=128, dropout=0.4)
    m_p.eval()
    with torch.no_grad():
        s_p = torch.sigmoid(m_p(Xt_p.to(device))).cpu().numpy()

    print("[fit-final] probs+dinov2 on full val")
    m_pd = train_one(Xv_pd, yv, device=device, epochs=best_ep_pd, hidden=128, dropout=0.4)
    m_pd.eval()
    with torch.no_grad():
        s_pd = torch.sigmoid(m_pd(Xt_pd.to(device))).cpu().numpy()

    # per-image scores and the exact escalation needed for 95 % of cloud accuracy
    def exact_esc(score):
        order = np.argsort(-score, kind="stable")
        ok_e, ok_c = test_edge_right.astype(int), test_cloud_right.astype(int)
        acc, n = ok_e.sum(), len(score)
        for k, i in enumerate(order, 1):
            acc += ok_c[i] - ok_e[i]
            if acc / n >= target:
                return k / n
        return None
    exact = {"learned_probs": exact_esc(s_p), "learned_probs_dinov2": exact_esc(s_pd)}
    np.savez(out_dir / "learned_router_scores.npz", image_paths=np.array(test_paths),
             learned_probs=s_p, learned_probs_dinov2=s_pd)
    print("[exact esc@95]", exact)

    # AUC on test
    test_y_np = (test_cloud_right & ~test_edge_right).astype(int)
    auc_p = roc_auc_score(test_y_np, s_p)
    auc_pd = roc_auc_score(test_y_np, s_pd)
    print(f"\n[test-AUC] probs_only      = {auc_p:.4f}")
    print(f"[test-AUC] probs+dinov2    = {auc_pd:.4f}")

    # Heuristics on test
    test_conf = test_probs.max(-1).values.numpy()
    top2 = test_probs.topk(2, -1).values
    test_margin = (top2[:, 0] - top2[:, 1]).numpy()
    test_entropy = -(test_probs.clamp_min(1e-12) *
                     test_probs.clamp_min(1e-12).log()).sum(-1).numpy()
    auc_conf = roc_auc_score(test_y_np, 1 - test_conf)
    auc_marg = roc_auc_score(test_y_np, 1 - test_margin)
    auc_ent = roc_auc_score(test_y_np, test_entropy)
    print(f"[test-AUC] 1-confidence    = {auc_conf:.4f}")
    print(f"[test-AUC] 1-margin        = {auc_marg:.4f}")
    print(f"[test-AUC] entropy         = {auc_ent:.4f}")
    (out_dir / "learned_router_exact.json").write_text(json.dumps(
        {"edge": "MobileNetV3-Small", "target_acc": target, "esc_for_95": exact,
         "epochs": {"probs": best_ep_p, "probs_dinov2": best_ep_pd},
         "test_auc_useful": {"learned_probs": auc_p, "learned_probs_dinov2": auc_pd, "max_prob": auc_conf,
                             "margin": auc_marg, "entropy": auc_ent}}, indent=1))

    # Sweep curves
    curves = {
        "learned_v3_probs":       sweep(s_p, test_edge_pred, test_cloud_pred, test_true,
                                        higher_is_uncertain=True),
        "learned_v3_probs+dino":  sweep(s_pd, test_edge_pred, test_cloud_pred, test_true,
                                        higher_is_uncertain=True),
        "confidence":             sweep(test_conf, test_edge_pred, test_cloud_pred, test_true,
                                        higher_is_uncertain=False),
        "margin":                 sweep(test_margin, test_edge_pred, test_cloud_pred, test_true,
                                        higher_is_uncertain=False),
        "entropy":                sweep(test_entropy, test_edge_pred, test_cloud_pred, test_true,
                                        higher_is_uncertain=True),
    }
    for name, c in curves.items():
        c["signal"] = name
        c.to_csv(out_dir / f"v3_{name.replace('+', '_')}_curve.csv", index=False)

    # Oracle
    benefit = test_cloud_right.astype(int) - test_edge_right.astype(int)
    order = np.argsort(-benefit)
    oracle_rows = []
    cum = np.zeros(len(test_true), dtype=bool)
    for k in range(len(test_true) + 1):
        if k > 0:
            cum[order[k - 1]] = True
        final = np.where(cum, test_cloud_pred, test_edge_pred)
        oracle_rows.append({"k": k, "offload_rate": k / len(test_true),
                            "accuracy": float(accuracy_score(test_true, final))})
    oracle = pd.DataFrame(oracle_rows)

    print("\n" + "=" * 80)
    print(f"[FULL TEST] N={len(test_true)}  edge={test_edge_acc*100:.2f}%  "
          f"cloud={test_cloud_acc*100:.2f}%  95% target={target*100:.2f}%")
    print("=" * 80)
    for budget in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        print(f"\nAt offload \u2248 {budget*100:.0f}%:")
        print(f"  {'signal':<24} {'acc':>8} {'%cloud':>8}")
        for name, c in curves.items():
            r = at_offload(c, budget)
            print(f"  {name:<24} {r['accuracy']*100:7.2f}% {r['accuracy']/test_cloud_acc*100:7.2f}%")
        idx = (oracle["offload_rate"] - budget).abs().idxmin()
        r = oracle.loc[idx]
        print(f"  {'oracle':<24} {r['accuracy']*100:7.2f}% {r['accuracy']/test_cloud_acc*100:7.2f}%")

    print("\n" + "=" * 80)
    print("GO/NO-GO check on FULL test (95% of cloud at \u226430% offload)")
    print("=" * 80)
    for name, c in curves.items():
        hits = c[c["accuracy"] >= target]
        if not len(hits):
            print(f"  {name:<24} NEVER")
            continue
        r = hits.sort_values("offload_rate").iloc[0]
        verdict = "GO " if r["offload_rate"] <= 0.30 else "NO-GO"
        print(f"  {name:<24} earliest hit offload={r['offload_rate']*100:5.2f}%  "
              f"acc={r['accuracy']*100:.2f}%  -> {verdict}")
    hits = oracle[oracle["accuracy"] >= target]
    if len(hits):
        r = hits.iloc[0]
        print(f"  {'oracle':<24} earliest hit offload={r['offload_rate']*100:5.2f}%  "
              f"acc={r['accuracy']*100:.2f}%")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5.5))
        plt.plot(oracle["offload_rate"] * 100, oracle["accuracy"] * 100,
                 label="oracle", color="black", linewidth=2)
        for name, color in [("learned_v3_probs+dino", "tab:red"),
                            ("learned_v3_probs", "tab:purple"),
                            ("confidence", "tab:blue"),
                            ("margin", "tab:green"),
                            ("entropy", "tab:orange")]:
            c = curves[name].sort_values("offload_rate")
            plt.plot(c["offload_rate"] * 100, c["accuracy"] * 100,
                     label=name, color=color, marker=".", linewidth=1.4)
        plt.axhline(test_cloud_acc * 100, ls="--", color="grey", label="cloud-only")
        plt.axhline(test_edge_acc * 100, ls=":", color="grey", label="edge-only")
        plt.axhline(target * 100, ls="-.", color="red", label="95% target")
        plt.axvline(30, ls="-.", color="green", label="30% offload budget")
        plt.xlabel("Offload rate (%)")
        plt.ylabel("Accuracy (%)")
        plt.title("Learned router v3 (train on val, eval on full test)")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "learned_router_v3_compare.png", dpi=150)
        print(f"\n[plot] {out_dir / 'learned_router_v3_compare.png'}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
