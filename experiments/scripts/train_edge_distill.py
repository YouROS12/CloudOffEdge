"""Edge model trained (or fine-tuned) with the cloud ensemble as teacher.

Two uses, same loss (focal loss on labels + temperature-scaled KL to the
ensemble's probabilities, Hinton-style):

1. Offline distillation: train MobileNetV3-L from ImageNet weights on the
   training split with the ensemble's soft labels (same recipe, split, sampler
   and checkpoint selection as experiments/scripts/train_edge_model.py).
     python experiments/scripts/train_edge_distill.py offline --out-dir outputs/edge_model_v3l_kd

2. Closing the loop (pilot): the test split is divided, per class, into a
   "week 1" stream and a "week 2" evaluation half. The deployed edge model
   escalates week-1 images whose margin is below tau; only those come back from
   the cloud. The model is fine-tuned on them (mixed with replayed training
   images) and evaluated on week 2. Variants differ only in the labels the
   escalated images carry:
     none        no update (baseline)
     replay      fine-tune on replayed training images only (control: more training, no new data)
     hard        ensemble's top-1 label
     soft        ensemble's probabilities (distillation)
     soft_conf   distillation on escalated images where the ensemble is confident (p_max >= 0.5)
     oracle      true label (as if the farmer confirmed every escalation)
     python experiments/scripts/train_edge_distill.py loop --out results/edge_loop
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_edge_model as base  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "artifacts"
TAU = 0.138424


class TeacherDataset(Dataset):
    """(image, label, teacher logits or NaN row, weight)."""

    def __init__(self, items, teacher, transform):
        self.items, self.teacher, self.transform = items, teacher, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (path, label), t = self.items[i], self.teacher[i]
        from PIL import Image
        return self.transform(Image.open(path).convert("RGB")), int(label), t


def kd_loss(logits, y, t_logits, focal, alpha: float, T: float):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * focal(y); rows without a teacher use focal only."""
    has_t = ~torch.isnan(t_logits[:, 0])
    loss = focal(logits, y) * (1 - alpha * has_t.float().mean())
    if has_t.any():
        kl = F.kl_div(F.log_softmax(logits[has_t].float() / T, -1), F.softmax(t_logits[has_t] / T, -1),
                      reduction="batchmean") * T * T
        loss = loss + alpha * has_t.float().mean() * kl
    return loss


def train(model, loader, epochs, lr, device, alpha, T, warmup_frac=0.05, eval_fn=None):
    focal = base.FocalLoss(gamma=2.0, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    total = epochs * len(loader)
    warm = max(1, int(total * warmup_frac))
    step, best, best_state, hist = 0, -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, y, t in loader:
            x, y, t = x.to(device, non_blocking=True), y.to(device), t.to(device)
            for g in opt.param_groups:
                g["lr"] = lr * base.cosine_with_warmup(step, total, warm)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
            loss = kd_loss(logits.float(), y, t, focal, alpha, T)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
            step += 1
        row = {"epoch": ep, "loss": tot / max(1, n), "elapsed_s": time.time() - t0}
        if eval_fn:
            row.update(eval_fn(model))
            if row["val_macro_f1"] > best:
                best, best_state = row["val_macro_f1"], {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist.append(row)
        print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict(model, items, tf, device, bs=128):
    model.eval()
    ds = base._ItemsDataset(items, tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    out = []
    for x, _ in dl:
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out)


def margin(logits):
    p = F.softmax(logits, -1).topk(2, -1).values
    return (p[:, 0] - p[:, 1])


def operating(edge_logits, cloud_pred, y, tau=TAU, target=0.95):
    """Edge accuracy; at tau: escalation and delivered accuracy; escalation needed for target x cloud acc."""
    e_pred, m = edge_logits.argmax(-1), margin(edge_logits)
    e_ok, c_ok = (e_pred == y), (cloud_pred == y)
    esc = m < tau
    fin = torch.where(esc, c_ok, e_ok)
    order = torch.argsort(m)  # escalate least-confident first
    n, goal = len(y), target * c_ok.float().mean().item()
    need = None
    for k in range(n + 1):
        sel = torch.zeros(n, dtype=torch.bool)
        sel[order[:k]] = True
        if torch.where(sel, c_ok, e_ok).float().mean().item() >= goal:
            need = k / n
            break
    return {"edge_acc": e_ok.float().mean().item(), "esc_at_tau": esc.float().mean().item(),
            "acc_at_tau": fin.float().mean().item(), "cloud_acc": c_ok.float().mean().item(),
            "esc_for_95": need}


def load_split(seed=1):
    base._add_turboquant_root_to_path()
    from plantwild import load_plantwild
    train_full, test_items, names = load_plantwild()
    soft = torch.load(OUT / "cloud_soft_labels.pt", weights_only=False)
    assert soft["train_labels"].tolist() == [l for _, l in train_full], "teacher/train order mismatch"
    assert soft["test_labels"].tolist() == [l for _, l in test_items], "teacher/test order mismatch"
    return train_full, test_items, names, soft


def offline(args):
    torch.manual_seed(args.seed)
    device = "cuda"
    train_full, test_items, names, soft = load_split()
    idx = {p: i for i, (p, _) in enumerate(train_full)}
    tr, va = base.stratified_train_val_split(train_full, val_frac=0.1, seed=args.seed)
    t_tr = soft["train_logits"][[idx[p] for p, _ in tr]]
    train_tf, eval_tf = base.build_transforms(224)
    dl = DataLoader(TeacherDataset(tr, t_tr, train_tf), batch_size=64, sampler=base.make_balanced_sampler(tr),
                    num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(base._ItemsDataset(va, eval_tf), batch_size=128, num_workers=4, pin_memory=True)
    model = base.build_model("mobilenet_v3_large", len(names)).to(device)
    ev = lambda m: {f"val_{k}": v for k, v in base.evaluate(m, val_dl, device).items()}  # noqa: E731
    hist = train(model, dl, args.epochs, 5e-4, device, args.alpha, args.T, eval_fn=ev)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = soft["test_labels"]
    logits = predict(model, test_items, eval_tf, device)
    res = operating(logits, soft["test_logits"].argmax(-1), y)
    torch.save({"state_dict": model.state_dict(), "arch": "mobilenet_v3_large", "n_classes": len(names),
                "class_names": names, "img_size": 224, "kd": {"alpha": args.alpha, "T": args.T}}, out / "best.pt")
    torch.save(logits, out / "test_logits.pt")
    (out / "result.json").write_text(json.dumps({"history": hist, "test": res}, indent=1))
    print("[test]", res)


def loop(args):
    device = "cuda"
    train_full, test_items, names, soft = load_split()
    y_test, c_logits = soft["test_labels"], soft["test_logits"]
    c_pred, c_conf = c_logits.argmax(-1), F.softmax(c_logits, -1).max(-1).values
    # per-class halves: week 1 (stream) / week 2 (evaluation)
    rng = random.Random(args.seed)
    by = {}
    for i, (_, l) in enumerate(test_items):
        by.setdefault(l, []).append(i)
    w1, w2 = [], []
    for l, ids in by.items():
        rng.shuffle(ids)
        h = len(ids) // 2
        w1 += ids[:h]
        w2 += ids[h:]
    w1, w2 = sorted(w1), sorted(w2)
    _, eval_tf = base.build_transforms(224)
    train_tf, _ = base.build_transforms(224)
    ck = torch.load(OUT / "edge_model_v3l" / "best.pt", map_location="cpu", weights_only=False)

    def fresh():
        m = base.build_model("mobilenet_v3_large", len(names))
        m.load_state_dict(ck["state_dict"])
        return m.to(device)

    m0 = fresh()
    base_logits = predict(m0, test_items, eval_tf, device)
    print(f"[loop] baseline edge accuracy on the full test split: "
          f"{(base_logits.argmax(-1) == y_test).float().mean():.4f} (paper: 0.6932)")
    esc_w1 = [i for i in w1 if margin(base_logits[i:i + 1]).item() < TAU]
    print(f"[loop] week1={len(w1)} week2={len(w2)} escalated in week1={len(esc_w1)} "
          f"(cloud correct on {sum(int(c_pred[i] == y_test[i]) for i in esc_w1)})")
    tr_items, _ = base.stratified_train_val_split(train_full, val_frac=0.1, seed=1)
    nan = torch.full((len(names),), float("nan"))
    results = {"meta": {"n_week1": len(w1), "n_week2": len(w2), "n_escalated_week1": len(esc_w1), "tau": TAU,
                        "epochs": args.epochs, "replay_per_epoch": args.replay, "stream_repeat": args.repeat,
                        "lr": args.lr, "alpha": args.alpha, "T": args.T}}
    w2t = torch.tensor(w2)
    results["none"] = operating(base_logits[w2t], c_pred[w2t], y_test[w2t])
    print("none", results["none"], flush=True)

    for variant in ["replay", "hard", "soft", "soft_conf", "oracle"]:
        torch.manual_seed(args.seed)
        stream_items, stream_t = [], []
        for i in esc_w1:
            p = test_items[i][0]
            if variant == "hard":
                stream_items.append((p, int(c_pred[i]))); stream_t.append(nan)
            elif variant == "soft":
                stream_items.append((p, int(c_pred[i]))); stream_t.append(c_logits[i])
            elif variant == "soft_conf" and c_conf[i] >= 0.5:
                stream_items.append((p, int(c_pred[i]))); stream_t.append(c_logits[i])
            elif variant == "oracle":
                stream_items.append((p, int(y_test[i]))); stream_t.append(nan)
        rs = random.Random(args.seed)
        rep = rs.sample(tr_items, args.replay)
        items = stream_items * args.repeat + rep
        teach = (torch.stack(stream_t) if stream_t else torch.empty(0, len(names))).repeat(args.repeat, 1)
        teach = torch.cat([teach, nan.repeat(len(rep), 1)])
        dl = DataLoader(TeacherDataset(items, teach, train_tf), batch_size=64, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True, drop_last=True)
        m = fresh()
        train(m, dl, args.epochs, args.lr, device, args.alpha, args.T, warmup_frac=0.1)
        lg = predict(m, [test_items[i] for i in w2], eval_tf, device)
        results[variant] = operating(lg, c_pred[w2t], y_test[w2t])
        results[variant]["n_stream"] = len(stream_items)
        print(variant, results[variant], flush=True)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "loop_results.json").write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("offline")
    a.add_argument("--out-dir", default=str(OUT / "edge_model_v3l_kd"))
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--alpha", type=float, default=0.5)
    a.add_argument("--T", type=float, default=2.0)
    a.add_argument("--seed", type=int, default=1)
    b = sub.add_parser("loop")
    b.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "edge_loop"))
    b.add_argument("--epochs", type=int, default=8)
    b.add_argument("--replay", type=int, default=2000)
    b.add_argument("--repeat", type=int, default=4)
    b.add_argument("--lr", type=float, default=1e-4)
    b.add_argument("--alpha", type=float, default=0.5)
    b.add_argument("--T", type=float, default=2.0)
    b.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    offline(args) if args.cmd == "offline" else loop(args)


if __name__ == "__main__":
    main()
