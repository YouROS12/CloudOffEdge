"""Offline analyses added after the external review (no training, no device).

From the saved fp32 edge logits (experiments/scripts/predict_edge.py), the offline cloud
predictions and the two replays' per-image events:

  calibration   temperature scaling and alternative scalar signals, each with
                its own validation-calibrated threshold (R13)
  bootstrap     sampling distribution of the validation-calibrated threshold
                and of its test operating point (R14)
  iso_accuracy  escalation the distilled edge needs to match the original
                model's delivered accuracy (R12)
  macro_f1      delivered macro-F1 of every operating point and replay (R15)
  farms         per-farm always-cloud ceiling in both replays (R14/R15)
  exact_at_95   exact escalation @95 % of cloud accuracy per edge model and for
                the cloud-supervised verifiers, replacing threshold-grid readings
  budgets       top-1 at fixed escalation budgets by signal (exact, per image)

  python experiments/scripts/review_analyses.py -> results/review/analyses.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
OUT = Path(__file__).resolve().parents[2] / "artifacts"
RES = PAPER2 / "results"
TARGET = 0.95
EDGES = {"v3-Small": "edge_predictions", "v3-Large": "edge_v3l_predictions",
         "EfficientNet-B0": "edge_effb0_predictions", "v3-Large distilled": "edge_v3l_kd_predictions"}


def key(path: str, label: str) -> str:
    return label + "/" + path.replace("\\", "/").rsplit("/", 1)[-1]


def cloud(split: str) -> dict:
    with (OUT / f"cloud_predictions_{split}.csv").open(encoding="utf-8") as fh:
        return {key(r["image_path"], r["true_label"]): (r["cloud_pred"], r["true_label"]) for r in csv.DictReader(fh)}


def bundle(stem: str, split: str) -> dict:
    """Edge logits aligned with cloud predictions: logits, y, edge_ok, cloud_ok, class names."""
    d = torch.load(OUT / f"{stem}_{split}.logits.pt", weights_only=False)
    names = list(d["class_names"])
    cl = cloud(split)
    keys = [key(p, t) for p, t in zip(d["image_paths"], d["true_labels"])]
    y = np.array([names.index(t) for t in d["true_labels"]])
    c = np.array([names.index(cl[k][0]) for k in keys])
    lg = d["logits"].float()
    return {"logits": lg, "y": y, "cloud_pred": c, "edge_pred": lg.argmax(-1).numpy(), "names": names, "keys": keys}


# ----------------------------------------------------------------------------- signals
def scores(lg: torch.Tensor, T: float = 1.0) -> dict:
    """Uncertainty scores (higher = escalate first)."""
    p = F.softmax(lg / T, -1)
    top = p.topk(2, -1).values
    z = lg.topk(2, -1).values
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    return {"margin": (1 - (top[:, 0] - top[:, 1])).numpy(), "max_prob": (1 - top[:, 0]).numpy(),
            "entropy": ent.numpy(), "logit_margin": (-(z[:, 0] - z[:, 1])).numpy()}


def fit_temperature(lg: torch.Tensor, y: np.ndarray) -> float:
    logt = torch.zeros(1, requires_grad=True)
    yt = torch.tensor(y)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logt.exp(), yt)
        loss.backward()
        return loss
    opt.step(step)
    return float(logt.exp())


def ece(lg: torch.Tensor, y: np.ndarray, T: float = 1.0, bins: int = 15) -> float:
    p = F.softmax(lg / T, -1)
    conf, pred = p.max(-1)
    ok = (pred.numpy() == y)
    conf = conf.numpy()
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def delivered(u, thr, e_ok, c_ok):
    esc = u >= thr
    return esc, np.where(esc, c_ok, e_ok)


def calibrate(u, e_ok, c_ok, target=TARGET):
    """Smallest escalation set (highest u first) reaching target x cloud accuracy; returns threshold on u."""
    order = np.argsort(-u, kind="stable")
    n, goal = len(u), target * c_ok.mean()
    acc = e_ok.sum()
    if acc / n >= goal:
        return np.inf
    for k, i in enumerate(order, 1):
        acc += int(c_ok[i]) - int(e_ok[i])
        if acc / n >= goal:
            # escalate every image whose score is at least this one's (ties escalate together)
            return u[i]
    return -np.inf


def esc_at(u, e_ok, c_ok, goal_acc):
    order = np.argsort(-u, kind="stable")
    n, acc = len(u), e_ok.sum()
    if acc / n >= goal_acc:
        return 0.0
    for k, i in enumerate(order, 1):
        acc += int(c_ok[i]) - int(e_ok[i])
        if acc / n >= goal_acc:
            return k / n
    return None


def operating_point(u_val, u_test, v, t):
    ev, cv = v["edge_pred"] == v["y"], v["cloud_pred"] == v["y"]
    et, ct = t["edge_pred"] == t["y"], t["cloud_pred"] == t["y"]
    thr = calibrate(u_val, ev, cv)
    esc, fin = delivered(u_test, thr, et, ct)
    useful = (~et) & ct
    return {"test_escalation": float(esc.mean()), "test_acc": float(fin.mean()),
            "test_pct_of_cloud": float(fin.mean() / ct.mean()), "test_reduction_x": float(1 / esc.mean()),
            "test_esc_for_95": esc_at(u_test, et, ct, TARGET * ct.mean()),
            "test_auc_useful": float(roc_auc_score(useful, u_test))}


def calibration_block(stem: str) -> dict:
    v, t = bundle(stem, "val"), bundle(stem, "test")
    T = fit_temperature(v["logits"], v["y"])
    out = {"temperature_fitted_on_val": T,
           "ece_test_raw": ece(t["logits"], t["y"]), "ece_test_scaled": ece(t["logits"], t["y"], T),
           "nll_test_raw": float(F.cross_entropy(t["logits"], torch.tensor(t["y"]))),
           "nll_test_scaled": float(F.cross_entropy(t["logits"] / T, torch.tensor(t["y"]))), "signals": {}}
    sv1, st1 = scores(v["logits"]), scores(t["logits"])
    svT, stT = scores(v["logits"], T), scores(t["logits"], T)
    for name in ("margin", "max_prob", "entropy"):
        out["signals"][name] = operating_point(sv1[name], st1[name], v, t)
        out["signals"][name + "_temperature_scaled"] = operating_point(svT[name], stT[name], v, t)
    out["signals"]["logit_margin"] = operating_point(sv1["logit_margin"], st1["logit_margin"], v, t)
    return out


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_block(stem: str, B: int = 2000, seed: int = 0) -> dict:
    v, t = bundle(stem, "val"), bundle(stem, "test")
    uv, ut = scores(v["logits"])["margin"], scores(t["logits"])["margin"]
    ev, cv = v["edge_pred"] == v["y"], v["cloud_pred"] == v["y"]
    et, ct = t["edge_pred"] == t["y"], t["cloud_pred"] == t["y"]
    rng = np.random.default_rng(seed)
    taus, pct, esc = [], [], []
    for _ in range(B):
        i = rng.integers(0, len(uv), len(uv))
        thr = calibrate(uv[i], ev[i], cv[i])
        e, fin = delivered(ut, thr, et, ct)
        taus.append(1 - thr)
        pct.append(fin.mean() / ct.mean())
        esc.append(e.mean())
    taus, pct, esc = map(np.array, (taus, pct, esc))
    q = lambda a: [float(np.percentile(a, 5)), float(np.median(a)), float(np.percentile(a, 95))]  # noqa: E731
    # conservative variant: the 95th percentile of the bootstrap thresholds (escalates more)
    tau95 = float(np.percentile(taus, 95))
    e95, f95 = delivered(ut, 1 - tau95, et, ct)
    return {"B": B, "tau_p5_med_p95": q(taus), "test_pct_of_cloud_p5_med_p95": q(pct),
            "test_escalation_p5_med_p95": q(esc), "share_meeting_95pct_on_test": float((pct >= TARGET).mean()),
            "conservative_tau_p95": tau95, "conservative_test_escalation": float(e95.mean()),
            "conservative_test_pct_of_cloud": float(f95.mean() / ct.mean()),
            "conservative_test_reduction_x": float(1 / e95.mean())}


# ----------------------------------------------------------------------------- macro-F1, farms, iso-accuracy
def macro_f1(y, pred, labels):
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def events(run: str) -> list[dict]:
    with (RES / run / "events.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verifier_block() -> dict:
    """Exact escalation @95 % and AUC of the three trained verifiers on the test split."""
    from train_verifier import VerifierMLP, build_features
    blob = torch.load(OUT / "edge_v3l_test_emb.pt", weights_only=False)
    cl = cloud("test")
    names = list(blob["class_names"])
    keys = [key(p, names[int(y)]) for p, y in zip(blob["image_paths"], blob["true_labels"])]
    y = blob["true_labels"].numpy()
    c = np.array([names.index(cl[k][0]) for k in keys])
    e_ok, c_ok = blob["edge_pred"].numpy() == y, c == y
    X = build_features(blob["edge_emb"].float(), blob["edge_logits"].float())
    out = {}
    for tag in ("useful", "disagree", "correct"):
        ck = torch.load(OUT / f"verifier_v3l_{tag}.pt", weights_only=False)
        m = VerifierMLP(ck["input_dim"], hidden_dim=ck["hidden_dim"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        with torch.no_grad():
            s = m(X).numpy()
        r = esc_at(s, e_ok, c_ok, TARGET * c_ok.mean())
        k = int(round(r * len(s)))
        sel = np.zeros(len(s), bool)
        sel[np.argsort(-s, kind="stable")[:k]] = True
        out[tag] = {"best_val_auc": ck["best_val_auc"], "test_auc_useful": float(roc_auc_score((~e_ok) & c_ok, s)),
                    "esc_for_95": r, "acc_at_95": float(np.where(sel, c_ok, e_ok).mean())}
    return out


def budgets_block(stem="edge_v3l_predictions") -> dict:
    t = bundle(stem, "test")
    e_ok, c_ok = t["edge_pred"] == t["y"], t["cloud_pred"] == t["y"]
    sc = scores(t["logits"])
    useful = (~e_ok) & c_ok
    n = len(e_ok)
    out = {}
    for b in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        k = int(round(b * n))
        row = {}
        for name in ("max_prob", "margin", "entropy"):
            sel = np.zeros(n, bool)
            sel[np.argsort(-sc[name], kind="stable")[:k]] = True
            row[name] = float(np.where(sel, c_ok, e_ok).mean())
        row["oracle"] = float((e_ok.sum() + min(k, useful.sum())) / n)
        out[f"{int(round(b * 100))}"] = row
    return out


def main() -> None:
    rep = {}
    rep["verifiers"] = verifier_block()
    rep["budgets"] = budgets_block()
    rep["exact_at_95"] = {}
    for arch, stem in EDGES.items():
        t = bundle(stem, "test")
        et, ct = t["edge_pred"] == t["y"], t["cloud_pred"] == t["y"]
        u = scores(t["logits"])["margin"]
        r = esc_at(u, et, ct, TARGET * ct.mean())
        sel = np.zeros(len(u), bool)
        sel[np.argsort(-u, kind="stable")[:int(round(r * len(u)))]] = True
        rep["exact_at_95"][arch] = {"edge_acc": float(et.mean()), "cloud_acc": float(ct.mean()), "esc_for_95": r,
                                    "acc_at_95": float(np.where(sel, ct, et).mean())}

    rep["calibration"] = {"v3-Large": calibration_block("edge_v3l_predictions"),
                          "v3-Large distilled": calibration_block("edge_v3l_kd_predictions")}
    rep["bootstrap"] = {"v3-Large": bootstrap_block("edge_v3l_predictions"),
                        "v3-Large distilled": bootstrap_block("edge_v3l_kd_predictions")}

    # iso-accuracy: distilled escalation needed to match the original model's delivered accuracy
    kd = bundle("edge_v3l_kd_predictions", "test")
    u = scores(kd["logits"])["margin"]
    e_ok, c_ok = kd["edge_pred"] == kd["y"], kd["cloud_pred"] == kd["y"]
    calib = json.loads((RES / "threshold_calibration.json").read_text(encoding="utf-8"))
    rep["iso_accuracy"] = {}
    for tag, goal in {"offline_original_at_tau": calib["test"]["acc"],
                      "replay1_delivered": json.loads((RES / "fleet_run" / "summary.json").read_text())["overall"]["final_acc"]}.items():
        r = esc_at(u, e_ok, c_ok, goal)
        rep["iso_accuracy"][tag] = {"goal_acc": goal, "distilled_escalation": r, "distilled_reduction_x": 1 / r}

    # macro-F1 (offline operating points; labels = all 89 classes)
    rep["macro_f1"] = {}
    labels = list(range(89))
    for arch, stem, tau in (("v3-Large", "edge_v3l_predictions", calib["tau"]),
                            ("v3-Large distilled", "edge_v3l_kd_predictions",
                             json.loads((RES / "threshold_calibration_kd.json").read_text())["tau"])):
        t = bundle(stem, "test")
        m = 1 - scores(t["logits"])["margin"]
        fin = np.where(m < tau, t["cloud_pred"], t["edge_pred"])
        rep["macro_f1"][arch] = {"edge_only": macro_f1(t["y"], t["edge_pred"], labels),
                                 "at_tau": macro_f1(t["y"], fin, labels),
                                 "always_cloud": macro_f1(t["y"], t["cloud_pred"], labels)}
    names = bundle("edge_v3l_predictions", "test")["names"]
    cl_test = cloud("test")
    rep["farms"] = {}
    for run in ("fleet_run", "fleet_run_improved"):
        ev = events(run)
        y = [names.index(r["true_label"]) for r in ev]
        fin = [names.index(r["final_label"]) for r in ev]
        edge = [names.index(r["edge_label"]) for r in ev]
        off_cloud = [names.index(cl_test[r["image"]][0]) for r in ev]
        rep["macro_f1"][run] = {"delivered": macro_f1(y, fin, labels), "edge_only": macro_f1(y, edge, labels),
                                "always_cloud_offline": macro_f1(y, off_cloud, labels)}
        farms = {}
        for f in sorted({r["farm"] for r in ev}):
            idx = [i for i, r in enumerate(ev) if r["farm"] == f]
            n = len(idx)
            cloud_acc = sum(off_cloud[i] == y[i] for i in idx) / n
            acc = sum(fin[i] == y[i] for i in idx) / n
            farms[f] = {"n": n, "delivered_acc": acc, "always_cloud_offline_acc": cloud_acc,
                        "edge_acc": sum(edge[i] == y[i] for i in idx) / n, "pct_of_cloud": acc / cloud_acc}
        rep["farms"][run] = farms

    out = RES / "review" / "analyses.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
