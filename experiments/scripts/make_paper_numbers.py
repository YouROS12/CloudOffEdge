"""Write paper/manuscript/numbers.tex: one LaTeX macro per measured value.

Every number the hardware run produces enters the manuscript through these
macros, so nothing is transcribed by hand. Missing inputs leave the macro as
an amber \\res{pending} box, so the paper always compiles.

  python experiments/scripts/make_paper_numbers.py --run results/fleet_run \
      --verify results/trt_verify_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
FARMS = {"A": "orchard", "B": "solanaceae", "C": "field crops", "D": "market garden"}
WORD = {"A": "A", "B": "B", "C": "C", "D": "D"}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\,\\%"


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(PAPER2 / "results" / "fleet_run"))
    ap.add_argument("--verify", default=str(PAPER2 / "results" / "trt_verify_report.json"))
    ap.add_argument("--out", default=str(PAPER2 / "paper" / "numbers.tex"))
    ap.add_argument("--offline-cloud", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "cloud_predictions_test.csv"),
                    help="offline (desktop) cloud predictions, to measure the Nano's agreement on every escalated image")
    ap.add_argument("--images-per-day", type=int, default=100)
    ap.add_argument("--uplink-mbps", type=float, default=1.0)
    args = ap.parse_args()

    m: dict[str, str] = {}

    # ---- TensorRT parity (verify_report.json from cloud/build_trt.py --verify-only)
    vp = Path(args.verify)
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        names = {"dinov2_vitl14": "DinoTwo", "dinov3_vitl16": "DinoThree", "clip_vitl14": "Clip"}
        for k, tag in names.items():
            m[f"RCosMin{tag}"] = f"{v[k]['cos_min']:.4f}"
            m[f"RCosMean{tag}"] = f"{v[k]['cos_mean']:.4f}"
            m[f"RMs{tag}"] = f"{v[k]['ms']:,.0f}".replace(",", "\\,")
        m["RCosMinAll"] = f"{min(v[k]['cos_min'] for k in names):.4f}"
        n_ref = 24
        m["RAgree"] = f"{round(v['top1_agree'] * n_ref)}/{n_ref}"

    # ---- engines alone, fed workstation-preprocessed tensors
    ve = vp.with_name("trt_verify_engine_only.json")
    if ve.exists():
        v = json.loads(ve.read_text(encoding="utf-8"))
        for k, tag in {"dinov2_vitl14": "DinoTwo", "dinov3_vitl16": "DinoThree", "clip_vitl14": "Clip"}.items():
            m[f"REngCosMin{tag}"] = f"{v[k]['cos_min']:.4f}"
            m[f"REngCosMean{tag}"] = f"{v[k]['cos_mean']:.4f}"
        m["REngAgree"] = f"{round(v['top1_agree'] * 24)}/24"

    # ---- fleet run
    run = Path(args.run)
    if (run / "summary.json").exists():
        s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        with (run / "events.csv").open(encoding="utf-8") as fh:
            ev = list(csv.DictReader(fh))
        o = s["overall"]
        esc = [r for r in ev if int(r["route_decision"]) and int(r["cloud_ok"])]
        m["RNImg"] = f"{o['n']:,}".replace(",", "\\,")
        m["RNEsc"] = f"{sum(int(r['route_decision']) for r in ev):,}".replace(",", "\\,")
        m["REscRate"] = pct(o["route_decision_rate"])
        m["RUpMB"] = f"{o['upload_mb']:.1f}"
        m["RCloudOnlyMB"] = f"{o['cloud_only_mb']:.1f}"
        m["RReduction"] = f"{o['bandwidth_reduction_x']:.1f}"
        m["RKbPerImg"] = f"{o['kb_per_image']:.1f}\\,KB"
        m["RKbPerImgNum"] = f"{o['kb_per_image']:.1f}"
        m["RCloudOnlyKbPerImg"] = f"{o['cloud_only_kb_per_image']:.1f}"
        m["RAcc"] = pct(o["final_acc"])
        m["REdgeAcc"] = pct(o["edge_acc"])
        m["RCloudFail"] = str(o["cloud_failures"])
        m["RMaxPerHour"] = str(s["cloud_requests_per_sim_hour"]["max"])
        m["RAnswerMs"] = f"{med([float(r['latency_ms']) for r in ev]):.0f}"
        lat = sorted(float(r["latency_ms"]) for r in ev)
        m["RAnswerPNinetyFiveMs"] = f"{lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.0f}"
        m["RAnswerEscMs"] = f"{med([float(r['latency_ms']) for r in ev if int(r['route_decision'])]):.0f}"
        # verifier break-even (Eq. 3) at the deployed payload: M / S, M = 2 x (1280 + 89) bytes
        S, M = o["cloud_only_kb_per_image"], 2 * (1280 + 89) / 1024
        m["RVerifierBreakEven"] = f"{100 * M / S:.1f}"
        # offline bandwidth (Eq. 3) with S = the measured mean payload, so offline and deployment agree in units
        m["RSKb"] = f"{S:.1f}"
        res = PAPER2 / "results"
        sw = list(csv.DictReader((res / "option_A_3arch" / "three_arch_sweep.csv").open(encoding="utf-8")))
        ver = list(csv.DictReader((res / "option_B_verifier_disagree" / "verifier_sweep.csv").open(encoding="utf-8")))

        def first_rate(rows):  # smallest escalation rate reaching 95 % of cloud accuracy on the test curve
            return min(float(r["offload_rate"]) for r in rows if float(r["accuracy"]) >= 0.7617)
        r_cross = first_rate([r for r in sw if r["arch"] == "v3-Large" and r["signal"] == "margin"])
        r_ver = first_rate([r for r in ver if r["signal"] == "verifier"])
        m["ROffKbCross"] = f"{r_cross * S:.1f}"
        m["ROffKbVer"] = f"{M + r_ver * S:.1f}"
        edge_csv = Path(args.offline_cloud).with_name("edge_v3l_predictions_test.csv")
        tau = json.loads((res / "threshold_calibration.json").read_text(encoding="utf-8")).get("tau", 0.138424) \
            if (res / "threshold_calibration.json").exists() else 0.138424
        if edge_csv.exists():
            with edge_csv.open(encoding="utf-8") as fh:
                margins = [float(r["edge_margin"]) for r in csv.DictReader(fh)]
            m["ROffKbOp"] = f"{sum(x < tau for x in margins) / len(margins) * S:.1f}"
        m["REdgeMs"] = f"{med([float(r['edge_ms']) for r in ev]):.0f}"
        m["RNetMs"] = f"{med([float(r['network_ms']) for r in esc]):.0f}"
        m["RCloudS"] = f"{med([float(r['cloud_ms']) for r in esc]) / 1000:.1f}"
        m["RPayloadKb"] = f"{med([float(r['bandwidth_kb']) for r in esc]):.1f}"
        m["RDelayMean"] = f"{statistics.mean(float(r['answer_delay_h']) for r in esc):.1f}" if esc else "--"
        m["RDelayMax"] = f"{max(float(r['answer_delay_h']) for r in esc):.0f}" if esc else "--"
        m["RWallMin"] = f"{s.get('wall_clock_minutes', float('nan')):.0f}"
        oc = Path(args.offline_cloud)
        if oc.exists() and esc:
            with oc.open(encoding="utf-8") as fh:
                off = {r["true_label"] + "/" + r["image_path"].replace("\\", "/").rsplit("/", 1)[-1]: r["cloud_pred"]
                       for r in csv.DictReader(fh)}
            pairs = [(r["final_label"], off.get(r["image"])) for r in esc if off.get(r["image"])]
            agree = sum(a == b for a, b in pairs)
            m["RCloudAgreeFleet"] = f"{agree:,}/{len(pairs):,}".replace(",", "\\,")
            m["RCloudAgreeFleetPct"] = pct(agree / len(pairs))
            m["RCloudAccEsc"] = pct(sum(r["final_label"] == r["true_label"] for r in esc) / len(esc))
            m["ROfflineCloudAccEsc"] = pct(sum(off.get(r["image"]) == r["true_label"] for r in esc) / len(esc))
        fl = run / "flushes.csv"
        if fl.exists():
            with fl.open(encoding="utf-8") as fh:
                flushes = list(csv.DictReader(fh))
            m["RNFlush"] = str(len(flushes))
            m["RLoadS"] = f"{med([float(f['load_ms_total']) for f in flushes]) / 1000:.0f}"
            m["RBatchMed"] = f"{med([float(f['n']) for f in flushes]):.0f}"
            for k, tag in {"dinov2_vitl14": "DinoTwo", "dinov3_vitl16": "DinoThree", "clip_vitl14": "Clip"}.items():
                m[f"RLoad{tag}S"] = f"{med([json.loads(f['load_ms'])[k] for f in flushes]) / 1000:.0f}"
        for f, d in s["per_farm"].items():
            t = WORD[f]
            m[f"RFarm{t}Esc"] = pct(d["route_decision_rate"])
            m[f"RFarm{t}Up"] = f"{d['upload_mb']:.1f}"
            m[f"RFarm{t}Cloud"] = f"{d['cloud_only_mb']:.1f}"
            m[f"RFarm{t}Red"] = f"{d['bandwidth_reduction_x']:.1f}"
            m[f"RFarm{t}Acc"] = pct(d["final_acc"])
            m[f"RFarm{t}EdgeAcc"] = pct(d["edge_acc"])
        rates = {f: d["route_decision_rate"] for f, d in s["per_farm"].items()}
        fmax, fmin = max(rates, key=rates.get), min(rates, key=rates.get)
        m["RFarmMaxName"] = f"farm~{fmax} ({FARMS[fmax]})"
        m["RFarmMaxEsc"] = pct(rates[fmax], 0)
        m["RFarmMinName"] = f"farm~{fmin} ({FARMS[fmin]})"
        m["RFarmMinEsc"] = pct(rates[fmin], 0)
        # derived: monthly volume for a farm capturing N images/day, uplink time per escalation
        kb, kb_c = o["kb_per_image"], o["cloud_only_kb_per_image"]
        m["RImgPerDay"] = str(args.images_per_day)
        m["RMonthMB"] = f"{kb * args.images_per_day * 30 / 1024:.0f}"
        m["RMonthCloudMB"] = f"{kb_c * args.images_per_day * 30 / 1024:.0f}"
        m["RUplinkMbps"] = f"{args.uplink_mbps:g}"
        m["RUplinkS"] = f"{float(m['RPayloadKb']) * 1024 * 8 / (args.uplink_mbps * 1e6):.1f}"

    # ---- activation ranges (experiments/scripts/profile_activation_ranges.py)
    res = PAPER2 / "results"
    ar = res / "activation_ranges.json"
    if ar.exists():
        a = json.loads(ar.read_text(encoding="utf-8"))
        for k, tag in {"dinov2_vitl14": "DinoTwo", "dinov3_vitl16": "DinoThree", "clip_vitl14": "Clip"}.items():
            m[f"RAct{tag}"] = f"{a['backbones'][k]['residual_max']:,.0f}".replace(",", "\\,")
        m["RActNImg"] = str(a["n_images"])

    # ---- memory and power of a batched flush on the Nano (experiments/scripts/measure_nano_flush.py)
    pf = res / "nano_flush_profile" / "runs.json"
    if pf.exists() and json.loads(pf.read_text(encoding="utf-8"))["runs"]:
        sys_path = str(PAPER2 / "experiments" / "scripts")
        import sys as _sys
        _sys.path.insert(0, sys_path)
        from make_paper_figures import flush_profile  # same computation as the figure
        runs = flush_profile(res / "nano_flush_profile")["runs"]
        mean = lambda k: statistics.mean(r[k] for r in runs)  # noqa: E731
        m["RNanoRuns"] = str(len(runs))
        m["RNanoBatchN"] = str(runs[0]["n"])
        m["RNanoIdleW"] = f"{mean('p_idle_w'):.1f}"
        m["RNanoMeanW"] = f"{mean('p_mean_w'):.1f}"
        m["RNanoMaxW"] = f"{max(r['p_max_w'] for r in runs):.1f}"
        m["RNanoEImgJ"] = f"{mean('e_per_img_j'):.0f}"
        m["RNanoEIncrImgJ"] = f"{mean('e_incr_per_img_j'):.0f}"
        m["RNanoWhPerThousand"] = f"{mean('e_per_img_j') * 1000 / 3600:.0f}"
        m["RNanoPeakGB"] = f"{max(r['ram_peak_mb'] for r in runs) / 1024:.2f}"
        m["RNanoIdleGB"] = f"{mean('ram_idle_mb') / 1024:.2f}"
        m["RNanoRiseGB"] = f"{statistics.mean((r['ram_peak_mb'] - r['ram_idle_mb']) for r in runs) / 1024:.2f}"
        m["RNanoTotalGB"] = f"{runs[0]['ram_total_mb'] / 1024:.2f}"
        m["RNanoFlushMin"] = f"{mean('dur_s') / 60:.1f}"

    # ---- the ensemble as teacher (experiments/scripts/train_edge_distill.py, eval_kd_operating_point.py)
    kd = res / "edge_loop" / "kd_operating_point.json"
    if kd.exists():
        k = json.loads(kd.read_text(encoding="utf-8"))
        for tag, key in (("Base", "baseline"), ("Kd", "distilled")):
            d = k[key]
            m[f"R{tag}EdgeAcc"] = f"{100 * d['test_edge_acc']:.1f}"
            m[f"R{tag}Tau"] = f"{d['tau']:.3f}"
            m[f"R{tag}Esc"] = f"{100 * d['test_escalation']:.1f}"
            m[f"R{tag}Acc"] = f"{100 * d['test_acc']:.1f}"
            m[f"R{tag}Pct"] = f"{100 * d['test_pct_of_cloud']:.1f}"
            m[f"R{tag}Red"] = f"{d['test_reduction_x']:.1f}"
            m[f"R{tag}EscNF"] = f"{100 * d['test_esc_for_95']:.1f}"
    lp = res / "edge_loop" / "loop_results.json"
    if lp.exists():
        L = json.loads(lp.read_text(encoding="utf-8"))
        m["RLoopNWeek"] = f"{L['meta']['n_week1']:,}".replace(",", "\\,")
        m["RLoopNEsc"] = str(L["meta"]["n_escalated_week1"])
        for tag, key in (("None", "none"), ("Replay", "replay"), ("Hard", "hard"), ("Soft", "soft"),
                         ("SoftConf", "soft_conf"), ("Oracle", "oracle")):
            m[f"RLoop{tag}Edge"] = f"{100 * L[key]['edge_acc']:.1f}"
            m[f"RLoop{tag}Esc"] = f"{100 * L[key]['esc_for_95']:.1f}"
        m["RLoopConfN"] = str(L["soft_conf"]["n_stream"])

    # ---- edge energy (experiments/scripts/measure_pi_energy.py) and system energy per diagnosis
    pe = res / "pi_energy" / "run.json"
    if pe.exists() and "RNanoEImgJ" in m:
        p = json.loads(pe.read_text(encoding="utf-8"))
        m["RPiIdleW"] = f"{p['p_idle_w']:.1f}"
        m["RPiLoadW"] = f"{p['p_load_mean_w']:.1f}"
        m["RPiEImgJ"] = f"{p['energy_per_image_j']:.2f}"
        m["RPiEIncrJ"] = f"{p['incremental_per_image_j']:.2f}"
        m["RPiN"] = str(p["n"])
        r_esc = o["route_decision_rate"]
        e_nano = statistics.mean(r["e_per_img_j"] for r in runs)
        e_sys = p["energy_per_image_j"] + r_esc * e_nano
        m["RSysEJ"] = f"{e_sys:.1f}"
        m["RSysCloudEJ"] = f"{e_nano:.0f}"
        m["RSysERatio"] = f"{e_nano / e_sys:.1f}"

    # ---- CLIP in fp16 on the Nano (jetson_deploy/cloud/rebuild_clip16_check.sh)
    cf = res / "clip_fp16_check.json"
    if cf.exists():
        c = json.loads(cf.read_text(encoding="utf-8"))
        m["RClipSixteenMs"] = f"{c['fp16']['ms_per_img']:,}".replace(",", "\\,")
        m["RClipThirtyTwoMs"] = f"{c['fp32_deployed']['ms_per_img']:,}".replace(",", "\\,")
        m["RClipSixteenLoadS"] = f"{c['fp16']['load_s_verify']:.0f}"
        m["RClipThirtyTwoLoadS"] = f"{c['fp32_deployed']['load_s_verify']:.0f}"
        m["RClipSixteenCos"] = f"{c['fp16_vs_fp32_engines_same_inputs']['cos_min']:.4f}"

    # ---- improved configuration replay (distilled edge, fp16 CLIP): results/fleet_run_improved
    ri = res / "fleet_run_improved"
    if (ri / "summary.json").exists():
        si = json.loads((ri / "summary.json").read_text(encoding="utf-8"))
        oi = si["overall"]
        with (ri / "events.csv").open(encoding="utf-8") as fh:
            evi = list(csv.DictReader(fh))
        esci = [r for r in evi if int(r["route_decision"]) and int(r["cloud_ok"])]
        m["RImpNEsc"] = f"{sum(int(r['route_decision']) for r in evi):,}".replace(",", "\\,")
        m["RImpEscRate"] = pct(oi["route_decision_rate"])
        m["RImpUpMB"] = f"{oi['upload_mb']:.1f}"
        m["RImpCloudOnlyMB"] = f"{oi['cloud_only_mb']:.1f}"
        m["RImpReduction"] = f"{oi['bandwidth_reduction_x']:.1f}"
        m["RImpKbPerImg"] = f"{oi['kb_per_image']:.1f}\\,KB"
        m["RImpAcc"] = pct(oi["final_acc"])
        m["RImpEdgeAcc"] = pct(oi["edge_acc"])
        m["RImpCloudFail"] = str(oi["cloud_failures"])
        m["RImpAnswerMs"] = f"{med([float(r['latency_ms']) for r in evi]):.0f}"
        m["RImpCloudS"] = f"{med([float(r['cloud_ms']) for r in esci]) / 1000:.1f}"
        m["RImpWallMin"] = f"{si.get('wall_clock_minutes', float('nan')):.0f}"
        m["RImpDelayMax"] = f"{max(float(r['answer_delay_h']) for r in esci):.0f}"
        fli = ri / "flushes.csv"
        if fli.exists():
            fl_i = list(csv.DictReader(fli.open(encoding="utf-8")))
            m["RImpNFlush"] = str(len(fl_i))
            m["RImpLoadS"] = f"{med([float(f['load_ms_total']) for f in fl_i]) / 1000:.0f}"
            m["RImpBatchMed"] = f"{med([float(f['n']) for f in fl_i]):.0f}"
        oc = Path(args.offline_cloud)
        if oc.exists() and esci:
            with oc.open(encoding="utf-8") as fh:
                off = {r["true_label"] + "/" + r["image_path"].replace("\\", "/").rsplit("/", 1)[-1]: r["cloud_pred"]
                       for r in csv.DictReader(fh)}
            pairs = [(r["final_label"], off.get(r["image"])) for r in esci if off.get(r["image"])]
            m["RImpCloudAgree"] = f"{sum(a == b for a, b in pairs):,}/{len(pairs):,}".replace(",", "\\,")
        for f, d in si["per_farm"].items():
            m[f"RImpFarm{WORD[f]}Esc"] = pct(d["route_decision_rate"])
            m[f"RImpFarm{WORD[f]}Red"] = f"{d['bandwidth_reduction_x']:.1f}"
            m[f"RImpFarm{WORD[f]}Acc"] = pct(d["final_acc"])
        pf2 = res / "nano_flush_profile_clip16" / "runs.json"
        if pf2.exists() and json.loads(pf2.read_text(encoding="utf-8"))["runs"] and pe.exists():
            from make_paper_figures import flush_profile
            runs2 = flush_profile(res / "nano_flush_profile_clip16")["runs"]
            e2 = statistics.mean(r["e_per_img_j"] for r in runs2)
            m["RNanoTwoEImgJ"] = f"{e2:.0f}"
            m["RNanoTwoMeanW"] = f"{statistics.mean(r['p_mean_w'] for r in runs2):.1f}"
            m["RNanoTwoFlushMin"] = f"{statistics.mean(r['dur_s'] for r in runs2) / 60:.1f}"
            m["RNanoTwoPeakGB"] = f"{max(r['ram_peak_mb'] for r in runs2) / 1024:.2f}"
            e_sys2 = json.loads(pe.read_text(encoding="utf-8"))["energy_per_image_j"] + oi["route_decision_rate"] * e2
            m["RImpSysEJ"] = f"{e_sys2:.1f}"
            m["RImpSysERatio"] = f"{e2 / e_sys2:.1f}"
        kdc = res / "threshold_calibration_kd.json"
        if kdc.exists():
            kc = json.loads(kdc.read_text(encoding="utf-8"))
            m["RImpTau"] = f"{kc['tau']:.3f}"
            m["RImpOffEscRate"] = pct(kc["test"]["escalation"])

    names = ["RCosMinDinoTwo", "RCosMeanDinoTwo", "RCosMinDinoThree", "RCosMeanDinoThree", "RCosMinClip",
             "RCosMeanClip", "RMsDinoTwo", "RMsDinoThree", "RMsClip", "RCosMinAll", "RAgree",
             "RNImg", "RNEsc", "REscRate", "RUpMB", "RCloudOnlyMB", "RReduction", "RKbPerImg", "RKbPerImgNum",
             "RCloudOnlyKbPerImg", "RAcc", "REdgeAcc", "RCloudFail", "RMaxPerHour", "RAnswerMs", "REdgeMs",
             "RNetMs", "RCloudS", "RPayloadKb", "RDelayMean", "RDelayMax", "RWallMin", "RNFlush", "RLoadS",
             "RAnswerPNinetyFiveMs", "RAnswerEscMs", "RVerifierBreakEven",
             "RSKb", "ROffKbOp", "ROffKbCross", "ROffKbVer",
             "RActDinoTwo", "RActDinoThree", "RActClip", "RActNImg",
             "RNanoRuns", "RNanoBatchN", "RNanoIdleW", "RNanoMeanW", "RNanoMaxW", "RNanoEImgJ",
             "RNanoEIncrImgJ", "RNanoWhPerThousand", "RNanoPeakGB", "RNanoIdleGB", "RNanoRiseGB",
             "RNanoTotalGB", "RNanoFlushMin",
             *[f"R{t}{k}" for t in ("Base", "Kd") for k in ("EdgeAcc", "Tau", "Esc", "Acc", "Pct", "Red", "EscNF")],
             "RLoopNWeek", "RLoopNEsc", "RLoopConfN",
             *[f"RLoop{t}{k}" for t in ("None", "Replay", "Hard", "Soft", "SoftConf", "Oracle") for k in ("Edge", "Esc")],
             "RPiIdleW", "RPiLoadW", "RPiEImgJ", "RPiEIncrJ", "RPiN", "RSysEJ", "RSysCloudEJ", "RSysERatio",
             "RClipSixteenMs", "RClipThirtyTwoMs", "RClipSixteenLoadS", "RClipThirtyTwoLoadS", "RClipSixteenCos",
             "RImpNEsc", "RImpEscRate", "RImpUpMB", "RImpCloudOnlyMB", "RImpReduction", "RImpKbPerImg", "RImpAcc",
             "RImpEdgeAcc", "RImpCloudFail", "RImpAnswerMs", "RImpCloudS", "RImpWallMin", "RImpDelayMax",
             "RImpNFlush", "RImpLoadS", "RImpBatchMed", "RImpCloudAgree", "RImpTau", "RImpOffEscRate",
             "RNanoTwoEImgJ", "RNanoTwoMeanW", "RNanoTwoFlushMin", "RNanoTwoPeakGB", "RImpSysEJ", "RImpSysERatio",
             *[f"RImpFarm{t}{k}" for t in "ABCD" for k in ("Esc", "Red", "Acc")],
             "RLoadDinoTwoS", "RLoadDinoThreeS", "RLoadClipS", "RBatchMed", "RFarmMaxName", "RFarmMaxEsc", "RFarmMinName", "RFarmMinEsc", "RImgPerDay",
             "RMonthMB", "RMonthCloudMB", "RUplinkMbps", "RUplinkS",
             "RCloudAgreeFleet", "RCloudAgreeFleetPct", "RCloudAccEsc", "ROfflineCloudAccEsc",
             "REngCosMinDinoTwo", "REngCosMeanDinoTwo", "REngCosMinDinoThree", "REngCosMeanDinoThree",
             "REngCosMinClip", "REngCosMeanClip", "REngAgree"]
    names += [f"RFarm{t}{k}" for t in "ABCD" for k in ("Esc", "Up", "Cloud", "Red", "Acc", "EdgeAcc")]
    lines = ["% Generated by experiments/scripts/make_paper_numbers.py -- do not edit by hand.",
             f"% run: {Path(__import__('os').path.relpath(run, PAPER2)).as_posix()}  "
             f"verify: {Path(__import__('os').path.relpath(vp, PAPER2)).as_posix()}"]
    for n in names:
        val = m.get(n, "\\res{pending}")
        lines.append(f"\\newcommand{{\\{n}}}{{{val}\\xspace}}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {sum(n in m for n in names)}/{len(names)} values measured")


if __name__ == "__main__":
    main()
