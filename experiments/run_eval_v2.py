"""

Tactic-level: per-tactic P/R/F1
Chain-level: vs reference attack flow (LCS, step coverage, order accuracy)
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MITRE_CSV  = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"
sys.path.insert(0, str(ROOT))

from pipeline.evaluator import load_ground_truth
from experiments.run_eval import (
    load_tactic_map, resolve_tactic, patch_candidate_tactics,
    tid_match, evaluate_ttp_lenient,
)
from experiments.chain_align import evaluate_chain_alignment, get_flow


def main(min_conf: float = 0.0):
    tm = load_tactic_map(MITRE_CSV)

    eval_results = []
    per_scen_rows = []

    for ann in sorted(OUTPUT_DIR.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt:
            continue
        with open(ann,"r",encoding="utf-8") as f: ann_data = json.load(f)
        scenario = ann_data.get("scenario", ann.parent.name)
        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")

        rec = {"scenario": scenario}

        # TTP eval (per-group)
        if ttp_fp.exists():
            with open(ttp_fp,"r",encoding="utf-8") as f: ttp = json.load(f)
            patch_candidate_tactics(ttp, tm)
            if min_conf > 0:
                conf_map = {g["group_id"]: float(g.get("confidence") or 0)
                            for g in ann_data.get("groups", [])}
                ttp = [r for r in ttp if conf_map.get(r["group_id"], 0) >= min_conf]
            rec["ttp"] = evaluate_ttp_lenient(gt, ttp)

        # Chain alignment
        if vit_fp.exists():
            with open(vit_fp,"r",encoding="utf-8") as f: vit = json.load(f)
            rec["chain"] = evaluate_chain_alignment(scenario, vit)

        eval_results.append(rec)

        t = rec.get("ttp", {})
        c = rec.get("chain", {})
        per_scen_rows.append({
            "scenario": scenario[:55],
            "n_tp_groups": t.get("n", 0),
            "h@1":     round(t.get("hit_at_1", 0), 3),
            "h@5":     round(t.get("hit_at_5", 0), 3),
            "mrr":     round(t.get("mrr", 0), 3),
            "h@1_l":   round(t.get("hit_at_1_lenient", 0), 3),
            "h@5_l":   round(t.get("hit_at_5_lenient", 0), 3),
            "mrr_l":   round(t.get("mrr_lenient", 0), 3),
            "ref_steps":  c.get("ref_steps"),
            "pred_steps": c.get("pred_steps"),
            "step_cov":   c.get("step_coverage", 0),
            "tac_jacc":   c.get("tactic_jaccard", 0),
            "tac_lcs":    c.get("tactic_lcs_norm", 0),
            "tech_lcs":   c.get("technique_lcs_norm", 0),
            "order_acc":  c.get("order_accuracy", 0),
        })

    # Aggregate
    def _avg(rows, key):
        vals = [r.get(key, 0) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0

    ttps   = [r["ttp"]   for r in eval_results if r.get("ttp",{}).get("n",0) > 0]
    chains = [r["chain"] for r in eval_results if "chain" in r and "error" not in r["chain"]]

    print()
    print("═" * 100)
    print(f"  AGGREGATE  (min_confidence={min_conf})")
    print("═" * 100)
    print(f"  scenarios={len(eval_results)}  TP-groups-evaluated={sum(t['n'] for t in ttps)}  "
          f"flows={len(chains)}")
    print()
    print(f"  TTP strict  : H@1={_avg(ttps,'hit_at_1'):.4f}  "
          f"H@5={_avg(ttps,'hit_at_5'):.4f}  MRR={_avg(ttps,'mrr'):.4f}")
    print(f"  TTP lenient : H@1={_avg(ttps,'hit_at_1_lenient'):.4f}  "
          f"H@5={_avg(ttps,'hit_at_5_lenient'):.4f}  "
          f"MRR={_avg(ttps,'mrr_lenient'):.4f}")
    print()
    print(f"  Chain align : step_cov={_avg(chains,'step_coverage'):.4f}  "
          f"tactic_jacc={_avg(chains,'tactic_jaccard'):.4f}  "
          f"tactic_lcs={_avg(chains,'tactic_lcs_norm'):.4f}")
    print(f"              : technique_lcs={_avg(chains,'technique_lcs_norm'):.4f}  "
          f"order_acc={_avg(chains,'order_accuracy'):.4f}")
    print()

    # Per-tactic breakdown (from TTP details)
    tactic_tp = Counter(); tactic_fp = Counter(); tactic_fn = Counter()
    for r in eval_results:
        for d in r.get("ttp", {}).get("details", []):
            gt_t = d["gt_tactic"]; pred_t = d.get("pred_tactic","")
            if gt_t == pred_t: tactic_tp[gt_t] += 1
            else:
                tactic_fn[gt_t] += 1
                tactic_fp[pred_t] += 1
    all_tactics = sorted(set(tactic_tp) | set(tactic_fp) | set(tactic_fn))
    print(f"  PER-TACTIC")
    print(f"    {'Tactic':<28s} {'P':>5s} {'R':>5s} {'F1':>5s}  TP  FP  FN")
    for t in all_tactics:
        if not t: continue
        tp, fp, fn = tactic_tp[t], tactic_fp[t], tactic_fn[t]
        p = tp/(tp+fp) if (tp+fp) else 0
        r = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        print(f"    {t:<28s} {p:5.2f} {r:5.2f} {f1:5.2f}  {tp:>3d} {fp:>3d} {fn:>3d}")
    print()

    # Per-scenario table
    print("─" * 145)
    print(f"  {'scenario':<55s} {'nTP':>4s}  "
          f"{'H@1':>4s} {'H@5':>4s} {'MRR':>4s} {'H1L':>4s} {'H5L':>4s}  "
          f"{'rfs':>3s} {'pds':>3s} {'cov':>4s} {'Tjc':>4s} {'Tls':>4s} {'Hls':>4s} {'ord':>4s}")
    print("─" * 145)
    for row in per_scen_rows:
        print(f"  {row['scenario']:<55s} {row['n_tp_groups']:>4d}  "
              f"{row['h@1']:>4.2f} {row['h@5']:>4.2f} {row['mrr']:>4.2f} "
              f"{row['h@1_l']:>4.2f} {row['h@5_l']:>4.2f}  "
              f"{(row['ref_steps'] or 0):>3d} {(row['pred_steps'] or 0):>3d} "
              f"{row['step_cov']:>4.2f} {row['tac_jacc']:>4.2f} "
              f"{row['tac_lcs']:>4.2f} {row['tech_lcs']:>4.2f} "
              f"{row['order_acc']:>4.2f}")
    print("─" * 145)

    # Save
    out_json = OUTPUT_DIR / "eval_v2_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    out_csv = OUTPUT_DIR / "eval_v2_per_scenario.csv"
    if per_scen_rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_scen_rows[0].keys()))
            w.writeheader(); w.writerows(per_scen_rows)
    print(f"\n  saved: {out_json}, {out_csv}")


if __name__ == "__main__":
    main(min_conf=0.0)
