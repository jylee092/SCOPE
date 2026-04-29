"""
Plausibility-based TTP evaluation.



----

"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT))

from pipeline.evaluator import load_ground_truth
from experiments.run_eval import (
    load_tactic_map, patch_candidate_tactics,
)
from experiments.attack_flows import get_flow, all_acceptable_tids

MITRE_CSV = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"


def tid_family_match(pred: str, target: str) -> bool:
    """pred...target...parent/child/...True."""
    if not pred or not target:
        return False
    if pred == target:
        return True
    return pred.split(".")[0] == target.split(".")[0]


def plausibility_hit(pred_tids: list[str], acceptable: set[str], k: int) -> tuple[int, int, float]:
    """pred top-K ...acceptable set ...match...hit."""
    h1 = 0
    hk = 0
    rr = 0.0
    for rank, p in enumerate(pred_tids[:k], start=1):
        matched = any(tid_family_match(p, a) for a in acceptable)
        if matched:
            if rank == 1:
                h1 = 1
            hk = 1
            if rr == 0:
                rr = 1.0 / rank
    return h1, hk, rr


_STRONG_REASONS = ("anchor-kw", "rule-exact", "rule-fam-strong")


def _is_strong_tp(gt_note: str) -> bool:
    """annotation...gt_notes...TP ...

    """
    if not gt_note:
        return False
    s = gt_note.replace("auto: ", "").strip()
    return any(s.startswith(r) for r in _STRONG_REASONS)


def evaluate_scenario_plausible(
    scenario: str,
    gt: dict,
    ttp_results: list[dict],
    k: int = 5,
    strong_only: bool = False,
) -> dict:
    flow = get_flow(scenario)
    if not flow:
        return {"n": 0, "note": "no reference flow"}
    acceptable = all_acceptable_tids(flow)

    metrics = []
    for r in ttp_results:
        gid = r["group_id"]
        if gid not in gt or not gt[gid]["is_tp"]:
            continue
        if strong_only and not _is_strong_tp(gt[gid].get("notes", "")):
            continue
        candidates = r.get("similar_techniques", [])
        ranked = [c["technique_id"] for c in candidates]
        h1, hk, rr = plausibility_hit(ranked, acceptable, k)

        strict_gt = gt[gid]["technique_id"]
        strict_h1 = int(bool(ranked) and tid_family_match(ranked[0], strict_gt))

        metrics.append({
            "group_id": gid,
            "gt_technique_id": strict_gt,
            "acceptable_tids": sorted(acceptable),
            "pred_top1": ranked[0] if ranked else "",
            "pred_top5": ranked[:k],
            "strict_h1": strict_h1,
            "plausible_h1": h1,
            f"plausible_h{k}": hk,
            "plausible_rr": rr,
        })

    n = len(metrics)
    if n == 0:
        return {"n": 0, "acceptable_tids": sorted(acceptable)}
    return {
        "n": n,
        "acceptable_tids": sorted(acceptable),
        "strict_h1":     sum(m["strict_h1"] for m in metrics) / n,
        "plausible_h1":  sum(m["plausible_h1"] for m in metrics) / n,
        f"plausible_h{k}": sum(m[f"plausible_h{k}"] for m in metrics) / n,
        "plausible_mrr": sum(m["plausible_rr"] for m in metrics) / n,
        "details": metrics,
    }


def _run(strong_only: bool, csv_name: str, label: str):
    tm = load_tactic_map(MITRE_CSV)
    per_scen = []
    total_strict_h1 = total_plausible_h1 = total_plausible_h5 = total_plausible_mrr = 0.0
    total_n = 0

    for ann in sorted(OUTPUT_DIR.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt:
            continue
        with open(ann, encoding="utf-8") as f:
            ann_data = json.load(f)
        scenario = ann_data.get("scenario", ann.parent.name)
        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        if not ttp_fp.exists():
            continue
        with open(ttp_fp, encoding="utf-8") as f:
            ttp = json.load(f)
        patch_candidate_tactics(ttp, tm)

        rec = evaluate_scenario_plausible(scenario, gt, ttp, k=5, strong_only=strong_only)
        if rec["n"] == 0:
            continue

        per_scen.append({
            "scenario": scenario[:60],
            "n": rec["n"],
            "strict_h1":     round(rec["strict_h1"], 3),
            "plausible_h1":  round(rec["plausible_h1"], 3),
            "plausible_h5":  round(rec["plausible_h5"], 3),
            "plausible_mrr": round(rec["plausible_mrr"], 3),
            "acceptable":    ",".join(rec["acceptable_tids"]),
        })
        total_n += rec["n"]
        total_strict_h1    += rec["strict_h1"] * rec["n"]
        total_plausible_h1 += rec["plausible_h1"] * rec["n"]
        total_plausible_h5 += rec["plausible_h5"] * rec["n"]
        total_plausible_mrr+= rec["plausible_mrr"] * rec["n"]

    if total_n == 0:
        print(f"[{label}] No groups evaluated")
        return

    print("=" * 95)
    print(f"  {label}  (acceptable = scenario flow primary+alts+parent/child)")
    print("=" * 95)
    print(f"  scenarios={len(per_scen)}  total TP groups={total_n}")
    print()
    print(f"  Strict    H@1  : {total_strict_h1 / total_n:.4f}")
    print(f"  Plausible H@1  : {total_plausible_h1 / total_n:.4f}")
    print(f"  Plausible H@5  : {total_plausible_h5 / total_n:.4f}")
    print(f"  Plausible MRR  : {total_plausible_mrr / total_n:.4f}")
    print()

    n_s = len(per_scen)
    if n_s > 0:
        print(f"  Macro (scenario-avg):")
        print(f"    Strict H@1     = {sum(r['strict_h1']     for r in per_scen) / n_s:.4f}")
        print(f"    Plausible H@1  = {sum(r['plausible_h1']  for r in per_scen) / n_s:.4f}")
        print(f"    Plausible H@5  = {sum(r['plausible_h5']  for r in per_scen) / n_s:.4f}")
        print(f"    Plausible MRR  = {sum(r['plausible_mrr'] for r in per_scen) / n_s:.4f}")
        print()

    with open(OUTPUT_DIR / csv_name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_scen[0].keys()))
        w.writeheader(); w.writerows(per_scen)
    print(f"  saved: {OUTPUT_DIR / csv_name}\n")


def main():
    _run(strong_only=False, csv_name="eval_plausible_all.csv",    label="ALL TPs (anchor-kw + sample-kw)")
    _run(strong_only=True,  csv_name="eval_plausible_strong.csv", label="STRONG TPs ONLY (anchor-kw + rule-exact + rule-fam)")


if __name__ == "__main__":
    main()
