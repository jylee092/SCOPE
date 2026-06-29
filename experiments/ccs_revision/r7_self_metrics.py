"""
R7 (CCS reviewer ⑦): Per-submodule self-evaluation metrics.

Isolates each pipeline stage's own competence using the manually-labelled
gt_ fields already on disk -- NO new inference run.

IMPORTANT: the mapping-module metrics reuse the project's canonical evaluation
machinery (experiments.run_eval_plausible.evaluate_scenario_plausible) so the
numbers are defined exactly as in the paper:
  - ranked prediction list = ttp_mapping["similar_techniques"]  (NOT the
    coarsened top-level "technique_id" field)
  - matching            = tid_family_match (parent/child family match)
  - plausible_*         = match against the scenario acceptable set
                          (reference flow primary + alts + parent/child)
  - strict_h1           = family match of top-1 against the per-group gt label

Submodules reported:
  (1) Grouping module      : purity (group precision), completeness (ref recall)
  (2) Mapping module       : strict_h1 / plausible_h1 / plausible_h5 / MRR
                             (TP groups only -- isolates mapper from grouping FP)
  (3) Hallucination proxy  : false-plausible-assignment rate = fraction of FP
                             (non-attack) groups whose mapper top-1 nonetheless
                             lands in the scenario acceptable set.

Writes solely under output/_ccs_revision/R7_self_metrics/.
Run:  python -m experiments.ccs_revision.r7_self_metrics
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from pipeline.evaluator import load_ground_truth
from experiments.run_eval import load_tactic_map, patch_candidate_tactics
from experiments.run_eval_plausible import evaluate_scenario_plausible, tid_family_match
from experiments.attack_flows import get_flow, all_acceptable_tids

OUT_DIR = ROOT / "output" / "_ccs_revision" / "R7_self_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MITRE_CSV = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"


def _root(tid: str) -> str:
    return tid.split(".")[0] if tid else ""


def _is_true(v) -> bool:
    return v is True or str(v).lower() == "true"


def main():
    tm = load_tactic_map(MITRE_CSV)
    rows = []
    tot_n = 0
    tot_strict = tot_p1 = tot_p5 = tot_mrr = 0.0
    fp_total = fp_plausible = 0  # hallucination proxy pool

    for ann_p in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ann_p)
        ann_fp, map_fp = config.ANNOTATION_JSON_PATH, config.TTP_MAPPING_JSON_PATH
        if not ann_fp.exists() or not map_fp.exists():
            continue
        scenario = config.DATASET_NAME
        flow = get_flow(scenario)
        if not flow:
            continue
        acceptable = all_acceptable_tids(flow)

        gt = load_ground_truth(ann_fp)
        ann = json.load(open(ann_fp, encoding="utf-8"))
        ttp = json.load(open(map_fp, encoding="utf-8"))
        patch_candidate_tactics(ttp, tm)
        pred_by_gid = {m["group_id"]: m for m in ttp}

        # (2) Mapping module -- canonical plausibility eval (TP groups only)
        rec = evaluate_scenario_plausible(scenario, gt, ttp, k=5, strong_only=False)
        nmap = rec.get("n", 0)

        # (1) Grouping module + (3) hallucination proxy
        groups = ann.get("groups", [])
        n_tp = sum(1 for g in groups if _is_true(g.get("gt_is_true_positive")))
        n_fp = len(groups) - n_tp
        ref_roots = {_root(t) for t in acceptable}
        covered = set()
        for g in groups:
            if not _is_true(g.get("gt_is_true_positive")):
                # hallucination proxy: does FP group's top-1 look plausible?
                fp_total += 1
                cand = pred_by_gid.get(g["group_id"], {}).get("similar_techniques", [])
                if cand:
                    top1 = cand[0].get("technique_id")
                    if any(tid_family_match(top1, a) for a in acceptable):
                        fp_plausible += 1
                continue
            gt_root = _root(g.get("gt_technique_id"))
            if gt_root in ref_roots:
                covered.add(gt_root)

        total_g = n_tp + n_fp
        rows.append({
            "scenario": scenario,
            "n_groups": total_g, "n_tp": n_tp, "n_fp": n_fp,
            "purity": round(n_tp / total_g, 4) if total_g else 0.0,
            "ref_roots": len(ref_roots), "covered_roots": len(covered),
            "completeness": round(len(covered) / len(ref_roots), 4) if ref_roots else 0.0,
            "map_n": nmap,
            "strict_h1": round(rec.get("strict_h1", 0.0), 4) if nmap else 0.0,
            "plausible_h1": round(rec.get("plausible_h1", 0.0), 4) if nmap else 0.0,
            "plausible_h5": round(rec.get("plausible_h5", 0.0), 4) if nmap else 0.0,
            "plausible_mrr": round(rec.get("plausible_mrr", 0.0), 4) if nmap else 0.0,
        })
        if nmap:
            tot_n += nmap
            tot_strict += rec["strict_h1"] * nmap
            tot_p1 += rec["plausible_h1"] * nmap
            tot_p5 += rec["plausible_h5"] * nmap
            tot_mrr += rec["plausible_mrr"] * nmap

    n = len(rows)
    if n == 0:
        print("[R7] no scenarios found"); return

    grouping_macro = {
        "purity": round(mean(r["purity"] for r in rows), 4),
        "completeness": round(mean(r["completeness"] for r in rows), 4),
        "purity_micro": round(sum(r["n_tp"] for r in rows) / sum(r["n_groups"] for r in rows), 4),
    }
    # macro over scenarios that actually have TP groups (matches canonical _run,
    # which `continue`s when rec["n"]==0). Including 0-TP scenarios as 0.0 would
    # understate the mapper's competence.
    map_rows = [r for r in rows if r["map_n"] > 0]
    mapping_macro = {
        "n_scenarios": len(map_rows),
        "strict_h1": round(mean(r["strict_h1"] for r in map_rows), 4),
        "plausible_h1": round(mean(r["plausible_h1"] for r in map_rows), 4),
        "plausible_h5": round(mean(r["plausible_h5"] for r in map_rows), 4),
        "plausible_mrr": round(mean(r["plausible_mrr"] for r in map_rows), 4),
    }
    mapping_micro = {
        "n_tp_groups": tot_n,
        "strict_h1": round(tot_strict / tot_n, 4) if tot_n else 0.0,
        "plausible_h1": round(tot_p1 / tot_n, 4) if tot_n else 0.0,
        "plausible_h5": round(tot_p5 / tot_n, 4) if tot_n else 0.0,
        "plausible_mrr": round(tot_mrr / tot_n, 4) if tot_n else 0.0,
    }
    hallucination = {
        "fp_groups": fp_total,
        "fp_plausible_assignment": fp_plausible,
        "false_plausible_rate": round(fp_plausible / fp_total, 4) if fp_total else 0.0,
    }

    summary = {"n_scenarios": n, "grouping_macro": grouping_macro,
               "mapping_macro": mapping_macro, "mapping_micro": mapping_micro,
               "hallucination_proxy": hallucination, "rows": rows}
    with open(OUT_DIR / "self_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "self_metrics_per_scenario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"[R7] {n} scenarios -> {OUT_DIR}")
    print("  -- (1) Grouping module --")
    print(f"     purity       macro {grouping_macro['purity']}  (micro {grouping_macro['purity_micro']}, TP {sum(r['n_tp'] for r in rows)}/{sum(r['n_groups'] for r in rows)})")
    print(f"     completeness macro {grouping_macro['completeness']}")
    print("  -- (2) Mapping module (LLM desc->TID, TP groups, canonical eval) --")
    print(f"     strict_h1    macro {mapping_macro['strict_h1']}  micro {mapping_micro['strict_h1']}")
    print(f"     plausible_h1 macro {mapping_macro['plausible_h1']}  micro {mapping_micro['plausible_h1']}")
    print(f"     plausible_h5 macro {mapping_macro['plausible_h5']}  micro {mapping_micro['plausible_h5']}")
    print(f"     plausible_mrr macro {mapping_macro['plausible_mrr']}  micro {mapping_micro['plausible_mrr']}")
    print("  -- (3) Hallucination proxy --")
    print(f"     false-plausible rate {hallucination['false_plausible_rate']}  ({fp_plausible}/{fp_total} FP groups get a plausible top-1)")


if __name__ == "__main__":
    main()
