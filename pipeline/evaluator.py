"""
Evaluation Harness


----
- Group-level: TTP Hit@1, Hit@K, MRR, Tactic accuracy
- Scenario-level: Tactic-sequence Jaccard, Edit distance
- Aggregate: macro-averaged precision/recall/F1 per tactic

--------
load_ground_truth(annotation_path)
evaluate_ttp_mapping(gt, ttp_results)
evaluate_tactic_chain(gt, viterbi_breakdown)
aggregate_report(eval_results_list)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def load_ground_truth(annotation_path: str | Path) -> dict:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for g in data["groups"]:
        if g.get("gt_is_true_positive") is None:
            continue
        gt[g["group_id"]] = {
            "technique_id": g["gt_technique_id"],
            "technique_name": g.get("gt_technique_name", ""),
            "tactic": g["gt_tactic"],
            "is_tp": bool(g["gt_is_true_positive"]),
            "notes": g.get("gt_notes", ""),
        }
    return gt


def evaluate_ttp_mapping(
    gt: dict,
    ttp_results: list[dict],
    k: int = 5,
) -> dict:
    """...TTP ..."""
    metrics = []
    for r in ttp_results:
        gid = r["group_id"]
        if gid not in gt:
            continue
        truth = gt[gid]
        if not truth["is_tp"]:
            continue

        gt_tid = truth["technique_id"]
        candidates = r.get("similar_techniques", [])
        ranked_tids = [c["technique_id"] for c in candidates]

        hit_at_1 = 1 if ranked_tids and ranked_tids[0] == gt_tid else 0
        hit_at_k = 1 if gt_tid in ranked_tids[:k] else 0

        rr = 0.0
        for rank, tid in enumerate(ranked_tids, start=1):
            if tid == gt_tid:
                rr = 1.0 / rank
                break

        pred_tactic = candidates[0].get("tactic", "") if candidates else ""

        metrics.append({
            "group_id": gid,
            "gt_technique_id": gt_tid,
            "gt_tactic": truth["tactic"],
            "pred_technique_id": ranked_tids[0] if ranked_tids else "",
            "pred_tactic": pred_tactic,
            "hit_at_1": hit_at_1,
            f"hit_at_{k}": hit_at_k,
            "reciprocal_rank": rr,
        })

    n = len(metrics)
    if n == 0:
        return {"n": 0, "hit_at_1": 0, f"hit_at_{k}": 0, "mrr": 0, "details": []}

    return {
        "n": n,
        "hit_at_1": sum(m["hit_at_1"] for m in metrics) / n,
        f"hit_at_{k}": sum(m[f"hit_at_{k}"] for m in metrics) / n,
        "mrr": sum(m["reciprocal_rank"] for m in metrics) / n,
        "details": metrics,
    }


def _tactic_prf(gt_tactics: list[str], pred_tactics: list[str]) -> dict:
    """Multi-label tactic precision/recall/F1."""
    gt_set = set(gt_tactics)
    pred_set = set(pred_tactics)
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _edit_distance(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def evaluate_tactic_chain(
    gt: dict,
    viterbi_breakdown: list[dict],
) -> dict:
    """Viterbi ...tactic ...GT..."""
    gt_tactics = []
    pred_tactics = []

    for step in viterbi_breakdown:
        gid = step["group_id"]
        pred_tactics.append(step["tactic"])
        if gid in gt and gt[gid]["is_tp"]:
            gt_tactics.append(gt[gid]["tactic"])
        else:
            gt_tactics.append("_FP_")

    gt_set = set(gt_tactics) - {"_FP_"}
    pred_set = set(pred_tactics)

    jaccard = (
        len(gt_set & pred_set) / len(gt_set | pred_set)
        if (gt_set | pred_set) else 0.0
    )

    gt_seq_clean = [t for t in gt_tactics if t != "_FP_"]
    pred_seq_clean = [step["tactic"] for step in viterbi_breakdown
                      if step["group_id"] in gt and gt[step["group_id"]]["is_tp"]]
    edit_dist = _edit_distance(gt_seq_clean, pred_seq_clean)
    max_len = max(len(gt_seq_clean), len(pred_seq_clean), 1)
    norm_edit = round(1 - edit_dist / max_len, 4)

    prf = _tactic_prf(gt_seq_clean, pred_seq_clean)

    return {
        "gt_sequence": gt_seq_clean,
        "pred_sequence": pred_seq_clean,
        "tactic_set_jaccard": round(jaccard, 4),
        "normalized_edit_similarity": norm_edit,
        **prf,
        "num_gt_groups": len(gt_seq_clean),
        "num_pred_groups": len(pred_seq_clean),
    }


def aggregate_report(eval_list: list[dict], output_path: str | Path | None = None) -> dict:
    """...evaluate ...aggregate metrics ..."""
    n = len(eval_list)
    if n == 0:
        return {}

    ttp_metrics = [e["ttp"] for e in eval_list if "ttp" in e]
    chain_metrics = [e["chain"] for e in eval_list if "chain" in e]

    agg = {"num_scenarios": n}

    if ttp_metrics:
        total_groups = sum(m["n"] for m in ttp_metrics)
        agg["ttp"] = {
            "total_groups": total_groups,
            "macro_hit_at_1": round(sum(m["hit_at_1"] for m in ttp_metrics) / len(ttp_metrics), 4),
            "macro_mrr": round(sum(m["mrr"] for m in ttp_metrics) / len(ttp_metrics), 4),
        }
        for k_key in [k for k in ttp_metrics[0] if k.startswith("hit_at_") and k != "hit_at_1"]:
            agg["ttp"][f"macro_{k_key}"] = round(
                sum(m.get(k_key, 0) for m in ttp_metrics) / len(ttp_metrics), 4
            )

    if chain_metrics:
        agg["chain"] = {
            "macro_jaccard": round(
                sum(m["tactic_set_jaccard"] for m in chain_metrics) / len(chain_metrics), 4
            ),
            "macro_edit_sim": round(
                sum(m["normalized_edit_similarity"] for m in chain_metrics) / len(chain_metrics), 4
            ),
            "macro_precision": round(
                sum(m["precision"] for m in chain_metrics) / len(chain_metrics), 4
            ),
            "macro_recall": round(
                sum(m["recall"] for m in chain_metrics) / len(chain_metrics), 4
            ),
            "macro_f1": round(
                sum(m["f1"] for m in chain_metrics) / len(chain_metrics), 4
            ),
        }

    # per-tactic breakdown
    tactic_tp = Counter()
    tactic_fp = Counter()
    tactic_fn = Counter()
    for e in eval_list:
        if "ttp" not in e:
            continue
        for d in e["ttp"].get("details", []):
            gt_t = d["gt_tactic"]
            pred_t = d.get("pred_tactic", "")
            if gt_t == pred_t:
                tactic_tp[gt_t] += 1
            else:
                tactic_fn[gt_t] += 1
                tactic_fp[pred_t] += 1

    all_tactics = sorted(set(tactic_tp) | set(tactic_fp) | set(tactic_fn))
    per_tactic = {}
    for t in all_tactics:
        tp = tactic_tp[t]
        fp = tactic_fp[t]
        fn = tactic_fn[t]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        per_tactic[t] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
    agg["per_tactic"] = per_tactic

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"  Aggregate report: {output_path}")

    return agg


def print_report(agg: dict) -> None:
    """Aggregate report..."""
    print("\n" + "═" * 75)
    print("  EVALUATION REPORT")
    print("═" * 75)
    print(f"  Scenarios: {agg.get('num_scenarios', 0)}")

    if "ttp" in agg:
        t = agg["ttp"]
        print(f"\n  [TTP Mapping]  groups={t['total_groups']}")
        print(f"    Hit@1 = {t['macro_hit_at_1']:.4f}")
        for k, v in t.items():
            if k.startswith("macro_hit_at_") and k != "macro_hit_at_1":
                print(f"    {k.replace('macro_', '').replace('_', '@')} = {v:.4f}")
        print(f"    MRR   = {t['macro_mrr']:.4f}")

    if "chain" in agg:
        c = agg["chain"]
        print(f"\n  [Tactic Chain]")
        print(f"    Jaccard     = {c['macro_jaccard']:.4f}")
        print(f"    Edit Sim    = {c['macro_edit_sim']:.4f}")
        print(f"    Precision   = {c['macro_precision']:.4f}")
        print(f"    Recall      = {c['macro_recall']:.4f}")
        print(f"    F1          = {c['macro_f1']:.4f}")

    if "per_tactic" in agg:
        print(f"\n  [Per-Tactic Breakdown]")
        print(f"    {'Tactic':<30s} {'P':>6s} {'R':>6s} {'F1':>6s}  TP  FP  FN")
        print(f"    {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*3} {'─'*3} {'─'*3}")
        for tactic, m in sorted(agg["per_tactic"].items()):
            print(f"    {tactic:<30s} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f}"
                  f"  {m['tp']:>3d} {m['fp']:>3d} {m['fn']:>3d}")

    print("═" * 75)
