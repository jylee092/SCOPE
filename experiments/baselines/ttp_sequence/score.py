"""
Score the DeepAG (TID-adapted) baseline against SCOPE's plausibility-based
ground truth. Identical metric definitions as the Sigma scorer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.attack_flows import get_flow, all_acceptable_tids
from experiments.chain_align import evaluate_chain_alignment

import config

DEEPAG_BASELINE_DIR = config.OUTPUT_BASE_DIR / "baselines" / "deepag"
SCOPE_OUTPUT_DIR    = config.OUTPUT_BASE_DIR


def _load_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def find_scope_files(result_path: Path) -> tuple[Path | None, Path | None]:
    rel = result_path.relative_to(DEEPAG_BASELINE_DIR).parent
    scope_dir = SCOPE_OUTPUT_DIR / rel
    stem = scope_dir.name
    feat = scope_dir / f"{stem}_feature_result.json"
    ann = scope_dir / f"{stem}_annotation.json"
    return (feat if feat.exists() else None,
            ann if ann.exists() else None)


def _topk_for_group(group_idxs: set[int], alerts: list[dict], k: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for a in alerts:
        if a.get("event_index") not in group_idxs:
            continue
        for tid in a.get("topk_tids", []):
            if tid in seen:
                continue
            seen.add(tid)
            out.append(tid)
            if len(out) >= k:
                return out
    return out


def _match_acceptable(pred_tids: list[str], acceptable: set[str]) -> bool:
    accept_roots = {a.split(".")[0] for a in acceptable}
    for p in pred_tids:
        if p in acceptable:
            return True
        if p.split(".")[0] in accept_roots:
            return True
    return False


def score_scenario(result_path: Path, k: int = 5) -> dict | None:
    pred = _load_json(result_path)
    if not pred:
        return None
    scenario_name = pred["scenario"]
    alerts: list[dict] = pred.get("notes", {}).get("alerts", [])

    flow = get_flow(scenario_name)
    if not flow:
        return None
    acceptable = all_acceptable_tids(flow)

    feat_path, ann_path = find_scope_files(result_path)
    feat = _load_json(feat_path) if feat_path else None
    ann = _load_json(ann_path) if ann_path else None

    hit_k_count = 0
    tp_count = 0
    if feat and ann:
        idxs_by_group = {g["group_id"]: set(g.get("all_idxs", []) or [])
                         for g in feat}
        for g in ann.get("groups", []):
            if not g.get("gt_is_true_positive"):
                continue
            gid = g.get("group_id")
            group_idxs = idxs_by_group.get(gid, set())
            tp_count += 1
            if not group_idxs:
                continue
            topk = _topk_for_group(group_idxs, alerts, k)
            if _match_acceptable(topk, acceptable):
                hit_k_count += 1
    hit_k = (hit_k_count / tp_count) if tp_count else None

    breakdown = []
    last = (None, None)
    for a in alerts:
        if not a.get("topk_tids"):
            continue
        tid = a["topk_tids"][0]
        tac = (a.get("topk_tactics") or [""])[0]
        if (tid, tac) == last:
            continue
        breakdown.append({"technique_id": tid, "tactic": tac})
        last = (tid, tac)

    chain = evaluate_chain_alignment(scenario_name, breakdown, ref_flow=flow)

    return {
        "scenario": scenario_name,
        "ref_steps": chain.get("ref_steps"),
        "n_alerts": len(alerts),
        "n_pred_steps": len(breakdown),
        "tp_groups": tp_count,
        "tp_groups_hit_k": hit_k_count,
        f"hit@{k}": hit_k,
        "step_coverage": chain.get("step_coverage"),
        "tactic_lcs_norm": chain.get("tactic_lcs_norm"),
        "technique_lcs_norm": chain.get("technique_lcs_norm"),
        "order_accuracy": chain.get("order_accuracy"),
        "tactic_jaccard": chain.get("tactic_jaccard"),
        "n_deepag_used": pred.get("notes", {}).get("n_deepag_top1_used", 0),
    }


def main(k: int = 5) -> None:
    paths = sorted(DEEPAG_BASELINE_DIR.rglob("result.json"))
    print(f"Found {len(paths)} DeepAG result files")
    rows: list[dict] = []
    for p in paths:
        r = score_scenario(p, k=k)
        if r:
            rows.append(r)
    print(f"Scored {len(rows)} scenarios\n")

    print(f"{'scenario':<55} {'ref':>3} {'alerts':>6} {'pred':>4} "
          f"{'TP':>3} {'H@K':>5} {'tech':>5} {'tac':>5} {'step':>5} {'ord':>5} {'dpag':>4}")
    print("-" * 120)
    for r in rows:
        h = r[f"hit@{k}"]
        h_str = "  -- " if h is None else f"{h:.2f}"
        print(f"{r['scenario'][:54]:<55} "
              f"{r['ref_steps']:>3} {r['n_alerts']:>6} {r['n_pred_steps']:>4} "
              f"{r['tp_groups']:>3} "
              f"{h_str:>5} "
              f"{r['technique_lcs_norm']:.2f} "
              f"{r['tactic_lcs_norm']:.2f} "
              f"{r['step_coverage']:.2f} "
              f"{r['order_accuracy']:.2f} "
              f"{r['n_deepag_used']:>4}")

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return mean(vals) if vals else None

    print("\n" + "=" * 120)
    print(f"Macro averages over {len(rows)} scenarios:")
    print(f"  Hit@{k}            : {_avg(f'hit@{k}'):.4f}")
    print(f"  technique-LCS     : {_avg('technique_lcs_norm'):.4f}")
    print(f"  tactic-LCS        : {_avg('tactic_lcs_norm'):.4f}")
    print(f"  step coverage     : {_avg('step_coverage'):.4f}")
    print(f"  order accuracy    : {_avg('order_accuracy'):.4f}")
    print(f"  tactic Jaccard    : {_avg('tactic_jaccard'):.4f}")
    n_tp_total = sum(r["tp_groups"] for r in rows)
    n_hit_total = sum(r["tp_groups_hit_k"] for r in rows)
    print(f"  TP groups (total) : {n_tp_total}, micro Hit@{k} = "
          f"{n_hit_total / n_tp_total if n_tp_total else 0:.4f}")
    n_deepag_total = sum(r["n_deepag_used"] for r in rows)
    n_alerts_total = sum(r["n_alerts"] for r in rows)
    print(f"  DeepAG-top1 used  : {n_deepag_total}/{n_alerts_total} "
          f"({100 * n_deepag_total / max(n_alerts_total,1):.1f}% of alerts)")

    out_csv = DEEPAG_BASELINE_DIR / "_scores.json"
    with open(out_csv, "w", encoding="utf-8") as f:
        json.dump({
            "k": k,
            "macro": {
                f"hit@{k}":  _avg(f'hit@{k}'),
                "technique_lcs_norm": _avg('technique_lcs_norm'),
                "tactic_lcs_norm":    _avg('tactic_lcs_norm'),
                "step_coverage":      _avg('step_coverage'),
                "order_accuracy":     _avg('order_accuracy'),
            },
            "micro_hit_k":          (n_hit_total / n_tp_total) if n_tp_total else 0,
            "tp_groups_total":      n_tp_total,
            "tp_groups_hit_total":  n_hit_total,
            "n_deepag_used_total":  n_deepag_total,
            "n_alerts_total":       n_alerts_total,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
