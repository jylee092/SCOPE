"""
Score Sigma baseline against SCOPE's plausibility-based GT.

Metrics (matching paper §7):
- Hit@5         : per-TP-group Top-5 plausibility hit, scenario-macro
- tech-LCS      : technique sequence LCS vs attack_flows reference, scenario-macro
- tac-LCS       : tactic sequence LCS vs attack_flows reference, scenario-macro
- step coverage : fraction of reference steps matched by predicted sequence

Inputs:
- output/baselines/sigma/<rel>/result.json     (per-alert Top-5 + sequences)
- output/<rel>/<stem>_feature_result.json     (SCOPE groups with all_idxs)
- output/<rel>/<stem>_annotation.json         (SCOPE TP labels)

Group→sigma alignment uses the row-index space produced by
pipeline.data_loader.load_and_normalize (shared between SCOPE and our adapter).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.attack_flows import get_flow, all_acceptable_tids
from experiments.chain_align import evaluate_chain_alignment, step_match, lcs_length

import config

SIGMA_BASELINE_DIR = config.OUTPUT_BASE_DIR / "baselines" / "sigma"
SCOPE_OUTPUT_DIR = config.OUTPUT_BASE_DIR


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def _load_json(p: Path) -> dict | list | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def find_scope_files(sigma_result_path: Path) -> tuple[Path | None, Path | None]:
    """Given output/baselines/sigma/<rel>/result.json, locate the matching
    SCOPE feature_result.json + annotation.json under output/<rel>/."""
    rel = sigma_result_path.relative_to(SIGMA_BASELINE_DIR).parent
    scope_dir = SCOPE_OUTPUT_DIR / rel
    stem = scope_dir.name
    feat = scope_dir / f"{stem}_feature_result.json"
    ann = scope_dir / f"{stem}_annotation.json"
    return (feat if feat.exists() else None,
            ann if ann.exists() else None)


# ----------------------------------------------------------------------------
# Group → Sigma alert alignment
# ----------------------------------------------------------------------------

def _topk_for_group(group_idxs: set[int], alerts: list[dict], k: int) -> list[str]:
    """Union of priority-ordered top-K TIDs from sigma alerts whose event_index
    falls within the group's event set. Priority preserved across alerts by
    iterating in alert order (which is chronological after match-event sort)."""
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
    """A pred TID matches the acceptable set if it equals an acceptable TID
    or shares the same parent technique (e.g., T1003 ≡ T1003.001)."""
    accept_roots = {a.split(".")[0] for a in acceptable}
    for p in pred_tids:
        if p in acceptable:
            return True
        if p.split(".")[0] in accept_roots:
            return True
    return False


# ----------------------------------------------------------------------------
# Per-scenario scoring
# ----------------------------------------------------------------------------

def score_scenario(sigma_result_path: Path, k: int = 5) -> dict | None:
    sigma = _load_json(sigma_result_path)
    if not sigma:
        return None
    scenario_name = sigma["scenario"]
    alerts: list[dict] = sigma.get("notes", {}).get("alerts", [])

    flow = get_flow(scenario_name)
    if not flow:
        return None
    acceptable = all_acceptable_tids(flow)

    feat_path, ann_path = find_scope_files(sigma_result_path)
    feat = _load_json(feat_path) if feat_path else None
    ann = _load_json(ann_path) if ann_path else None

    # ------- Hit@K over TP groups (matched against SCOPE GT) -------
    hit_k_count = 0
    tp_count = 0
    if feat and ann:
        # idxs per group
        idxs_by_group = {g["group_id"]: set(g.get("all_idxs", []) or [])
                         for g in feat}
        for g in ann.get("groups", []):
            if not g.get("gt_is_true_positive"):
                continue
            gid = g.get("group_id")
            group_idxs = idxs_by_group.get(gid, set())
            if not group_idxs:
                tp_count += 1
                continue
            topk = _topk_for_group(group_idxs, alerts, k)
            tp_count += 1
            if _match_acceptable(topk, acceptable):
                hit_k_count += 1
    hit_k = (hit_k_count / tp_count) if tp_count else None

    # ------- Chain LCS over time-ordered alert sequence -------
    # Build a parallel (technique, tactic) breakdown from alerts, dedup
    # consecutive identical (TID, TAC) pairs, then run SCOPE's chain_align.
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
    }


# ----------------------------------------------------------------------------
# Aggregate
# ----------------------------------------------------------------------------

def main(k: int = 5) -> None:
    paths = sorted(SIGMA_BASELINE_DIR.rglob("result.json"))
    print(f"Found {len(paths)} sigma result files")
    rows: list[dict] = []
    for p in paths:
        r = score_scenario(p, k=k)
        if r:
            rows.append(r)
    print(f"Scored {len(rows)} scenarios\n")

    # Per-scenario detail
    print(f"{'scenario':<55} {'ref':>3} {'alerts':>6} {'pred':>4} "
          f"{'TP':>3} {'H@K':>5} {'tech':>5} {'tac':>5} {'step':>5} {'ord':>5}")
    print("-" * 110)
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
              f"{r['order_accuracy']:.2f}")

    # Macro averages
    def _avg(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return mean(vals) if vals else None

    print("\n" + "=" * 110)
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

    # Save detailed CSV
    out_dir = SIGMA_BASELINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "_scores.json"
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
            "micro_hit_k": (n_hit_total / n_tp_total) if n_tp_total else 0,
            "tp_groups_total": n_tp_total,
            "tp_groups_hit_total": n_hit_total,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
