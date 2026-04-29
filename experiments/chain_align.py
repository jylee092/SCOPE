"""
Chain alignment metric: pred Viterbi chain vs reference attack flow.


Metric
------

"""
from __future__ import annotations
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.attack_flows import ATTACK_FLOWS, get_flow


def step_match(ref_step: dict, pred_tid: str) -> bool:
    """ref_step ...tid ...alts ...pred_tid ..."""
    if not pred_tid: return False
    cands = {ref_step["tid"], *ref_step.get("alts", [])}
    if pred_tid in cands: return True
    p_root = pred_tid.split(".")[0]
    for c in cands:
        c_root = c.split(".")[0]
        if c_root == p_root: return True
    return False


def lcs_length(seq_a: list, seq_b: list, eq=lambda a,b: a==b) -> int:
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0: return 0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if eq(seq_a[i-1], seq_b[j-1]):
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def evaluate_chain_alignment(
    scenario: str,
    viterbi_breakdown: list[dict],
    ref_flow: list[dict] | None = None,
) -> dict:
    if ref_flow is None:
        ref_flow = get_flow(scenario)
    if not ref_flow:
        return {"error": "no reference flow"}

    pred_tids   = [s["technique_id"] for s in viterbi_breakdown]
    pred_tactics= [s["tactic"]       for s in viterbi_breakdown]
    ref_tids    = [s["tid"]          for s in ref_flow]
    ref_tactics = [s["tactic"]       for s in ref_flow]

    matched = sum(
        1 for r in ref_flow
        if any(step_match(r, p) for p in pred_tids)
    )
    step_coverage = matched / len(ref_flow)

    # tactic_set_jaccard
    rs = set(ref_tactics); ps = set(pred_tactics)
    tac_jaccard = len(rs & ps) / len(rs | ps) if (rs | ps) else 0

    tac_lcs = lcs_length(ref_tactics, pred_tactics)
    tac_lcs_norm = tac_lcs / len(ref_tactics) if ref_tactics else 0

    def tech_eq(r_step, p_tid): return step_match(r_step, p_tid)
    tech_lcs = lcs_length(ref_flow, pred_tids, eq=tech_eq)
    tech_lcs_norm = tech_lcs / len(ref_flow) if ref_flow else 0

    matched_indices_in_pred = []
    for r in ref_flow:
        for i, p in enumerate(pred_tids):
            if step_match(r, p):
                matched_indices_in_pred.append(i)
                break
        else:
            matched_indices_in_pred.append(None)
    valid = [i for i in matched_indices_in_pred if i is not None]
    if len(valid) <= 1:
        order_acc = 1.0 if valid else 0.0
    else:
        ordered = sum(1 for a,b in zip(valid, valid[1:]) if a < b)
        order_acc = ordered / (len(valid) - 1)

    return {
        "ref_steps":    len(ref_flow),
        "pred_steps":   len(viterbi_breakdown),
        "step_coverage": round(step_coverage, 4),
        "tactic_jaccard": round(tac_jaccard, 4),
        "tactic_lcs_norm": round(tac_lcs_norm, 4),
        "technique_lcs_norm": round(tech_lcs_norm, 4),
        "order_accuracy": round(order_acc, 4),
        "matched_pairs": [
            {"ref_step": i+1, "ref_tid": ref_flow[i]["tid"],
             "ref_tactic": ref_flow[i]["tactic"],
             "pred_idx": matched_indices_in_pred[i],
             "pred_tid": pred_tids[matched_indices_in_pred[i]] if matched_indices_in_pred[i] is not None else None}
            for i in range(len(ref_flow))
        ],
    }
