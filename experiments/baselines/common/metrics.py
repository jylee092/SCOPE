"""
Baseline 공용 평가 metrics.

GLIDE의 evaluator.py와 일관되지만 입력 형식이 다름 (per-group annotation이 아닌
per-scenario tactic/technique sequence).
"""
from __future__ import annotations

import json
from pathlib import Path


def _lcs(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def load_gt_sequences(annotation_path: Path) -> tuple[list[str], list[str]]:
    """full variant의 annotation JSON에서 GT tactic/technique 시퀀스 추출."""
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    tactics = []
    techs = []
    for g in data["groups"]:
        if not g.get("gt_is_true_positive"):
            continue
        tactics.append(g.get("gt_tactic") or "")
        techs.append(g.get("gt_technique_id") or "")
    return [t for t in tactics if t], [t for t in techs if t]


def tactic_set_prf(gt: list[str], pred: list[str]) -> dict:
    gt_s, pr_s = set(gt), set(pred)
    tp = len(gt_s & pr_s)
    p = tp / len(pr_s) if pr_s else 0.0
    r = tp / len(gt_s) if gt_s else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def sequence_similarity(gt: list[str], pred: list[str]) -> dict:
    if not gt and not pred:
        return {"jaccard": 1.0, "lcs_norm": 1.0}
    gt_s, pr_s = set(gt), set(pred)
    jacc = len(gt_s & pr_s) / len(gt_s | pr_s) if (gt_s | pr_s) else 0.0
    max_len = max(len(gt), len(pred), 1)
    lcs_norm = _lcs(gt, pred) / max_len
    return {"jaccard": round(jacc, 4), "lcs_norm": round(lcs_norm, 4)}


def evaluate_prediction(pred_tactic_seq: list[str], pred_tech_seq: list[str],
                        gt_tactic_seq: list[str], gt_tech_seq: list[str]) -> dict:
    return {
        "num_gt_steps": len(gt_tactic_seq),
        "num_pred_steps": len(pred_tactic_seq),
        "tactic": {
            **tactic_set_prf(gt_tactic_seq, pred_tactic_seq),
            **sequence_similarity(gt_tactic_seq, pred_tactic_seq),
        },
        "technique": {
            **tactic_set_prf(gt_tech_seq, pred_tech_seq),
            **sequence_similarity(gt_tech_seq, pred_tech_seq),
        },
    }
