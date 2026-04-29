"""Score the no_grouping / no_shared_entity ablation variants by reading
their viterbi.json files and running chain_align against attack_flows."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment


def score_variant(variant: str) -> dict:
    base = config.OUTPUT_BASE_DIR / f"ablation_{variant}"
    rows = []
    for vit_path in sorted(base.rglob("*_viterbi.json")):
        with open(vit_path, encoding="utf-8") as f:
            breakdown = json.load(f)
        if not isinstance(breakdown, list):
            continue
        scenario = vit_path.stem.replace("_viterbi", "")
        flow = get_flow(scenario)
        if not flow:
            continue
        chain = evaluate_chain_alignment(scenario, breakdown, ref_flow=flow)
        rows.append({
            "scenario": scenario,
            "tech_lcs": chain.get("technique_lcs_norm"),
            "tac_lcs":  chain.get("tactic_lcs_norm"),
            "step_cov": chain.get("step_coverage"),
            "order":    chain.get("order_accuracy"),
            "n_pred":   len(breakdown),
        })
    if not rows:
        return {"variant": variant, "n": 0}
    return {
        "variant":  variant,
        "n":        len(rows),
        "tech_lcs": mean(r["tech_lcs"] for r in rows if r["tech_lcs"] is not None),
        "tac_lcs":  mean(r["tac_lcs"]  for r in rows if r["tac_lcs"]  is not None),
        "step_cov": mean(r["step_cov"] for r in rows if r["step_cov"] is not None),
        "order":    mean(r["order"]    for r in rows if r["order"]    is not None),
    }


if __name__ == "__main__":
    for v in ("no_grouping", "no_shared_entity", "anchor_only"):
        r = score_variant(v)
        print(f"  {r['variant']:<20} n={r.get('n', 0)}  "
              f"tech-LCS={r.get('tech_lcs', 0):.4f}  "
              f"tac-LCS={r.get('tac_lcs', 0):.4f}  "
              f"step={r.get('step_cov', 0):.4f}")
