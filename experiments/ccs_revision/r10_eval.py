"""
R10 eval/compare: chain-alignment metrics for each LLM-swap variant.

Runs the SAME evaluate_chain_alignment over each variant's Viterbi output so the
comparison is apples-to-apples. The gemini row is recomputed from canonical
output/ (cross-checks the headline numbers); template from the R10 folder.
GPT row is added automatically once output/_ccs_revision/R10_llm_swap/gpt exists.

Run:  python -m experiments.ccs_revision.r10_eval
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from experiments.chain_align import evaluate_chain_alignment
from experiments.attack_flows import get_flow

CANON = config.OUTPUT_BASE_DIR
R10 = CANON / "_ccs_revision" / "R10_llm_swap"
OUT_DIR = R10
VARIANTS = [
    ("gemini",   CANON),
    ("template", R10 / "template"),
    ("gpt",      R10 / "gpt"),
]


def eval_variant(base: Path):
    rows = []
    try:
        for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
            config.OUTPUT_BASE_DIR = base
            config.configure_dataset(ds)
            vp = config.VITERBI_JSON_PATH
            if not vp.exists():
                continue
            bd = json.load(open(vp, encoding="utf-8"))
            flow = get_flow(config.DATASET_NAME)
            if not flow:
                continue
            r = evaluate_chain_alignment(config.DATASET_NAME, bd, flow)
            if "error" in r:
                continue
            rows.append({"scenario": config.DATASET_NAME, **r})
    finally:
        config.OUTPUT_BASE_DIR = CANON
    return rows


def main():
    summary = {}
    per_variant_rows = {}
    for name, base in VARIANTS:
        if not base.exists():
            continue
        rows = eval_variant(base)
        if not rows:
            continue
        per_variant_rows[name] = rows
        summary[name] = {
            "n": len(rows),
            "tech_lcs": round(mean(r["technique_lcs_norm"] for r in rows), 4),
            "tac_lcs":  round(mean(r["tactic_lcs_norm"] for r in rows), 4),
            "step_cov": round(mean(r["step_coverage"] for r in rows), 4),
            "order":    round(mean(r["order_accuracy"] for r in rows), 4),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "llm_swap_compare.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[R10] LLM-swap comparison (macro over scenarios) -> {OUT_DIR/'llm_swap_compare.json'}")
    print(f"  {'variant':<10}{'n':>4}{'tech_lcs':>10}{'tac_lcs':>10}{'step_cov':>10}{'order':>9}")
    base = summary.get("gemini")
    for name, _ in VARIANTS:
        if name not in summary:
            continue
        m = summary[name]
        delta = ""
        if base and name != "gemini":
            delta = f"   Δtech {m['tech_lcs']-base['tech_lcs']:+.4f}"
        print(f"  {name:<10}{m['n']:>4}{m['tech_lcs']:>10.4f}{m['tac_lcs']:>10.4f}"
              f"{m['step_cov']:>10.4f}{m['order']:>9.4f}{delta}")


if __name__ == "__main__":
    main()
