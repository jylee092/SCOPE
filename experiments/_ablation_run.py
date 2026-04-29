"""
Q4 ablation runner: re-runs SCOPE on all 35 scenarios with one component
disabled at a time, then reports macro technique-LCS for each variant.

Variants (matching Table tab:ablation in the paper):
  no_causal      -- W_CAU = 0 (only tactical + semantic in fusion)
  no_semantic    -- W_SEM = 0 (only tactical + causal)
  no_tactical    -- W_TAC = 0 (only semantic + causal)
  top1_only      -- VITERBI_BEAM_K = 1 (greedy, no distributional mapping)
  no_shared_ent  -- GROUPING_USE_SHARED_ENTITY = False (lineage only) -- heavy
  no_grouping    -- single-event groups via build_solo_groups          -- heavy

Output: output/_ablation/<variant>/<scenario>/<scenario>_viterbi.json
       + a single _ablation_scores.json summary at output/_ablation/.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

# Reuse the cached-Viterbi-only re-runner; we patch config attrs before it runs.
from experiments._rerun_viterbi_only import run_one as rerun_viterbi_one
from experiments.run_eval_v2 import main as run_eval_v2_main
from experiments.chain_align import evaluate_chain_alignment
from experiments.attack_flows import get_flow


ABL_DIR = config.OUTPUT_BASE_DIR / "_ablation"
ABL_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Variant config overrides
# ----------------------------------------------------------------------

VARIANT_OVERRIDES: dict[str, dict] = {
    "no_causal":   {"W_CAU": 0.0, "USE_CAUSAL_SCORING": False},
    "no_semantic": {"W_SEM": 0.0, "USE_SEMANTIC_SCORING": False},
    "no_tactical": {"W_TAC": 0.0},                          # tac=0 keeps scorer alive
    "top1_only":   {"VITERBI_BEAM_K": 1, "VITERBI_MAX_SKIP": 0},
}


def _set_attrs(overrides: dict) -> dict:
    """Temporarily patch config; return previous values for restoration."""
    prev = {}
    for k, v in overrides.items():
        prev[k] = getattr(config, k, None)
        setattr(config, k, v)
    return prev


def _restore_attrs(prev: dict) -> None:
    for k, v in prev.items():
        if v is None and not hasattr(config, k):
            continue
        setattr(config, k, v)


# ----------------------------------------------------------------------
# Per-variant Viterbi re-run + scoring
# ----------------------------------------------------------------------

def _scenario_paths():
    return sorted(config.DATASET_FOLDER.rglob("*.json"))


def _backup_viterbi_files() -> dict:
    """Save current viterbi.json files keyed by scenario."""
    saved = {}
    for ds in _scenario_paths():
        config.configure_dataset(ds)
        if config.VITERBI_JSON_PATH.exists():
            saved[str(config.VITERBI_JSON_PATH)] = config.VITERBI_JSON_PATH.read_text(
                encoding="utf-8"
            )
    return saved


def _restore_viterbi_files(saved: dict) -> None:
    for path_str, content in saved.items():
        Path(path_str).write_text(content, encoding="utf-8")


def _score_all_scenarios() -> dict:
    """Compute macro technique-LCS using existing chain_align over current
    viterbi.json files."""
    rows = []
    for ds in _scenario_paths():
        config.configure_dataset(ds)
        if not config.VITERBI_JSON_PATH.exists():
            continue
        with open(config.VITERBI_JSON_PATH, encoding="utf-8") as f:
            breakdown = json.load(f)
        if not isinstance(breakdown, list):
            continue
        flow = get_flow(config.DATASET_NAME)
        if not flow:
            continue
        chain = evaluate_chain_alignment(config.DATASET_NAME, breakdown, ref_flow=flow)
        rows.append({
            "scenario": config.DATASET_NAME,
            "tech_lcs": chain.get("technique_lcs_norm"),
            "tac_lcs":  chain.get("tactic_lcs_norm"),
            "step_cov": chain.get("step_coverage"),
            "order":    chain.get("order_accuracy"),
        })
    if not rows:
        return {}
    from statistics import mean
    return {
        "n": len(rows),
        "tech_lcs": mean(r["tech_lcs"] for r in rows if r["tech_lcs"] is not None),
        "tac_lcs":  mean(r["tac_lcs"]  for r in rows if r["tac_lcs"]  is not None),
        "step_cov": mean(r["step_cov"] for r in rows if r["step_cov"] is not None),
        "order":    mean(r["order"]    for r in rows if r["order"]    is not None),
        "rows":     rows,
    }


def run_viterbi_variant(name: str) -> dict:
    """Run a Viterbi-only ablation variant across 35 scenarios.

    Strategy: monkey-patch config, run rerun_viterbi for each scenario, score,
    then restore the original viterbi.json files (kept as backup).
    """
    print(f"\n{'='*70}\n  Variant: {name}\n{'='*70}")
    overrides = VARIANT_OVERRIDES[name]
    print(f"  overrides: {overrides}")

    saved = _backup_viterbi_files()
    print(f"  backed up {len(saved)} viterbi files")

    prev = _set_attrs(overrides)
    t0 = time.time()
    n_ok = 0
    for ds in _scenario_paths():
        if rerun_viterbi_one(ds):
            n_ok += 1
    print(f"  re-ran viterbi on {n_ok} scenarios in {time.time()-t0:.1f}s")

    scores = _score_all_scenarios()
    _restore_attrs(prev)
    _restore_viterbi_files(saved)
    print(f"  restored {len(saved)} original viterbi files")

    print(f"  → tech-LCS = {scores.get('tech_lcs', 0.0):.4f}")
    return {"variant": name, **scores}


# ----------------------------------------------------------------------

def main(variants: list[str] | None = None) -> None:
    variants = variants or list(VARIANT_OVERRIDES.keys())
    out = []
    for v in variants:
        if v in VARIANT_OVERRIDES:
            res = run_viterbi_variant(v)
            out.append(res)
        else:
            print(f"[skip] {v}: not a Viterbi-only variant")

    summary_path = ABL_DIR / "_ablation_scores.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {summary_path}")
    print("\n--- Summary ---")
    for r in out:
        print(f"  {r['variant']:<15}  tech-LCS={r.get('tech_lcs', 0.0):.4f}  "
              f"tac-LCS={r.get('tac_lcs', 0.0):.4f}  "
              f"step={r.get('step_cov', 0.0):.4f}")


if __name__ == "__main__":
    main()
