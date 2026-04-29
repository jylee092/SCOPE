"""



    python experiments/ablation/aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from experiments.ablation.variants import VARIANT_NAMES
from pipeline.evaluator import (
    load_ground_truth, evaluate_ttp_mapping, evaluate_tactic_chain,
)


def _variant_output_dir(variant: str, dataset_rel: Path) -> Path:
    if variant == "full":
        return config.OUTPUT_BASE_DIR / dataset_rel
    return config.OUTPUT_BASE_DIR / f"ablation_{variant}" / dataset_rel


def _full_annotation_path(dataset_rel: Path, stem: str) -> Path:
    """full variant...annotation JSON (GT ..."""
    return config.OUTPUT_BASE_DIR / dataset_rel / f"{stem}_annotation.json"


def evaluate_variant(variant: str, datasets: list[Path]) -> list[dict]:
    results: list[dict] = []
    for ds in datasets:
        rel = ds.relative_to(config.DATASET_FOLDER).with_suffix("")
        stem = ds.stem
        ann_path = _full_annotation_path(rel, stem)
        if not ann_path.exists():
            continue
        gt = load_ground_truth(ann_path)
        if not gt:
            continue

        out_dir = _variant_output_dir(variant, rel)
        ttp_path = out_dir / f"{stem}_ttp_mapping.json"
        vit_path = out_dir / f"{stem}_viterbi.json"

        entry = {"variant": variant, "scenario": stem}

        if variant != "no_grouping" and ttp_path.exists():
            with open(ttp_path, encoding="utf-8") as f:
                ttp = json.load(f)
            entry["ttp"] = evaluate_ttp_mapping(gt, ttp)

        if vit_path.exists():
            with open(vit_path, encoding="utf-8") as f:
                vit = json.load(f)
            entry["chain"] = evaluate_tactic_chain(gt, vit)

        if "ttp" in entry or "chain" in entry:
            results.append(entry)
    return results


def build_comparison(by_variant: dict[str, list[dict]]) -> dict:
    """variant...macro ..."""
    out: dict[str, dict] = {}
    for variant, entries in by_variant.items():
        ttp_m = [e["ttp"] for e in entries if "ttp" in e]
        chain_m = [e["chain"] for e in entries if "chain" in e]

        metrics: dict = {"num_scenarios": len(entries)}
        if ttp_m:
            metrics["ttp"] = {
                "macro_hit_at_1": round(sum(m["hit_at_1"] for m in ttp_m) / len(ttp_m), 4),
                "macro_mrr":      round(sum(m["mrr"] for m in ttp_m) / len(ttp_m), 4),
            }
            for k_key in [k for k in ttp_m[0] if k.startswith("hit_at_") and k != "hit_at_1"]:
                metrics["ttp"][f"macro_{k_key}"] = round(
                    sum(m.get(k_key, 0) for m in ttp_m) / len(ttp_m), 4
                )
        if chain_m:
            metrics["chain"] = {
                "macro_jaccard":  round(sum(m["tactic_set_jaccard"] for m in chain_m) / len(chain_m), 4),
                "macro_edit_sim": round(sum(m["normalized_edit_similarity"] for m in chain_m) / len(chain_m), 4),
                "macro_precision": round(sum(m["precision"] for m in chain_m) / len(chain_m), 4),
                "macro_recall":    round(sum(m["recall"] for m in chain_m) / len(chain_m), 4),
                "macro_f1":        round(sum(m["f1"] for m in chain_m) / len(chain_m), 4),
            }
        out[variant] = metrics
    return out


def print_comparison_table(comparison: dict) -> None:
    print("\n" + "=" * 85)
    print("  ABLATION COMPARISON")
    print("=" * 85)

    variants = list(comparison.keys())
    print(f"\n  {'Metric':<20s} " + "  ".join(f"{v:>12s}" for v in variants))
    print(f"  {'-'*20} " + "  ".join("-" * 12 for _ in variants))

    rows = [
        ("TTP Hit@1",     ("ttp", "macro_hit_at_1")),
        ("TTP Hit@5",     ("ttp", "macro_hit_at_5")),
        ("TTP MRR",       ("ttp", "macro_mrr")),
        ("Chain Jaccard", ("chain", "macro_jaccard")),
        ("Chain Edit Sim",("chain", "macro_edit_sim")),
        ("Chain F1",      ("chain", "macro_f1")),
        ("Chain P",       ("chain", "macro_precision")),
        ("Chain R",       ("chain", "macro_recall")),
    ]
    for label, (cat, key) in rows:
        values = []
        for v in variants:
            val = comparison.get(v, {}).get(cat, {}).get(key)
            values.append(f"{val:>12.4f}" if isinstance(val, (int, float)) else f"{'--':>12s}")
        print(f"  {label:<20s} " + "  ".join(values))

    print(f"\n  {'# scenarios':<20s} " + "  ".join(
        f"{comparison[v].get('num_scenarios', 0):>12d}" for v in variants
    ))
    print("=" * 85)


def main():
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    by_variant: dict[str, list[dict]] = {}
    for variant in VARIANT_NAMES:
        by_variant[variant] = evaluate_variant(variant, datasets)
        print(f"  {variant}: {len(by_variant[variant])} scenarios evaluated")

    comparison = build_comparison(by_variant)

    out_path = _FINAL_CODE / "experiments" / "ablation" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"by_variant": by_variant, "comparison": comparison}, f,
                  ensure_ascii=False, indent=2)

    print_comparison_table(comparison)
    print(f"\n  ...: {out_path}")


if __name__ == "__main__":
    main()
