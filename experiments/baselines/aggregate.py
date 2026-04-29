"""

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
from experiments.baselines.common.adapter import load_prediction
from experiments.baselines.common.metrics import load_gt_sequences, evaluate_prediction
from experiments.baselines.run_all import BASELINES, _output_dir


def load_glide_full_sequence(dataset_rel: Path, stem: str) -> tuple[list[str], list[str]] | None:
    """GLIDE full variant...Viterbi ...sequence ..."""
    vit_path = config.OUTPUT_BASE_DIR / dataset_rel / f"{stem}_viterbi.json"
    if not vit_path.exists():
        return None
    with open(vit_path, encoding="utf-8") as f:
        viterbi = json.load(f)
    tac = [step["tactic"] for step in viterbi]
    tech = [step["technique_id"] for step in viterbi]
    return tac, tech


def evaluate_all(datasets: list[Path]) -> dict:
    by_method: dict[str, list[dict]] = {"glide_full": []}
    for name in BASELINES:
        by_method[name] = []

    for ds in datasets:
        rel = ds.relative_to(config.DATASET_FOLDER).with_suffix("")
        stem = ds.stem
        ann_path = config.OUTPUT_BASE_DIR / rel / f"{stem}_annotation.json"
        if not ann_path.exists():
            continue

        gt_tac, gt_tech = load_gt_sequences(ann_path)
        if not gt_tac:
            continue

        # GLIDE full
        glide = load_glide_full_sequence(rel, stem)
        if glide:
            by_method["glide_full"].append({
                "scenario": stem,
                **evaluate_prediction(glide[0], glide[1], gt_tac, gt_tech),
            })

        # Baselines
        for name in BASELINES:
            out = _output_dir(name, rel) / "result.json"
            if not out.exists():
                continue
            pred = load_prediction(out)
            by_method[name].append({
                "scenario": stem,
                **evaluate_prediction(
                    pred.tactic_sequence, pred.technique_sequence, gt_tac, gt_tech,
                ),
            })
    return by_method


def macro_average(entries: list[dict]) -> dict:
    if not entries:
        return {}
    def avg(path_a, path_b):
        vals = []
        for e in entries:
            v = e.get(path_a, {}).get(path_b)
            if v is not None:
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "n": len(entries),
        "tactic_f1":       avg("tactic", "f1"),
        "tactic_jaccard":  avg("tactic", "jaccard"),
        "tactic_lcs_norm": avg("tactic", "lcs_norm"),
        "technique_f1":    avg("technique", "f1"),
        "technique_lcs":   avg("technique", "lcs_norm"),
    }


def print_table(summary: dict) -> None:
    methods = list(summary.keys())
    print("\n" + "=" * 90)
    print("  BASELINE COMPARISON (vs GLIDE full)")
    print("=" * 90)

    rows = [
        ("# scenarios",      "n"),
        ("Tactic F1",        "tactic_f1"),
        ("Tactic Jaccard",   "tactic_jaccard"),
        ("Tactic LCS norm",  "tactic_lcs_norm"),
        ("Technique F1",     "technique_f1"),
        ("Technique LCS",    "technique_lcs"),
    ]

    print(f"\n  {'Metric':<20s} " + "  ".join(f"{m:>16s}" for m in methods))
    print(f"  {'-'*20} " + "  ".join("-" * 16 for _ in methods))
    for label, key in rows:
        vals = []
        for m in methods:
            v = summary[m].get(key)
            if v is None:
                vals.append(f"{'--':>16s}")
            elif isinstance(v, int):
                vals.append(f"{v:>16d}")
            else:
                vals.append(f"{v:>16.4f}")
        print(f"  {label:<20s} " + "  ".join(vals))
    print("=" * 90)


def main():
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    by_method = evaluate_all(datasets)
    summary = {m: macro_average(entries) for m, entries in by_method.items()}

    out_path = Path(_FINAL_CODE) / "experiments" / "baselines" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_scenario": by_method, "summary": summary}, f,
                  ensure_ascii=False, indent=2)

    print_table(summary)
    print(f"\n  ...: {out_path}")


if __name__ == "__main__":
    main()
