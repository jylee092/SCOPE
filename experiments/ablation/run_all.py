"""

    cd Final_Code
    python experiments/ablation/run_all.py
    python experiments/ablation/run_all.py --variants no_grouping no_llm
    python experiments/ablation/run_all.py --scenarios atomic/collection/msf_record_mic
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from experiments.ablation.variants import VARIANT_NAMES, run_variant_on_scenario


def discover_datasets(filter_prefix: str | None = None) -> list[Path]:
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    if filter_prefix:
        datasets = [d for d in datasets if filter_prefix in str(d.relative_to(config.DATASET_FOLDER))]
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                    choices=VARIANT_NAMES,
                    help="...variant (...: ...")
    ap.add_argument("--scenarios", type=str, default=None,
                    help="...prefix ...: atomic/collection)")
    ap.add_argument("--output", type=str, default="experiments/ablation/run_summary.json",
                    help="...")
    args = ap.parse_args()

    datasets = discover_datasets(args.scenarios)
    if not datasets:
        print("...")
        return

    print(f"\n{'='*75}")
    print(f"  Ablation Runner")
    print(f"  Variants : {args.variants}")
    print(f"  Datasets : {len(datasets)}")
    print(f"{'='*75}\n")

    summary: list[dict] = []
    failed: list[dict] = []

    for variant in args.variants:
        for ds in datasets:
            try:
                r = run_variant_on_scenario(variant, ds)
                summary.append(r)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"\n  ✗ {variant}/{ds.name} ...-- {err_msg}")
                traceback.print_exc()
                failed.append({"variant": variant, "scenario": ds.stem, "error": err_msg})

    output_path = _FINAL_CODE / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"runs": summary, "failed": failed}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*75}")
    print(f"  ...: ...{len(summary)} / ...{len(failed)}")
    print(f"  ...: {output_path}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
