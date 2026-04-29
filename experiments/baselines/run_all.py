"""

    python experiments/baselines/run_all.py
    python experiments/baselines/run_all.py --baselines event_level llm_shield
    python experiments/baselines/run_all.py --scenarios atomic/collection
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_FINAL_CODE = Path(__file__).resolve().parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config

BASELINES = {
    "sigma":       ("experiments.baselines.sigma.adapter", "SigmaAdapter"),
    "shield":      ("experiments.baselines.shield.adapter", "ShieldAdapter"),
    "event_level": ("experiments.baselines.event_level.adapter", "EventLevelAdapter"),
    "magic":       ("experiments.baselines.magic.adapter", "MagicAdapter"),
    "deepag":      ("experiments.baselines.ttp_sequence.adapter", "DeepAGAdapter"),
}


def _load(name: str):
    mod_name, cls_name = BASELINES[name]
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)()


def _output_dir(baseline_name: str, dataset_rel: Path) -> Path:
    return config.OUTPUT_BASE_DIR / "baselines" / baseline_name / dataset_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES.keys()),
                    choices=list(BASELINES.keys()))
    ap.add_argument("--scenarios", type=str, default=None,
                    help="...prefix ...")
    args = ap.parse_args()

    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    if args.scenarios:
        datasets = [d for d in datasets
                    if args.scenarios in str(d.relative_to(config.DATASET_FOLDER))]

    print(f"\n{'='*75}")
    print(f"  Baseline Runner")
    print(f"  Baselines: {args.baselines}")
    print(f"  Scenarios: {len(datasets)}")
    print(f"{'='*75}")

    for name in args.baselines:
        try:
            adapter = _load(name)
        except NotImplementedError as e:
            print(f"\n  [SKIP] {name}: {e}")
            continue
        except Exception as e:
            print(f"\n  [INIT FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        for ds in datasets:
            rel = ds.relative_to(config.DATASET_FOLDER).with_suffix("")
            out = _output_dir(name, rel)
            if (out / "result.json").exists():
                print(f"  [skip existing] {name}/{rel}")
                continue
            try:
                pred = adapter.predict(ds)
                saved = adapter.save_result(pred, out)
                print(f"  [OK] {name}/{rel} → {saved}")
            except NotImplementedError as e:
                print(f"  [NOT IMPL] {name}/{rel}: {e}")
                break
            except Exception as e:
                print(f"  [FAIL] {name}/{rel}: {type(e).__name__}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
