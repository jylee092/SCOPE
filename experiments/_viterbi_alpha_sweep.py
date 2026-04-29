"""
Viterbi transition_weight (α) sweep.

TTP_MAPPING + features 캐시를 그대로 두고, α 값만 바꿔가며 Viterbi만 재계산 →
per-group TTP hit rate를 측정하고 결과 테이블을 출력한다.

사용:
  python experiments/_viterbi_alpha_sweep.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

from experiments._rerun_viterbi_only import run_one  # noqa: E402
from experiments.run_eval_post_viterbi import _run as run_post_viterbi_eval  # noqa: E402


ALPHA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def rerun_all_scenarios() -> int:
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    ok = 0
    for i, ds in enumerate(datasets, 1):
        try:
            if run_one(ds):
                ok += 1
        except Exception as e:
            print(f"  [{i}] ERROR {ds.name}: {type(e).__name__}: {e}")
    return ok


def aggregate_from_csv(csv_path: Path) -> tuple[int, float, float, float]:
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows) or 1
    t5 = sum(int(r["top5_hit"]) for r in rows) / n
    f1 = sum(int(r["faiss_hit"]) for r in rows) / n
    vp = sum(int(r["viterbi_hit"]) for r in rows) / n
    return len(rows), t5, f1, vp


def main():
    summary = []
    for alpha in ALPHA_GRID:
        print(f"\n{'='*70}\n  α = {alpha}\n{'='*70}")
        config.VITERBI_TRANSITION_WEIGHT = alpha
        ok = rerun_all_scenarios()
        print(f"  Viterbi re-run: {ok} scenarios OK")

        run_post_viterbi_eval(strong_only=False, label=f"ALL α={alpha}")

        csv_all = config.OUTPUT_BASE_DIR / "eval_post_viterbi_all.csv"
        n, t5, f1, vp = aggregate_from_csv(csv_all)
        summary.append((alpha, n, t5, f1, vp))

    print(f"\n\n{'='*70}\n  SWEEP SUMMARY (ALL TPs, n=238)\n{'='*70}")
    print(f"  {'α':>6}  {'Top-5':>8}  {'FAISS-1':>8}  {'Viterbi':>8}  {'Δ(V-F)':>8}")
    print("  " + "-"*56)
    for alpha, n, t5, f1, vp in summary:
        print(f"  {alpha:>6.2f}  {t5:>8.4f}  {f1:>8.4f}  {vp:>8.4f}  {vp-f1:>+8.4f}")


if __name__ == "__main__":
    main()
