"""
Signature rerank weight (w_sig) sweep.

search_similar 안의 signature-based multiplicative boost 가중치를 바꿔가며
FAISS+BM25+signature → Viterbi → eval 을 반복 수행.

LLM description 및 FAISS 인덱스는 캐시에서 재사용, 매 iter 는 search_similar
의 rerank 만 다시 돈다.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

import subprocess
from experiments._rerun_viterbi_only import run_one as viterbi_run_one  # noqa: E402
from experiments.run_eval_post_viterbi import _run as run_post_viterbi_eval  # noqa: E402

def auto_label_main():
    subprocess.run([sys.executable, "-m", "experiments.auto_label_gt"], check=True, cwd=str(ROOT))


WEIGHT_GRID = [0.0, 0.4, 0.8, 1.2, 1.6]


def run_main_pipeline_for_all_scenarios() -> int:
    """
    main.py run_all 과 동일한 iteration 이지만 in-process 로 수행하여
    싱글톤 (FAISS index, embed model, semantic scorer) 을 공유한다.
    """
    from main import run_pipeline  # lazy import
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    ok = 0
    for ds in datasets:
        config.configure_dataset(ds)
        try:
            run_pipeline()
            ok += 1
        except Exception as e:
            print(f"  [err] {config.DATASET_NAME}: {type(e).__name__}: {e}")
    return ok


def clear_ttp_and_viterbi() -> None:
    base = config.OUTPUT_BASE_DIR
    for p in base.rglob("*_ttp_mapping.json"):
        if "_pre_t1112" in str(p) or "ablation" in str(p) or "_snapshot" in str(p):
            continue
        p.unlink()
    for p in base.rglob("*_viterbi.json"):
        if "_pre_t1112" in str(p) or "ablation" in str(p) or "_snapshot" in str(p):
            continue
        p.unlink()


def aggregate_from_csv(path: Path) -> tuple[int, float, float, float]:
    rows = list(csv.DictReader(open(path)))
    n = len(rows) or 1
    t5 = sum(int(r["top5_hit"]) for r in rows) / n
    f1 = sum(int(r["faiss_hit"]) for r in rows) / n
    vp = sum(int(r["viterbi_hit"]) for r in rows) / n
    return len(rows), t5, f1, vp


def main():
    summary = []
    for w in WEIGHT_GRID:
        print(f"\n{'='*70}\n  w_sig = {w}\n{'='*70}")
        config.SIGNATURE_WEIGHT = w

        clear_ttp_and_viterbi()
        t0 = time.time()
        ok = run_main_pipeline_for_all_scenarios()
        print(f"  pipeline: {ok} scenarios OK ({time.time()-t0:.0f}s)")

        # annotation 은 매 run 마다 재생성되므로 auto-label 로 GT 복구.
        print("  auto-labeling...")
        auto_label_main()

        print("  eval...")
        run_post_viterbi_eval(strong_only=False, label=f"ALL w={w}")

        csv_all = config.OUTPUT_BASE_DIR / "eval_post_viterbi_all.csv"
        n, t5, f1, vp = aggregate_from_csv(csv_all)
        summary.append((w, n, t5, f1, vp))

    print(f"\n\n{'='*70}\n  SIGNATURE WEIGHT SWEEP SUMMARY\n{'='*70}")
    print(f"  {'w_sig':>6}  {'n':>4}  {'Top-5':>8}  {'FAISS-1':>8}  {'Viterbi':>8}  {'Δ(V-F)':>8}")
    print("  " + "-"*56)
    for w, n, t5, f1, vp in summary:
        print(f"  {w:>6.2f}  {n:>4d}  {t5:>8.4f}  {f1:>8.4f}  {vp:>8.4f}  {vp-f1:>+8.4f}")


if __name__ == "__main__":
    main()
