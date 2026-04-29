"""
Viterbi structural-inference tuning sweep.

For each (max_skip, trans_w, self_loop, margin_gated) combo:
  1. Override the four config values at runtime
  2. Re-run Viterbi across all 35 scenarios (ttp_mapping+feature caches reused)
  3. Evaluate chain metrics (technique-LCS, tactic-LCS, step coverage, order acc)
     and per-group post-Viterbi plausibility (Top-5 hit, FAISS top-1, Viterbi pick)

Outputs a single table printed to stdout + saves results to
output/viterbi_tune_sweep_results.json.
"""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from experiments import _rerun_viterbi_only as vrerun

# Evaluators (invoked as subprocess is simpler than importing their mains).
import subprocess


# Sweep configs: (max_skip, trans_w, self_loop, margin_gated, label)
# Rationale: isolate each change's contribution incrementally from baseline.
COMBOS = [
    (0, 0.1, 1.0, False, "v7_base          (baseline: D=0, tw=0.1)"),
    (0, 0.3, 1.0, False, "tw=0.3"),
    (0, 0.5, 1.0, False, "tw=0.5"),
    (0, 0.7, 1.0, False, "tw=0.7"),
    (0, 1.0, 1.0, False, "tw=1.0  (equal-weight trans vs emission)"),
    (0, 0.5, 0.4, False, "tw=0.5 sl=0.4"),
    (0, 0.5, 0.3, False, "tw=0.5 sl=0.3"),
]


@contextmanager
def override_config(**overrides):
    """Temporarily override config module attributes."""
    saved = {k: getattr(config, k, None) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None and not hasattr(config, k):
                continue
            setattr(config, k, v)


def run_eval_script(module: str) -> str:
    """Run an eval script as subprocess, return stdout tail."""
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(ROOT),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"  [eval {module} ERR] {result.stderr[-400:]}")
    return (result.stdout or "") + (result.stderr or "")


def parse_v2_agg(text: str) -> dict:
    """Extract aggregate chain metrics from run_eval_v2 stdout."""
    # read saved JSON directly for precision
    p = config.OUTPUT_BASE_DIR / "eval_v2_results.json"
    if not p.exists(): return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    import statistics as st
    ch = lambda k: st.mean(s["chain"].get(k, 0) for s in data) if data else 0
    ttp_h1 = [s["ttp"].get("hit_at_1", 0) for s in data if s["ttp"].get("n", 0) > 0]
    ttp_h5 = [s["ttp"].get("hit_at_5", 0) for s in data if s["ttp"].get("n", 0) > 0]
    return {
        "scenarios": len(data),
        "tech_lcs": ch("technique_lcs_norm"),
        "tac_lcs": ch("tactic_lcs_norm"),
        "step_cov": ch("step_coverage"),
        "order_acc": ch("order_accuracy"),
        "ttp_h1": sum(ttp_h1)/len(ttp_h1) if ttp_h1 else 0,
        "ttp_h5": sum(ttp_h5)/len(ttp_h5) if ttp_h5 else 0,
    }


def parse_post_viterbi() -> dict:
    """Compute micro metrics from eval_post_viterbi_all.csv."""
    import csv
    p = config.OUTPUT_BASE_DIR / "eval_post_viterbi_all.csv"
    if not p.exists(): return {}
    rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r]
    n = len(rows)
    if not n: return {}
    return {
        "n_groups": n,
        "top5_mic": sum(int(r["top5_hit"]) for r in rows) / n,
        "faiss_mic": sum(int(r["faiss_hit"]) for r in rows) / n,
        "viterbi_mic": sum(int(r["viterbi_hit"]) for r in rows) / n,
        "delta_mic": (sum(int(r["viterbi_hit"]) for r in rows) - sum(int(r["faiss_hit"]) for r in rows)) / n,
    }


def run_combo(max_skip, trans_w, self_loop, margin_gated, label):
    print(f"\n{'='*100}\n  {label}\n{'='*100}")
    print(f"  → max_skip={max_skip}, trans_w={trans_w}, self_loop={self_loop}, margin_gated={margin_gated}")
    overrides = dict(
        VITERBI_MAX_SKIP=max_skip,
        VITERBI_TRANSITION_WEIGHT=trans_w,
        SELF_LOOP_TID_PENALTY=self_loop,
        VITERBI_MARGIN_GATED_ALPHA=margin_gated,
    )
    with override_config(**overrides):
        # 1) Re-run Viterbi for all scenarios (writes viterbi.json)
        datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
        ok = 0
        for ds in datasets:
            try:
                if vrerun.run_one(ds): ok += 1
            except Exception as e:
                print(f"  [vit ERR] {ds.name}: {type(e).__name__}: {e}")
        print(f"  viterbi recomputed: {ok}/{len(datasets)}")

        # 2) Run evaluators (overwrites eval_v2_results.json, eval_post_viterbi_all.csv)
        run_eval_script("experiments.run_eval_v2")
        run_eval_script("experiments.run_eval_post_viterbi")

    # 3) Parse both
    agg = parse_v2_agg("")
    pv  = parse_post_viterbi()
    result = {"label": label.strip(), "params": overrides, **agg, **pv}
    # Compact one-line print
    print(f"  → tech_lcs={agg.get('tech_lcs',0):.4f}  tac_lcs={agg.get('tac_lcs',0):.4f}  "
          f"step_cov={agg.get('step_cov',0):.4f}  order={agg.get('order_acc',0):.4f}")
    print(f"  → top5_mic={pv.get('top5_mic',0):.4f}  FAISS_mic={pv.get('faiss_mic',0):.4f}  "
          f"Vit_mic={pv.get('viterbi_mic',0):.4f}  Δ={pv.get('delta_mic',0):+.4f}")
    return result


def main():
    results = []
    for params in COMBOS:
        try:
            r = run_combo(*params)
            results.append(r)
        except Exception as e:
            print(f"  [COMBO FATAL] {params[-1]}: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"label": params[-1], "error": str(e)})

    # Write summary
    out_path = config.OUTPUT_BASE_DIR / "viterbi_tune_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print final table
    print(f"\n{'='*120}\n  SWEEP SUMMARY\n{'='*120}")
    hdr = f"{'config':<48s} {'tech_LCS':>9s} {'tac_LCS':>8s} {'step_cov':>9s} {'order':>7s} {'top5_mic':>9s} {'Vit_mic':>8s} {'Δ':>7s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['label']:<48s} ERR: {r['error']}")
            continue
        print(f"{r['label']:<48s} "
              f"{r.get('tech_lcs',0):>9.4f} {r.get('tac_lcs',0):>8.4f} "
              f"{r.get('step_cov',0):>9.4f} {r.get('order_acc',0):>7.4f} "
              f"{r.get('top5_mic',0):>9.4f} {r.get('viterbi_mic',0):>8.4f} "
              f"{r.get('delta_mic',0):>+7.4f}")
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
