# -*- coding: utf-8 -*-
"""Multi-seed robustness for the 3 cheap baselines (Sigma/MAGIC/DeepAG) on
seeds 1,2 (seed0 already done). No LLM. MAGIC/DeepAG read Sigma's per-seed
output, so Sigma runs first. Reuses pre-generated dropped scenarios.
SHIELD is intentionally excluded (LLM + degenerate empty-chain curve)."""
from __future__ import annotations
import sys
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import experiments._robustness_run as rob

SEEDS = [1, 2]
DROPS = [0.10, 0.25, 0.50]


def m(d):
    return mean(d.values()) if d else 0.0


results = {}
for seed in SEEDS:
    for drop in DROPS:
        pct = int(round(drop * 100))
        # order matters: sigma -> magic/deepag (they read sigma per-seed output)
        s = rob.run_sigma(drop, seed)
        g = rob.run_magic(drop, seed)
        a = rob.run_deepag(drop, seed)
        results[(seed, pct)] = {"sigma": m(s), "magic": m(g), "deepag": m(a),
                                 "n": len(s)}
        print(f"seed{seed} {pct:>2}% drop | "
              f"Sigma={m(s):.4f}  MAGIC={m(g):.4f}  DeepAG={m(a):.4f}  "
              f"(n={len(s)})", flush=True)

print("\n=== DONE ===", flush=True)
