# -*- coding: utf-8 -*-
"""Combined 3-seed aggregation: SCOPE (scope_d0_*) + Sigma/MAGIC/DeepAG.
mean +/- std over seeds {0,1,2} at each drop rate. No inference, no API."""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean, stdev
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
import experiments._robustness_run as rob
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment

RB = config.OUTPUT_BASE_DIR / "_robustness"
SEEDS = [0, 1, 2]
DROPS = [10, 25, 50]


def scope_mean(seed, drop):
    root = RB / f"scope_d0_drop{drop}_seed{seed}"
    if not root.exists():
        return None
    vals = []
    for vp in sorted(root.rglob("*_viterbi.json")):
        stem = vp.name[:-len("_viterbi.json")]
        flow = get_flow(stem)
        if not flow:
            continue
        try:
            bd = json.load(open(vp, encoding="utf-8"))
        except Exception:
            continue
        r = evaluate_chain_alignment(stem, bd, ref_flow=flow) if bd else {}
        vals.append(r.get("technique_lcs_norm", 0.0))
    return mean(vals) if vals else None


def baseline_mean(method, seed, drop):
    root = RB / f"{method}_drop{drop}_seed{seed}"
    if not root.exists():
        return None
    vals = []
    for rp in sorted(root.rglob("result.json")):
        scored = rob._score_alerts_chain_lcs(rp)
        if scored:
            vals.append(scored[1])
    return mean(vals) if vals else None


def agg(getter):
    row = {}
    for drop in DROPS:
        per = [getter(s, drop) for s in SEEDS]
        per = [p for p in per if p is not None]
        if len(per) >= 2:
            row[drop] = (per, mean(per), stdev(per))
        elif per:
            row[drop] = (per, per[0], 0.0)
        else:
            row[drop] = ([], float("nan"), float("nan"))
    return row


methods = {
    "SCOPE": scope_mean,
    "Sigma": lambda s, d: baseline_mean("sigma", s, d),
    "MAGIC": lambda s, d: baseline_mean("magic", s, d),
    "DeepAG": lambda s, d: baseline_mean("deepag", s, d),
}

print(f"{'method':<8}{'drop':>6} | {'seed0':>7}{'seed1':>7}{'seed2':>7} | {'mean':>7}{'std':>7}")
print("-" * 60)
final = {}
for name, fn in methods.items():
    row = agg(fn)
    final[name] = row
    for drop in DROPS:
        per, m, sd = row[drop]
        p = per + [float('nan')] * (3 - len(per))
        print(f"{name:<8}{drop:>5}% | {p[0]:>7.4f}{p[1]:>7.4f}{p[2]:>7.4f} | {m:>7.4f}{sd:>7.4f}")
    print()

print("=== PAPER-READY (mean +/- std, tech-LCS at D=0) ===")
for name in methods:
    parts = []
    for drop in DROPS:
        _, m, sd = final[name][drop]
        parts.append(f"{drop}%: {m:.3f}+/-{sd:.3f}")
    print(f"  {name:<7}: " + "   ".join(parts))
