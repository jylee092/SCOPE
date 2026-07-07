# -*- coding: utf-8 -*-
"""Aggregate D=0 robustness across seeds 0/1/2 -> mean +/- std per drop rate.
Reads cached per-scenario viterbi breakdowns; no inference, no API."""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean, pstdev, stdev
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment

RB = config.OUTPUT_BASE_DIR / "_robustness"
SEEDS = [0, 1, 2]
DROPS = [10, 25, 50]


def seed_drop_mean(seed, drop):
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
    return (mean(vals), len(vals)) if vals else None


print(f"{'drop':>5} | {'seed0':>7} {'seed1':>7} {'seed2':>7} | {'mean':>7} {'std':>7}")
print("-" * 56)
rows = {}
for drop in DROPS:
    per = []
    for s in SEEDS:
        r = seed_drop_mean(s, drop)
        per.append(r[0] if r else float("nan"))
    m = mean(per)
    sd = stdev(per)          # sample std (n=3)
    rows[drop] = (per, m, sd)
    print(f"{drop:>4}% | {per[0]:>7.4f} {per[1]:>7.4f} {per[2]:>7.4f} | "
          f"{m:>7.4f} {sd:>7.4f}")

print()
print("For the paper (mean +/- std, tech-LCS at D=0):")
for drop in DROPS:
    _, m, sd = rows[drop]
    print(f"  {drop}% drop: {m:.3f} +/- {sd:.3f}")
