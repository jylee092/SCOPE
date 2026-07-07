# -*- coding: utf-8 -*-
"""Measure real cold-run per-group description latency (Gemini-2.5-Flash).
Bypasses the cache and times _call_gemini on real prompts built from canonical
feature dicts. Reports median/mean/p90 so we can state a grounded cold-run
overhead bound instead of a generic estimate. Uses ~24 groups.
"""
from __future__ import annotations
import sys, time, json, random
from pathlib import Path
from statistics import mean, median
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from pipeline.mitre_mapper import build_prompt, _call_gemini, GEMINI_MODEL

# API key: prefer env, else .secrets file
key = config.GEMINI_API_KEY
if not key:
    kf = ROOT / ".secrets" / "gemini_key.txt"
    key = kf.read_text(encoding="utf-8").strip() if kf.exists() else ""
assert key, "no Gemini API key"

N = 24
INTER_CALL_SLEEP = 1.5   # politeness; NOT counted in per-call latency

# collect canonical feature dicts (skip ablation/robustness/revision dirs)
feats = []
skip = ("ablation", "_robustness", "_ccs_revision", "_cache", "baselines")
for fp in sorted((ROOT / "output").rglob("*_feature_result.json")):
    if any(s in str(fp) for s in skip):
        continue
    try:
        arr = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    for feat in arr:
        if isinstance(feat, dict) and feat.get("features"):
            feats.append(feat)

rng = random.Random(0)
rng.shuffle(feats)
feats = feats[:N]
print(f"model={GEMINI_MODEL}  sampled {len(feats)} groups for cold-run timing", flush=True)

lat = []
for i, feat in enumerate(feats, 1):
    prompt = build_prompt(feat)
    t0 = time.perf_counter()
    try:
        txt = _call_gemini(prompt, key)
    except Exception as e:
        print(f"  [{i}] FAILED {type(e).__name__}: {e}", flush=True)
        continue
    dt = time.perf_counter() - t0
    lat.append(dt)
    print(f"  [{i:>2}] {dt:6.2f}s  prompt~{len(prompt)}c resp~{len(txt)}c", flush=True)
    time.sleep(INTER_CALL_SLEEP)

if lat:
    lat_sorted = sorted(lat)
    p90 = lat_sorted[int(0.9 * (len(lat_sorted) - 1))]
    print("\n=== cold-run per-group description latency (Gemini-2.5-Flash) ===")
    print(f"  n={len(lat)}  median={median(lat):.2f}s  mean={mean(lat):.2f}s  "
          f"min={min(lat):.2f}s  max={max(lat):.2f}s  p90={p90:.2f}s")
    groups_per_scen = 779 / 35
    print(f"  per-scenario groups (mean) = {groups_per_scen:.0f}")
    print(f"  => cold-run LLM overhead/scenario: "
          f"median {median(lat)*groups_per_scen:.0f}s, "
          f"p90 {p90*groups_per_scen:.0f}s "
          f"(~{median(lat)*groups_per_scen/60:.1f}-{p90*groups_per_scen/60:.1f} min)")
    out = ROOT / "output" / "_coldrun_latency.json"
    json.dump({"n": len(lat), "median": median(lat), "mean": mean(lat),
               "min": min(lat), "max": max(lat), "p90": p90,
               "groups_per_scenario": groups_per_scen,
               "latencies": lat, "model": GEMINI_MODEL},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"  saved {out}")
