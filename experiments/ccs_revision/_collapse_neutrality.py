# -*- coding: utf-8 -*-
"""Is the parent-child collapse leniency in the acceptable-set metric method-neutral?

Concern: parent/sub-technique collapse in the acceptable-set metric
favours SCOPE (which retrieves sub-techniques). Direct test: measure the
collapse BENEFIT per method, holding alts and normaliser fixed, toggling ONLY
whether root-collapse is allowed.

    delta_collapse(method) = techLCS_norm(collapse=ON) - techLCS_norm(collapse=OFF)

If delta(SCOPE) <= delta(baselines), the leniency does NOT favour SCOPE.
Pure re-scoring of cached on-disk chains -- no inference, no LLM.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from experiments.attack_flows import get_flow
from experiments.chain_align import lcs_length


def match(ref_step, pred, collapse, use_alts):
    """Match pred against a reference step under two toggles.
    use_alts: include curated acceptable set; collapse: allow parent/sub root."""
    if not pred:
        return False
    cands = {ref_step["tid"]}
    if use_alts:
        cands |= set(ref_step.get("alts", []))
    if pred in cands:
        return True
    if collapse:
        pr = pred.split(".")[0]
        return any(c.split(".")[0] == pr for c in cands)
    return False


def tech_lcs_norm(ref_flow, pred_tids, collapse, use_alts=True):
    if not ref_flow:
        return 0.0
    L = lcs_length(ref_flow, pred_tids,
                   eq=lambda r, p: match(r, p, collapse, use_alts))
    return L / len(ref_flow)   # recall-normalised (headline convention)


def scope_chains():
    """(scenario, ref_flow, pred_tids) for SCOPE from per-dataset viterbi json."""
    out = []
    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        vp = config.VITERBI_JSON_PATH
        if not vp.exists():
            continue
        flow = get_flow(config.DATASET_NAME)
        if not flow:
            continue
        try:
            bd = json.load(open(vp, encoding="utf-8"))
        except Exception:
            continue
        pred = [s.get("technique_id") for s in bd if s.get("technique_id")]
        out.append((config.DATASET_NAME, flow, pred))
    return out


def baseline_chains(method):
    base = ROOT / "output" / "baselines" / method
    out = []
    for rp in sorted(base.rglob("result.json")):
        try:
            r = json.load(open(rp, encoding="utf-8"))
        except Exception:
            continue
        scen = r.get("scenario", rp.parent.name)
        flow = get_flow(scen)
        if not flow:
            continue
        pred = [t for t in r.get("technique_sequence", []) if t]
        out.append((scen, flow, pred))
    return out


def summarise(name, chains):
    if not chains:
        print(f"{name:<10} (no chains)")
        return None
    # three settings, same recall normaliser, toggling the two leniencies
    lenient = mean(tech_lcs_norm(f, p, True,  True)  for _, f, p in chains)   # alts + collapse
    no_col  = mean(tech_lcs_norm(f, p, False, True)  for _, f, p in chains)   # alts only
    strict  = mean(tech_lcs_norm(f, p, False, False) for _, f, p in chains)   # neither
    print(f"{name:<8} n={len(chains):<3} "
          f"lenient(alts+collapse)={lenient:.4f}  "
          f"altsOnly={no_col:.4f}  "
          f"STRICT(neither)={strict:.4f}  "
          f"| collapse+{lenient-no_col:+.3f} alts+{no_col-strict:+.3f}")
    return {"lenient": lenient, "no_col": no_col, "strict": strict}


print("=== leniency decomposition, per method (re-scoring only) ===")
res = {}
res["SCOPE"] = summarise("SCOPE", scope_chains())
for m in ["sigma", "magic", "shield", "deepag"]:
    res[m] = summarise(m, baseline_chains(m))

print("-" * 78)
s = res.get("SCOPE")
base = {k: v for k, v in res.items() if k != "SCOPE" and v}
if s and base:
    best_strict = max(v["strict"] for v in base.values())
    best_len = max(v["lenient"] for v in base.values())
    print(f"Ranking under STRICTEST metric (no alts, no collapse):")
    print(f"  SCOPE strict   = {s['strict']:.4f}")
    print(f"  best baseline  = {best_strict:.4f}")
    print(f"  margin         = {s['strict']-best_strict:+.4f}  "
          f"(lenient margin was {s['lenient']-best_len:+.4f})")
    print(f"VERDICT: SCOPE lead {'SURVIVES' if s['strict']>best_strict else 'DOES NOT survive'} "
          f"removing BOTH leniencies -> conclusion independent of the contested protocol.")
