"""
Strict-metric supplement for §7 (CCS reviewer defense).

Computes three additional metrics that complement the plausibility-based
chain-LCS already reported:

  1. Strict-TID LCS -- same as technique-LCS but with exact-string matching
     (no parent/sub-technique collapse). Pessimistic counterpart.
  2. Per-step Precision / Recall / F1 -- set-based agreement between the
     predicted chain's TID multi-set and the reference flow's TID set.
  3. False chain rate -- fraction of scenarios whose predicted chain has
     NO matched reference step (i.e., where SCOPE produces a chain that
     does not overlap with any reference TTP at all).

All three are computed from existing on-disk output/.../<scenario>_viterbi.json
files plus the reference flows in experiments.attack_flows. No new
inference run is needed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from experiments.attack_flows import get_flow
from experiments.chain_align import lcs_length


# --- LCS variants -----------------------------------------------------------

def strict_tid_lcs(ref_flow, pred_tids):
    """Exact-string LCS between ref TIDs and pred TIDs.
    No parent/sub-technique collapse, no alts."""
    ref_tids = [s["tid"] for s in ref_flow]
    if not ref_tids or not pred_tids:
        return 0.0
    L = lcs_length(ref_tids, pred_tids)
    return L / max(len(ref_tids), len(pred_tids))


# --- Per-step precision/recall/F1 (set-based) -------------------------------

def step_prf(ref_flow, pred_tids):
    """Set-based precision / recall / F1 over TIDs.

    Reference set = primary tids + alts (the acceptable set).
    Prediction set = unique TIDs in pred chain.
    A predicted TID is correct iff it equals an acceptable TID exactly
    OR shares a parent with one of them (parent-collapsed).

    We report two precisions/recalls:
      - strict (exact): no parent collapse
      - plausible (parent-collapsed): consistent with body-text Hit@5
    """
    if not pred_tids:
        return {"strict": (0, 0, 0), "plausible": (0, 0, 0)}

    accept = set()
    for s in ref_flow:
        accept.add(s["tid"])
        for a in s.get("alts", []):
            accept.add(a)

    pred_set = set(t for t in pred_tids if t)

    # strict
    tp_strict = len(pred_set & accept)
    p_strict  = tp_strict / len(pred_set) if pred_set else 0.0
    r_strict  = tp_strict / len(accept)   if accept   else 0.0
    f_strict  = 2 * p_strict * r_strict / (p_strict + r_strict) \
                if (p_strict + r_strict) else 0.0

    # plausible (parent-collapse on either side)
    accept_roots = {a.split(".")[0] for a in accept}
    pred_roots   = {p.split(".")[0] for p in pred_set}
    tp_plaus = len(pred_roots & accept_roots)
    p_plaus  = tp_plaus / len(pred_roots) if pred_roots else 0.0
    r_plaus  = tp_plaus / len(accept_roots) if accept_roots else 0.0
    f_plaus  = 2 * p_plaus * r_plaus / (p_plaus + r_plaus) \
               if (p_plaus + r_plaus) else 0.0

    return {
        "strict":    (round(p_strict, 4),  round(r_strict, 4),  round(f_strict, 4)),
        "plausible": (round(p_plaus, 4),   round(r_plaus, 4),   round(f_plaus, 4)),
    }


# --- False chain rate -------------------------------------------------------

def is_false_chain(ref_flow, pred_tids):
    """A predicted chain is 'false' if NO predicted TID matches any
    acceptable reference TID (under parent-collapse, the most lenient
    matching). I.e., a chain that bears no relation to the reference."""
    if not pred_tids:
        return True  # empty chain counts as miss
    accept_roots = {s["tid"].split(".")[0] for s in ref_flow}
    for s in ref_flow:
        for a in s.get("alts", []):
            accept_roots.add(a.split(".")[0])
    pred_roots = {p.split(".")[0] for p in pred_tids if p}
    return not (pred_roots & accept_roots)


# --- Main: walk all scenarios -----------------------------------------------

def main():
    rows = []
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    for ds in datasets:
        config.configure_dataset(ds)
        vit_p = config.VITERBI_JSON_PATH
        if not vit_p.exists():
            continue
        try:
            breakdown = json.load(open(vit_p, encoding="utf-8"))
        except Exception:
            continue
        scenario = config.DATASET_NAME
        flow = get_flow(scenario)
        if not flow:
            continue
        pred_tids = [s.get("technique_id") for s in breakdown if s.get("technique_id")]

        strict = strict_tid_lcs(flow, pred_tids)
        prf    = step_prf(flow, pred_tids)
        false  = is_false_chain(flow, pred_tids)

        rows.append({
            "scenario": scenario,
            "ref_steps": len(flow),
            "pred_steps": len(pred_tids),
            "strict_tid_lcs": round(strict, 4),
            "p_strict":    prf["strict"][0],
            "r_strict":    prf["strict"][1],
            "f1_strict":   prf["strict"][2],
            "p_plausible": prf["plausible"][0],
            "r_plausible": prf["plausible"][1],
            "f1_plausible":prf["plausible"][2],
            "false_chain": int(false),
        })

    print(f"{'scenario':<55} {'strict-LCS':>10} "
          f"{'P_str':>6} {'R_str':>6} {'F1_str':>7} "
          f"{'P_pla':>6} {'R_pla':>6} {'F1_pla':>7} "
          f"{'false':>5}")
    print('-' * 130)
    for r in rows:
        print(f"{r['scenario'][:54]:<55} {r['strict_tid_lcs']:>10.4f} "
              f"{r['p_strict']:>6.3f} {r['r_strict']:>6.3f} {r['f1_strict']:>7.3f} "
              f"{r['p_plausible']:>6.3f} {r['r_plausible']:>6.3f} {r['f1_plausible']:>7.3f} "
              f"{r['false_chain']:>5}")

    n = len(rows)
    if n == 0:
        print("no rows"); return
    macro_strict_lcs = mean(r["strict_tid_lcs"] for r in rows)
    macro_f1_strict  = mean(r["f1_strict"] for r in rows)
    macro_p_strict   = mean(r["p_strict"] for r in rows)
    macro_r_strict   = mean(r["r_strict"] for r in rows)
    macro_f1_plaus   = mean(r["f1_plausible"] for r in rows)
    macro_p_plaus    = mean(r["p_plausible"] for r in rows)
    macro_r_plaus    = mean(r["r_plausible"] for r in rows)
    fc_rate          = sum(r["false_chain"] for r in rows) / n

    print('=' * 130)
    print(f"Macro averages over {n} scenarios:")
    print(f"  Strict-TID LCS              : {macro_strict_lcs:.4f}")
    print(f"  Per-step Precision (strict) : {macro_p_strict:.4f}")
    print(f"  Per-step Recall    (strict) : {macro_r_strict:.4f}")
    print(f"  Per-step F1        (strict) : {macro_f1_strict:.4f}")
    print(f"  Per-step Precision (plaus.) : {macro_p_plaus:.4f}")
    print(f"  Per-step Recall    (plaus.) : {macro_r_plaus:.4f}")
    print(f"  Per-step F1        (plaus.) : {macro_f1_plaus:.4f}")
    print(f"  False chain rate            : {fc_rate:.4f}  ({sum(r['false_chain'] for r in rows)}/{n})")

    out = ROOT / "output" / "_strict_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_scenarios": n,
        "macro": {
            "strict_tid_lcs":  round(macro_strict_lcs, 4),
            "p_strict":        round(macro_p_strict, 4),
            "r_strict":        round(macro_r_strict, 4),
            "f1_strict":       round(macro_f1_strict, 4),
            "p_plausible":     round(macro_p_plaus, 4),
            "r_plausible":     round(macro_r_plaus, 4),
            "f1_plausible":    round(macro_f1_plaus, 4),
            "false_chain_rate":round(fc_rate, 4),
        },
        "rows": rows,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
