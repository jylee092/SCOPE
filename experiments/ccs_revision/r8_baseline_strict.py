"""
R8 (CCS reviewers ③ + ⑨): apply the SAME strict / standard / plausibility chain
metrics to ALL methods (SCOPE + every baseline), not just SCOPE.

Reviewers ③ and ⑨ both demand: report standard precision-oriented metrics under
one yardstick across all methods and show SCOPE still wins. R3 only scored SCOPE.
This closes that gap.

For each method, per scenario, from its technique sequence we compute:
  - tech_lcs_recall   : family-LCS / len(ref)          (current main metric)
  - tech_lcs_strict   : exact-string LCS / max(len)     (pessimistic)
  - tech_lcs_lenpen   : family-LCS / max(len(ref),len(pred))  (over-length penalised)
  - tac_lcs, step_cov, order  (chain_align)
  - per-step P/R/F1    : strict + parent-collapsed (set-based vs acceptable set)
  - false_chain_rate

Methods:
  SCOPE     : output/<rel>/<stem>_viterbi.json   (technique_id sequence)
  baselines : output/baselines/<name>/**/result.json  (technique_sequence)

Writes solely under output/_ccs_revision/R8_baseline_strict/.
Run:  python -m experiments.ccs_revision.r8_baseline_strict
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from experiments._strict_metrics import strict_tid_lcs, step_prf, is_false_chain
from experiments.chain_align import evaluate_chain_alignment, lcs_length, step_match
from experiments.attack_flows import get_flow

OUT_DIR = ROOT / "output" / "_ccs_revision" / "R8_baseline_strict"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINES = ["sigma", "shield", "ttp_sequence", "deepag", "magic"]


def load_scope():
    """scenario -> technique sequence, from SCOPE viterbi outputs."""
    out = {}
    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        vp = config.VITERBI_JSON_PATH
        if vp.exists():
            bd = json.load(open(vp, encoding="utf-8"))
            out[config.DATASET_NAME] = [s.get("technique_id") for s in bd if s.get("technique_id")]
    return out


def load_baseline(name):
    """scenario -> technique sequence, from a baseline's result.json files."""
    out = {}
    base = ROOT / "output" / "baselines" / name
    for rp in base.rglob("result.json"):
        try:
            d = json.load(open(rp, encoding="utf-8"))
        except Exception:
            continue
        sc = d.get("scenario")
        if sc is not None:
            # keep EMPTY sequences too -- an empty chain is a reconstruction
            # FAILURE and must be scored as a miss, not silently dropped.
            out[sc] = [t for t in (d.get("technique_sequence") or []) if t]
    return out


def lenpen_lcs(ref_flow, pred_tids):
    if not ref_flow or not pred_tids:
        return 0.0
    L = lcs_length(ref_flow, pred_tids, eq=lambda r, p: step_match(r, p))
    return L / max(len(ref_flow), len(pred_tids))


def score_method(name, seq_by_scen, only=None):
    rows = []
    for scen, pred in seq_by_scen.items():
        if only is not None and scen not in only:
            continue
        flow = get_flow(scen)
        if not flow:
            continue
        ca = evaluate_chain_alignment(scen, [{"technique_id": t, "tactic": ""} for t in pred], flow)
        if "error" in ca:
            continue
        prf = step_prf(flow, pred)
        rows.append({
            "scenario": scen,
            "tech_lcs_recall": ca["technique_lcs_norm"],
            "tech_lcs_strict": round(strict_tid_lcs(flow, pred), 4),
            "tech_lcs_lenpen": round(lenpen_lcs(flow, pred), 4),
            "tac_lcs": ca["tactic_lcs_norm"],
            "step_cov": ca["step_coverage"],
            "p_strict": prf["strict"][0], "r_strict": prf["strict"][1], "f1_strict": prf["strict"][2],
            "p_plaus": prf["plausible"][0], "r_plaus": prf["plausible"][1], "f1_plaus": prf["plausible"][2],
            "false_chain": int(is_false_chain(flow, pred)),
        })
    if not rows:
        return None
    keys = ["tech_lcs_recall", "tech_lcs_strict", "tech_lcs_lenpen", "tac_lcs", "step_cov",
            "p_strict", "r_strict", "f1_strict", "p_plaus", "r_plaus", "f1_plaus"]
    macro = {k: round(mean(r[k] for r in rows), 4) for k in keys}
    macro["false_chain_rate"] = round(mean(r["false_chain"] for r in rows), 4)
    macro["n"] = len(rows)
    return {"macro": macro, "rows": rows}


def main():
    methods = {"SCOPE": load_scope()}
    for b in BASELINES:
        seq = load_baseline(b)
        if seq:
            methods[b] = seq

    results = {}
    for name, seq in methods.items():
        r = score_method(name, seq)
        if r:
            results[name] = r

    json.dump({m: results[m]["macro"] for m in results},
              open(OUT_DIR / "r8_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # full per-scenario dump
    for m in results:
        with open(OUT_DIR / f"{m}_per_scenario.csv", "w", newline="", encoding="utf-8") as f:
            rows = results[m]["rows"]
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- HEAD-TO-HEAD on shared scenarios (fair: same scenario set) ----
    scope_seq = methods["SCOPE"]
    h2h = {}
    for b in BASELINES:
        if b not in methods:
            continue
        shared = sorted(set(scope_seq) & set(methods[b]))
        if not shared:
            continue
        s = score_method("SCOPE", scope_seq, only=shared)
        bb = score_method(b, methods[b], only=shared)
        if s and bb:
            h2h[b] = {"n_shared": len(shared), "SCOPE": s["macro"], b: bb["macro"]}
    json.dump(h2h, open(OUT_DIR / "r8_head2head.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"[R8] strict/standard metrics -> {OUT_DIR}")
    print("\n  == Each method on its OWN coverage (n varies -> NOT directly comparable) ==")
    hdr = f"  {'method':<14}{'n':>4}{'LCS_rec':>9}{'LCS_strict':>11}{'LCS_lenpen':>11}{'F1_str':>8}{'F1_pla':>8}{'false%':>8}"
    print(hdr)
    for m in ["SCOPE"] + [b for b in BASELINES if b in results]:
        x = results[m]["macro"]
        print(f"  {m:<14}{x['n']:>4}{x['tech_lcs_recall']:>9.3f}{x['tech_lcs_strict']:>11.3f}"
              f"{x['tech_lcs_lenpen']:>11.3f}{x['f1_strict']:>8.3f}{x['f1_plaus']:>8.3f}"
              f"{x['false_chain_rate']*100:>7.1f}%")

    print("\n  == HEAD-TO-HEAD on SHARED scenarios (fair comparison) ==")
    for b, d in h2h.items():
        print(f"  -- SCOPE vs {b}  (shared n={d['n_shared']}) --")
        for who in ("SCOPE", b):
            x = d[who]
            print(f"     {who:<10}{'':>3}LCS_rec {x['tech_lcs_recall']:.3f} | "
                  f"LCS_strict {x['tech_lcs_strict']:.3f} | LCS_lenpen {x['tech_lcs_lenpen']:.3f} | "
                  f"F1_str {x['f1_strict']:.3f} | false {x['false_chain_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
