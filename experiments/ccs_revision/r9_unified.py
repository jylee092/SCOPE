"""
R9 (CCS reviewers ③ + ⑨): ONE unified comparison table -- one ground truth,
one ruler, all methods. Resolves the "multiple inconsistent metric scripts"
problem by reporting every metric for SCOPE + every baseline together.

Two metric families, both applied uniformly:

  CHAIN reconstruction (reused from R8; GT = attack_flows reference; family
  match for recall, exact for strict):
    recall_LCS  : family-LCS / len(ref)        (lenient -- current headline)
    strict_LCS  : exact-LCS  / max(len)        (pessimistic -- reviewer ③)
    lenpen_LCS  : family-LCS / max(len)        (over-length penalised)
    F1_strict / F1_plaus : per-step set-based P/R/F1
    false_chain : fraction of scenarios with no reference overlap
  (empty baseline chains are scored as failures, not dropped.)

  PER-GROUP mapping (the native metric for mapper-style baselines such as
  SHIELD; plausibility Hit@5 over the 251 TP groups, identical definition
  across methods):
    SCOPE     : R7 self_metrics.json  (mapping plausible H@5)
    baselines : output/baselines/<m>/_scores.json  (hit@5)

Writes output/_ccs_revision/R9_unified/unified_compare.{json,csv}.
Run:  python -m experiments.ccs_revision.r9_unified
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from experiments.ccs_revision.r8_baseline_strict import (
    load_scope, load_baseline, score_method, BASELINES,
)

OUT_DIR = ROOT / "output" / "_ccs_revision" / "R9_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mapping_hit5():
    """Per-group plausibility Hit@5 (macro, micro) for every method."""
    out = {}
    # SCOPE from R7
    r7 = ROOT / "output" / "_ccs_revision" / "R7_self_metrics" / "self_metrics.json"
    if r7.exists():
        d = json.load(open(r7, encoding="utf-8"))
        out["SCOPE"] = {"hit5_macro": d["mapping_macro"]["plausible_h5"],
                        "hit5_micro": d["mapping_micro"]["plausible_h5"]}
    # baselines from their native _scores.json
    for b in BASELINES:
        sp = ROOT / "output" / "baselines" / b / "_scores.json"
        if sp.exists():
            d = json.load(open(sp, encoding="utf-8"))
            out[b] = {"hit5_macro": round(d.get("macro", {}).get("hit@5") or 0, 4),
                      "hit5_micro": round(d.get("micro_hit_k") or 0, 4)}
    return out


def main():
    # ---- CHAIN metrics (one ruler, all methods) ----
    methods = {"SCOPE": load_scope()}
    for b in BASELINES:
        seq = load_baseline(b)
        if seq:
            methods[b] = seq
    chain = {}
    for name, seq in methods.items():
        r = score_method(name, seq)
        if r:
            chain[name] = r["macro"]

    # ---- MAPPING Hit@5 ----
    mapping = mapping_hit5()

    order = ["SCOPE"] + [b for b in BASELINES if b in chain]
    unified = {}
    for m in order:
        c = chain.get(m, {})
        mp = mapping.get(m, {})
        unified[m] = {
            "n": c.get("n"),
            "chain_recall_LCS": c.get("tech_lcs_recall"),
            "chain_strict_LCS": c.get("tech_lcs_strict"),
            "chain_lenpen_LCS": c.get("tech_lcs_lenpen"),
            "chain_F1_strict": c.get("f1_strict"),
            "chain_F1_plaus": c.get("f1_plaus"),
            "chain_false_rate": c.get("false_chain_rate"),
            "map_hit5_macro": mp.get("hit5_macro"),
            "map_hit5_micro": mp.get("hit5_micro"),
        }

    json.dump(unified, open(OUT_DIR / "unified_compare.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(OUT_DIR / "unified_compare.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["method"] + list(next(iter(unified.values())).keys())
        w = csv.writer(f); w.writerow(cols)
        for m in order:
            w.writerow([m] + [unified[m][k] for k in cols[1:]])

    print(f"[R9] unified comparison (one GT, one ruler, all methods) -> {OUT_DIR}")
    print("\n  == CHAIN reconstruction ==")
    print(f"  {'method':<10}{'n':>4}{'recallLCS':>10}{'strictLCS':>10}{'lenpenLCS':>10}{'F1_str':>8}{'false%':>8}")
    for m in order:
        u = unified[m]
        print(f"  {m:<10}{u['n']:>4}{u['chain_recall_LCS']:>10.3f}{u['chain_strict_LCS']:>10.3f}"
              f"{u['chain_lenpen_LCS']:>10.3f}{u['chain_F1_strict']:>8.3f}{u['chain_false_rate']*100:>7.1f}%")
    print("\n  == PER-GROUP mapping (plausibility Hit@5) ==")
    print(f"  {'method':<10}{'Hit@5 macro':>13}{'Hit@5 micro':>13}")
    for m in order:
        u = unified[m]
        ma = u['map_hit5_macro']; mi = u['map_hit5_micro']
        print(f"  {m:<10}{(f'{ma:.3f}' if ma is not None else '--'):>13}{(f'{mi:.3f}' if mi is not None else '--'):>13}")


if __name__ == "__main__":
    main()
