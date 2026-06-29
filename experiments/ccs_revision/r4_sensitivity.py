"""
R4 (CCS reviewer ④): Transition-matrix disclosure + parameter sensitivity.

Part 1 -- Dense tactical transition matrix.
    Instantiates the production TacticalScorer (default config) and evaluates
    score(from, to) over ALL 14x14 tactic pairs, materialising the implicit
    rule-based matrix the reviewer asked to see. Emits weights + rule labels.

Part 2 -- Parameter sensitivity.
    Consolidates the existing on-disk alpha/bypass sweep
    (output/v22_alpha_bypass_sweep_results.json) into a clean table so the
    sensitivity of every headline metric to the transition weight alpha is
    disclosed in one place.

Reads only; writes solely under output/_ccs_revision/R4_sensitivity/.

Run:  python -m experiments.ccs_revision.r4_sensitivity
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.attack_chain import TacticalScorer

OUT_DIR = ROOT / "output" / "_ccs_revision" / "R4_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dump_transition_matrix():
    sc = TacticalScorer()
    tactics = sc._TACTICS
    weights, rules, full = {}, {}, []
    for f in tactics:
        weights[f], rules[f] = {}, {}
        for t in tactics:
            r = sc.score(f, t)
            weights[f][t] = round(r.weight, 3)
            rules[f][t] = r.rule
            full.append({
                "from": f, "to": t, "weight": round(r.weight, 3),
                "rule": r.rule, "rule_name": r.rule_name, "note": r.note,
            })

    # weight matrix CSV
    with open(OUT_DIR / "transition_matrix_weights.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["from\\to"] + tactics)
        for f in tactics:
            w.writerow([f] + [weights[f][t] for t in tactics])
    # rule-label matrix CSV
    with open(OUT_DIR / "transition_matrix_rules.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["from\\to"] + tactics)
        for f in tactics:
            w.writerow([f] + [rules[f][t] for t in tactics])
    # rule legend
    legend = dict(sc._RULES)
    with open(OUT_DIR / "rules_legend.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["rule", "name", "weight"])
        for rid, meta in legend.items():
            w.writerow([rid, meta["name"], meta["weight"]])
        for tac, wt in sc._SELF_LOOP_WEIGHTS.items():
            w.writerow([f"R1[{tac}]", f"Self-Loop ({tac})", wt])
        for tac, wt in sc._WILDCARD_IN_WEIGHTS.items():
            w.writerow([f"R4[{tac}]", f"Wildcard-IN ({tac})", wt])
    # full JSON
    with open(OUT_DIR / "transition_matrix.json", "w", encoding="utf-8") as fp:
        json.dump({"tactics": tactics, "weights": weights, "rules": rules, "pairs": full},
                  fp, ensure_ascii=False, indent=2)
    return tactics, weights


def consolidate_alpha_sweep():
    src = ROOT / "output" / "v22_alpha_bypass_sweep_results.json"
    if not src.exists():
        print(f"[R4] alpha sweep not found: {src}"); return None
    data = json.load(open(src, encoding="utf-8"))
    cols = ["variant", "alpha", "bypass_thr", "tech_lcs", "tac_lcs", "step_cov",
            "order", "viterbi_mic", "faiss_mic"]
    rows = [{c: row.get(c) for c in cols} for row in data]
    with open(OUT_DIR / "alpha_sensitivity.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader(); w.writerows(rows)
    return rows


def main():
    tactics, weights = dump_transition_matrix()
    print(f"[R4] transition matrix: {len(tactics)}x{len(tactics)} -> {OUT_DIR}")
    print("     files: transition_matrix_weights.csv / _rules.csv / rules_legend.csv / transition_matrix.json")

    rows = consolidate_alpha_sweep()
    if rows:
        print(f"[R4] alpha sensitivity: {len(rows)} variants -> alpha_sensitivity.csv")
        print(f"     {'variant':<22}{'tech_lcs':>9}{'tac_lcs':>9}{'step_cov':>9}{'order':>8}")
        for r in rows:
            print(f"     {str(r['variant'])[:21]:<22}{r['tech_lcs']:>9.4f}{r['tac_lcs']:>9.4f}"
                  f"{r['step_cov']:>9.4f}{r['order']:>8.4f}")


if __name__ == "__main__":
    main()
