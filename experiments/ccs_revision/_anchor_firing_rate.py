"""Anchor firing rate: does the anchor rule set R reach the GT attack steps?

Separates ANCHOR REACH (did a rule fire with the right technique) from
downstream label correctness. No new inference -- reads existing annotations.

Metrics (macro over scenarios, roots = parent technique of each ref step):
  - scenario_firing    : fraction of scenarios with >=1 TP group (anchor fired
                         on a real attack step somewhere)
  - rule_reach         : fraction of GT ref roots for which >=1 anchor fired
                         whose RULE prior family-matches that root
  - tp_reach (=complete): fraction of GT ref roots covered by a correctly
                         labelled TP group (== paper 'completeness')
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from experiments.attack_flows import get_flow, all_acceptable_tids


def _root(t: str) -> str:
    return (t or "").split(".")[0]


def _is_true(v) -> bool:
    return v is True or str(v).lower() == "true"


def main():
    rows = []
    for ann_p in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ann_p)
        afp = config.ANNOTATION_JSON_PATH
        if not afp.exists():
            continue
        scen = config.DATASET_NAME
        flow = get_flow(scen)
        if not flow:
            continue
        ref_roots = {_root(t) for t in all_acceptable_tids(flow)}
        groups = json.load(open(afp, encoding="utf-8")).get("groups", [])
        if not groups:
            continue

        rule_fired_roots = {_root(g.get("rule_technique_id")) for g in groups}
        tp_covered = {_root(g.get("gt_technique_id")) for g in groups
                      if _is_true(g.get("gt_is_true_positive"))}
        n_tp = sum(1 for g in groups if _is_true(g.get("gt_is_true_positive")))

        rule_reach = len(ref_roots & rule_fired_roots) / len(ref_roots) if ref_roots else 0.0
        tp_reach = len(ref_roots & tp_covered) / len(ref_roots) if ref_roots else 0.0
        rows.append({"scen": scen, "n_ref": len(ref_roots), "n_groups": len(groups),
                     "n_tp": n_tp, "rule_reach": rule_reach, "tp_reach": tp_reach,
                     "fired": int(n_tp > 0)})

    n = len(rows)
    print(f"scenarios: {n}")
    print(f"scenario_firing (>=1 TP group)   : {mean(r['fired'] for r in rows):.4f}  ({sum(r['fired'] for r in rows)}/{n})")
    print(f"rule_reach   (anchor rule fires) : {mean(r['rule_reach'] for r in rows):.4f}  (macro)")
    print(f"tp_reach     (=completeness)     : {mean(r['tp_reach'] for r in rows):.4f}  (macro)")
    # micro over ref roots
    tot_ref = sum(r['n_ref'] for r in rows)
    print(f"rule_reach micro                 : {sum(r['rule_reach']*r['n_ref'] for r in rows)/tot_ref:.4f}")


if __name__ == "__main__":
    main()
