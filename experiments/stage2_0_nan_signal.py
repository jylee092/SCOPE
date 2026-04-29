"""
Stage 2.0: nan-anchor 그룹의 cmd entries 에 attack signal 이 있을 때 자동 라벨.

규칙:
1. cls_cmdline 패턴 매치 → TP with pattern's tid (Stage 1.5 재활용)
2. attacker keyword 매치 + rule_tid 가 plausible → TP with rule_tid
3. 그 외 → benign
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MITRE_CSV  = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"
sys.path.insert(0, str(ROOT))

from experiments.stage1_5_pattern import cls_cmdline, cls_persistence, cls_lsass_access
from experiments.stage1_7_aggressive import (
    has_attacker_kw, _basename, _norm, ATTACKER_KEYWORDS, EXTENDED_NOISE_PROCS,
)


def _load_tactic_map():
    tm = {}
    with open(MITRE_CSV,"r",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = (row.get("ID") or "").strip()
            tac = (row.get("tactics") or "").split(",")[0].strip()
            if tid and tac: tm[tid] = tac
    return tm

def _resolve_tactic(tid, tm):
    if tid in tm: return tm[tid]
    return tm.get(tid.split(".")[0], "")

PLAUSIBLE_RULE_TIDS = {
    # rule_tid → (default_tid, default_tactic_via_csv)
    "T1003": ("T1003.001",   None),
    "T1003.001": ("T1003.001", None),
    "T1003.002": ("T1003.002", None),
    "T1003.003": ("T1003.003", None),
    "T1059":     ("T1059.003", None),
    "T1059.001": ("T1059.001", None),
    "T1059.003": ("T1059.003", None),
    "T1059.005": ("T1059.005", None),
    "T1053.005": ("T1053.005", None),
    "T1087":     ("T1087",     None),
    "T1087.001": ("T1087.001", None),
    "T1087.002": ("T1087.002", None),
    "T1069":     ("T1069",     None),
    "T1069.001": ("T1069.001", None),
    "T1069.002": ("T1069.002", None),
    "T1033":     ("T1033",     None),
    "T1018":     ("T1018",     None),
    "T1049":     ("T1049",     None),
    "T1016":     ("T1016",     None),
    "T1546.003": ("T1546.003", None),
    "T1547.001": ("T1547.001", None),
    "T1037.001": ("T1037.001", None),
    "T1036.005": ("T1036.005", None),
    "T1218.003": ("T1218.003", None),
    "T1543.003": ("T1543.003", None),
    "T1548.002": ("T1548.002", None),
    "T1562.002": ("T1562.002", None),
    "T1037":     ("T1037.001", None),
    "T1556":     ("T1556",     None),
    "T1558":     ("T1558",     None),
    "T1110":     ("T1110",     None),
}


def main():
    tm = _load_tactic_map()
    cnt = defaultdict(int)
    for ann in OUTPUT_DIR.rglob("*_annotation.json"):
        ftr = ann.with_name(ann.name.replace("_annotation.json","_feature_result.json"))
        if not ftr.exists(): continue
        with open(ann,"r",encoding="utf-8") as f: data = json.load(f)
        with open(ftr,"r",encoding="utf-8") as f: fd = json.load(f)
        fmap = {g["group_id"]: g.get("features",{}) for g in fd}
        scenario = data.get("scenario","")
        for g in data.get("groups", []):
            if g.get("gt_is_true_positive") is not None: continue
            features = fmap.get(g["group_id"], {})
            rule = g.get("rule_technique_id","")

            # 1. pattern classifier
            pattern_match = None
            for fn in (cls_cmdline, cls_persistence, cls_lsass_access):
                r = fn(features, scenario) if fn is cls_cmdline else fn(features)
                if r is not None and r[1] >= 0.85 and r[0] != "BENIGN":
                    pattern_match = r
                    break

            if pattern_match:
                tid, conf, ev = pattern_match
                g["gt_is_true_positive"] = True
                g["gt_technique_id"] = tid
                g["gt_tactic"] = _resolve_tactic(tid, tm)
                g["gt_label_source"] = "auto-stage2-pattern"
                g["gt_confidence"] = conf
                g["gt_notes"] = f"stage2-pattern: {ev}"
                cnt[f"tp:{tid}"] += 1
                continue

            # 2. attacker kw + plausible rule
            entries = features.get("command_script",{}).get("entries") or []
            full_cmd = " ".join(
                f"{(e.get('image') or '')} {(e.get('cmdline') or '')}" for e in entries
            )
            if has_attacker_kw(full_cmd) and rule in PLAUSIBLE_RULE_TIDS:
                default_tid, _ = PLAUSIBLE_RULE_TIDS[rule]
                g["gt_is_true_positive"] = True
                g["gt_technique_id"] = default_tid
                g["gt_tactic"] = _resolve_tactic(default_tid, tm)
                g["gt_label_source"] = "auto-stage2-rule-fallback"
                g["gt_confidence"] = 0.70
                g["gt_notes"] = f"stage2: attacker-kw + rule={rule} → {default_tid}"
                cnt[f"tp:{default_tid}"] += 1
                continue

            # 3. no clear attack signal → benign
            g["gt_is_true_positive"] = False
            g["gt_label_source"] = "auto-stage2-benign"
            g["gt_confidence"] = 0.70
            g["gt_notes"] = f"stage2-benign: no attack signal in cmd/chains, rule={rule}"
            cnt["benign"] += 1

        with open(ann,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    for k,v in sorted(cnt.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
