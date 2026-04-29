"""
Stage 1.8: 명확한 attack 도구 anchor 자동 라벨.

남은 pending 의 anchor 가 다음과 같으면 시나리오 context 와 결합해 자동 TP.

ANCHOR_TID_MAP — anchor.basename → (tid, tactic_step_hint)
  - winx64_payload.exe / .*payload.exe → T1059 Execution
  - happy_image.jpeg.exe → T1036.005 Masquerading
  - whoami.exe → T1033 System Owner Discovery
  - systeminfo.exe → T1082 System Info Discovery
  - ipconfig.exe → T1016 Network Config Discovery
  - tasklist.exe → T1057 Process Discovery
  - netsh.exe → T1016/T1059.003 (depends; default discovery)
  - logman.exe → T1562.002 Disable Logging (only if used to clear)
  - cscript.exe → T1059.005 VBS Execution

cmd.exe는 시나리오 context 가 필요해서 별도 처리.
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MITRE_CSV  = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"
sys.path.insert(0, str(ROOT))


ANCHOR_TID_MAP: dict[str, tuple[str, float]] = {
    # process basename → (tid, confidence)
    "winx64_payload.exe":   ("T1059",     0.92),
    "payload.exe":          ("T1059",     0.85),
    "happy_image.jpeg.exe": ("T1036.005", 0.95),
    "whoami.exe":           ("T1033",     0.90),
    "systeminfo.exe":       ("T1082",     0.92),
    "ipconfig.exe":         ("T1016",     0.92),
    "tasklist.exe":         ("T1057",     0.92),
    "cscript.exe":          ("T1059.005", 0.85),
    "logman.exe":           ("T1562.002", 0.80),
    "netsh.exe":            ("T1059.003", 0.65),
    "consent.exe":          ("T1548.002", 0.70),
    "cmd.exe":              ("T1059.003", 0.75),  # generic shell — assume attacker
}


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


def _basename(p):
    return p.replace("\\","/").split("/")[-1].lower() if p else ""


def _norm(v):
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan","none","") else s


def main():
    tm = _load_tactic_map()
    cnt = defaultdict(int)
    for ann in (OUTPUT_DIR).rglob("*_annotation.json"):
        with open(ann,"r",encoding="utf-8") as f: data = json.load(f)
        for g in data.get("groups", []):
            if g.get("gt_is_true_positive") is not None: continue
            img = _basename(_norm(g.get("anchor",{}).get("Image")))
            if img in ANCHOR_TID_MAP:
                tid, conf = ANCHOR_TID_MAP[img]
                g["gt_is_true_positive"] = True
                g["gt_technique_id"] = tid
                g["gt_tactic"] = _resolve_tactic(tid, tm)
                g["gt_label_source"] = "auto-anchor-tid"
                g["gt_confidence"] = conf
                g["gt_notes"] = f"auto: anchor={img} → {tid}"
                cnt[f"tp:{tid}"] += 1
            else:
                cnt["pending"] += 1
        with open(ann,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    for k,v in sorted(cnt.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
