"""
Stage 1.6: Meta-noise pattern classifier (scenario-independent).



----
A. SYSTEM_REG_NOISE: anchor in SYSTEM_PROCS, rule==T1112, no cmd
B. SYSTEM_RPC_NOISE: anchor in SYSTEM_PROCS, rule in {T1021.002, T1021.003,
C. SYSTEM_CRED_PROBE: anchor=lsass.exe AND chain target=lsass with
   GrantedAccess in BENIGN list → benign (Windows internal probing)
D. NAN_RPC_NOISE: anchor=nan, rule in {T1021.002, T1021.003}, no cmd →
E. CONSUMER_APPS_NOISE: anchor in CONSUMER_PROCS (msedge, backgroundtaskhost

"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT))


SYSTEM_PROCS = {
    "svchost.exe", "lsass.exe", "services.exe", "system",
    "wininit.exe", "csrss.exe", "smss.exe", "winlogon.exe",
    "dfsrs.exe", "spoolsv.exe", "wmiprvse.exe", "dllhost.exe",
}

CONSUMER_PROCS = {
    "msedge.exe", "msedgewebview2.exe", "backgroundtaskhost.exe",
    "runtimebroker.exe", "taskhostw.exe", "sihost.exe",
    "searchindexer.exe", "searchprotocolhost.exe", "searchfilterhost.exe",
    "msmpeng.exe", "mssense.exe", "smartscreen.exe",
    "compattelrunner.exe", "diagtrack.exe", "usocoreworker.exe",
    "wermgr.exe", "werfault.exe",
    "sdxhelper.exe", "officeclicktorun.exe", "onedrive.exe",
    "ctfmon.exe", "explorer.exe", "shellexperiencehost.exe",
    "startmenuexperiencehost.exe", "yourphone.exe",
    "audiodg.exe", "audiog.exe", "conhost.exe", "fontdrvhost.exe",
    "wuauclt.exe", "trustedinstaller.exe",
    "applicationframehost.exe", "ngentask.exe", "ngen.exe",
    "lockapp.exe", "lsm.exe",
}

NOISY_RULES = {"T1021.002", "T1021.003", "T1021.006", "T1112", "T1003",
               "T1546.015", "T1055", "T1033", "T1070.001", "T1070.004",
               "T1134", "T1047", "T1562.001", "T1556"}

LSASS_BENIGN_ACCESS = {"0x1000", "0x400", "0x1400", "0x100000", "0x40"}


def _norm(v) -> str:
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan","none","") else s


def _basename(p: str) -> str:
    return p.replace("\\","/").split("/")[-1].lower() if p else ""


def is_noise(group: dict, features: dict) -> tuple[bool, str]:
    """Return (is_benign_noise, reason)."""
    rule = group.get("rule_technique_id","")
    a = group.get("anchor") or {}
    img = _basename(_norm(a.get("Image")))
    has_cmd = bool(features.get("command_script",{}).get("entries"))

    if has_cmd:
        return (False, "")

    if img in CONSUMER_PROCS:
        return (True, f"consumer-app-noise: {img}")

    if img in SYSTEM_PROCS and rule in NOISY_RULES:
        chains = features.get("execution_context",{}).get("process_chains") or []
        for c in chains:
            target = _basename(c.get("child_image","") or "")
            if "lsass" in target:
                ga = (c.get("granted_access") or "").lower()
                if ga and ga not in {x.lower() for x in LSASS_BENIGN_ACCESS}:
                    return (False, "")
        return (True, f"system-proc-noise: {img} + rule={rule}")

    if not img and rule in {"T1021.002","T1021.003"}:
        return (True, f"nan-anchor-rpc-noise: rule={rule}")

    return (False, "")


def main() -> None:
    files = sorted(OUTPUT_DIR.rglob("*_annotation.json"))
    totals = defaultdict(int)

    for ann in files:
        ftr = ann.with_name(ann.name.replace("_annotation.json","_feature_result.json"))
        if not ftr.exists(): continue
        with open(ann,"r",encoding="utf-8") as f: data = json.load(f)
        with open(ftr,"r",encoding="utf-8") as f: fd = json.load(f)
        fmap = {g["group_id"]: g.get("features",{}) for g in fd}

        for g in data.get("groups", []):
            if g.get("gt_is_true_positive") is not None:
                continue
            features = fmap.get(g["group_id"], {})
            ok, reason = is_noise(g, features)
            if ok:
                g["gt_is_true_positive"] = False
                g["gt_technique_id"] = None
                g["gt_tactic"] = None
                g["gt_label_source"] = "auto-meta-noise"
                g["gt_confidence"] = 0.85
                g["gt_notes"] = f"meta-noise: {reason}"
                totals["meta-noise"] += 1
            else:
                totals["pending"] += 1

        with open(ann,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print("Stage 1.6 ...:")
    for k,v in totals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
