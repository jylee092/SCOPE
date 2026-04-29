"""

"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

sys.path.insert(0, str(ROOT))
from experiments.stage1_7_aggressive import (
    EXTENDED_NOISE_PROCS, ATTACKER_KEYWORDS, has_attacker_kw,
    _basename, _norm,
)

SYSTEM_CMD_IMAGES = EXTENDED_NOISE_PROCS | {
    "collectguestlogs.exe", "windowsazureguestagent.exe", "waappagent.exe",
    "mpcmdrun.exe", "msmpeng.exe", "mssense.exe",
    "officec2rclient.exe", "officeclicktorun.exe",
    "provtool.exe", "dsregcmd.exe",
    "smss.exe", "autochk.exe", "winit.exe", "wininit.exe",
    "identity_helper.exe", "msedgewebview2.exe",
    "checkstatus.bat", "npcap",  # Wireshark/Npcap installer parts
    "tiworker.exe", "tihost.exe", "trustedinstaller.exe",
    "rundll32.exe",  # rundll32 itself is dual-use; if no attacker kw, treat as benign
    "cleanmgr.exe", "perfmon.exe", "logman.exe",
}


def _is_system_cmd(image: str, cmdline: str) -> bool:
    img = _basename(image)
    if img in SYSTEM_CMD_IMAGES:
        return True
    # path-based heuristic
    p = (image or "").lower().replace("\\","/")
    return any(t in p for t in (
        "/program files/common files/microsoft shared/clicktorun/",
        "/programdata/microsoft/windows defender/",
        "/windowsazure/",
        "/programdata/azureconnectedmachineagent/",
        "/program files (x86)/microsoft/edge/",
        "/program files/windowsapps/",
        "/program files/microsoft sql server/",
    ))


def is_nan_benign(group: dict, features: dict) -> tuple[bool, str]:
    a = group.get("anchor") or {}
    img = _basename(_norm(a.get("Image")))
    if img and img != "nan":
        return (False, "")  # not nan-anchor

    # check cmd entries
    entries = features.get("command_script",{}).get("entries") or []
    if entries:
        all_cmds = " ".join(
            f"{e.get('image','')} {e.get('cmdline','')}" for e in entries
        )
        if has_attacker_kw(all_cmds):
            return (False, "")  # pending -- has attacker keyword
        non_system = [e for e in entries if not _is_system_cmd(e.get("image",""), e.get("cmdline",""))]
        if not non_system:
            return (True, f"all-system-cmd ({len(entries)} entries)")

    # no cmd entries -- check process_chains for attacker patterns
    chains = features.get("execution_context",{}).get("process_chains") or []
    if chains:
        chain_text = " ".join(
            f"{_basename(c.get('parent_image',''))} {_basename(c.get('child_image',''))}"
            for c in chains
        )
        if has_attacker_kw(chain_text):
            return (False, "")
        # all chains involve only system processes
        all_system = all(
            _basename(c.get("parent_image","")) in SYSTEM_CMD_IMAGES and
            _basename(c.get("child_image",""))  in SYSTEM_CMD_IMAGES
            for c in chains if c.get("parent_image") or c.get("child_image")
        )
        if all_system:
            return (True, f"chains: all system procs ({len(chains)})")

    # no features at all → benign
    if not entries and not chains:
        # check registry signals
        sigs = features.get("persistence",{}).get("registry_signals") or []
        if not sigs:
            return (True, "no features")

    return (False, "")


def main():
    cnt = defaultdict(int)
    for ann in OUTPUT_DIR.rglob("*_annotation.json"):
        ftr = ann.with_name(ann.name.replace("_annotation.json","_feature_result.json"))
        if not ftr.exists(): continue
        with open(ann,"r",encoding="utf-8") as f: data = json.load(f)
        with open(ftr,"r",encoding="utf-8") as f: fd = json.load(f)
        fmap = {g["group_id"]: g.get("features",{}) for g in fd}
        for g in data.get("groups", []):
            if g.get("gt_is_true_positive") is not None: continue
            ok, reason = is_nan_benign(g, fmap.get(g["group_id"], {}))
            if ok:
                g["gt_is_true_positive"] = False
                g["gt_label_source"] = "auto-nan-system"
                g["gt_confidence"] = 0.85
                g["gt_notes"] = f"nan-system: {reason}"
                cnt["benign"] += 1
            else:
                cnt["pending"] += 1
        with open(ann,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    for k,v in cnt.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
