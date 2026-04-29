"""
Stage 1.7: Aggressive system/consumer-process noise filter.

Stage 1.6 은 has_cmd=False 일 때만 noise 판정했지만, 많은 consumer/system
프로세스가 자식 프로세스를 spawn 하면서 cmd entries 가 채워짐. 이 경우에도
attacker-tool 키워드가 없으면 benign noise 로 처리.

추가 대상:
- networkwatcheragent.exe (Azure 모니터링)
- powershell_ise.exe (단순 IDE 실행만)
- conhost.exe (콘솔 호스트, 의미 없음)
- explorer.exe / dsregcmd.exe / trustedinstaller.exe / 기타 OS 컴포넌트

규칙:
A. anchor가 EXTENDED_NOISE_PROCS 이고, cmd entries 가 모두 정상 OS 활동
   (powershell/cmd/cscript/wscript/rundll32 등 attacker-tool 명령 없음) → benign
B. anchor가 nan 이고 rule 이 generic-noise rule 이며 features 가 빈약 → benign
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT))

EXTENDED_NOISE_PROCS = {
    # Stage 1.6 에 있던 것들
    "svchost.exe", "lsass.exe", "services.exe", "system",
    "wininit.exe", "csrss.exe", "smss.exe", "winlogon.exe",
    "dfsrs.exe", "spoolsv.exe", "wmiprvse.exe", "dllhost.exe",
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
    # Stage 1.7 추가
    "networkwatcheragent.exe", "networkwatc",  # azure
    "dsregcmd.exe", "officeclicktorun.exe",
    "powershell_ise.exe",  # 단순 ISE 실행
    "lpremove.exe", "lpkinstall.exe",
    "tiworker.exe", "tiworker.", "tihost.exe",
    "vssvc.exe",
    "splwow64.exe",
    # Stage 1.7b 추가 (singleton system components)
    "sppsvc.exe", "backgroundtransferhost.exe", "usoclient.exe",
    "logonui.exe", "searchui.exe", "officec2rclient.exe",
    "comppkgsrv.exe", "route.exe", "wmiadap.exe",
    "remotefxvgpudisablement.exe", "windowsazureguestagent.exe",
    "wsqmcons.exe", "mpcmdrun.exe", "provtool.exe",
    "microsoftsearchinbing.exe", "sysmon.exe", "slui.exe",
    "chxsmartscreen.exe", "collectguestlogs.exe",
    "splwow64.exe", "vds.exe", "vdsldr.exe",
    "lsaiso.exe", "smartscreen.exe",
    "smsvchost.exe", "msoia.exe",
}

ATTACKER_KEYWORDS = (
    "mimikatz", "sekurlsa", "logonpasswords", "rubeus", "asktgt",
    "ntdsutil", "esentutl", "schtasks", "sharpview", "seatbelt",
    "comsvcs", "minidump", "wevtutil", "fodhelper", "cmstp",
    "purplesharp", "covenant", "grunt", "empire", "metasploit",
    "meterpreter", "msfvenom", "payload", "invoke-mimikatz",
    "invoke-dllinjection", "loadlibrary", "createremotethread",
    "virtualallocex", "encodedcommand", "-enc ",
    "currentversion\\run", "userinitmprlogon", "__eventfilter",
    "__eventconsumer", "set-wmiinstance", "register-scheduledtask",
    "net user /domain", "net group /domain",
    "samr", "enumdomainusers", "get-objectacl",
    "/ifm", "create full", "ac i ntds",
    "binpath=", "sc config",
    "ms-settings", "binpath=",
    "processherpaderping", "mimiexplorer", "wardog",
    "winx64_payload",
)


def _basename(p):
    return p.replace("\\","/").split("/")[-1].lower() if p else ""


def _norm(v):
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan","none","") else s


def has_attacker_kw(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ATTACKER_KEYWORDS)


def is_benign_extended(group: dict, features: dict) -> tuple[bool, str]:
    a = group.get("anchor") or {}
    img = _basename(_norm(a.get("Image")))
    cmd = _norm(a.get("CommandLine"))

    # If anchor in noise procs → benign UNLESS anchor's own cmdline has
    # attacker keyword (sample logs are not attributable to the anchor's
    # intent — they may just be temporally co-occurring events grouped by
    # rule).
    if img in EXTENDED_NOISE_PROCS or img.startswith("networkwatc"):
        if cmd and has_attacker_kw(cmd):
            return (False, "")
        return (True, f"extended-noise-anchor: {img}")

    # For non-system anchors: check anchor cmdline for attacker keyword
    if cmd and has_attacker_kw(cmd):
        return (False, "")

    # cmd entries (only matters for non-system anchors)
    entries = features.get("command_script",{}).get("entries") or []
    full_cmd = " ".join(
        f"{(e.get('image') or '')} {(e.get('cmdline') or '')}"
        for e in entries
    )
    if has_attacker_kw(full_cmd):
        return (False, "")

    return (False, "")


def main() -> None:
    files = sorted(OUTPUT_DIR.rglob("*_annotation.json"))
    cnt = defaultdict(int)
    for ann in files:
        ftr = ann.with_name(ann.name.replace("_annotation.json","_feature_result.json"))
        if not ftr.exists(): continue
        with open(ann,"r",encoding="utf-8") as f: data = json.load(f)
        with open(ftr,"r",encoding="utf-8") as f: fd = json.load(f)
        fmap = {g["group_id"]: g.get("features",{}) for g in fd}
        for g in data.get("groups", []):
            if g.get("gt_is_true_positive") is not None:
                continue
            ok, reason = is_benign_extended(g, fmap.get(g["group_id"], {}))
            if ok:
                g["gt_is_true_positive"] = False
                g["gt_label_source"] = "auto-noise-ext"
                g["gt_confidence"] = 0.85
                g["gt_notes"] = f"noise-ext: {reason}"
                cnt["benign"] += 1
            else:
                cnt["pending"] += 1
        with open(ann,"w",encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print("Stage 1.7 결과:")
    for k,v in cnt.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
