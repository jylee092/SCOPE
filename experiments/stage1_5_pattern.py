"""
Stage 1.5: Feature-based pattern classifier.

feature_result.json 의 구조화된 feature(process_chains, command_script,
persistence.registry_signals 등)를 읽어 ATT&CK technique 으로 분류.

Stage 1 의 'pending' 그룹 중 패턴이 명확한 것을 사전 처리.
나머지만 Stage 2 (Claude LLM) 가 검토.

규칙 카테고리
------------
LSASS_ACCESS  — process_chain target=lsass + GrantedAccess high → T1003.001
LSASS_LOW     — process_chain target=lsass + GrantedAccess low (0x1000/0x400)
                 → benign (Windows 자체 housekeeping)
NTDS          — cmdline 에 ntdsutil + ifm/snapshot → T1003.003
SAM_COPY      — esentutl + (sam|/y/vss|shadowcopy) → T1003.002
COMSVCS       — rundll32 + comsvcs.dll + MiniDump → T1003.001
MIMIKATZ      — cmdline 에 mimikatz/sekurlsa/Invoke-Mimikatz → T1003.001
RUBEUS        — cmdline 에 Rubeus/asktgt → T1558
SCHTASKS_NEW  — schtasks + /create → T1053.005
WEVTUTIL_MOD  — wevtutil + sl/cl → T1562.002
REG_EVENTLOG  — reg add/delete + EventLog 경로 → T1562.002
REG_RUN       — registry signals \\Run\\ → T1547.001
WMI_SUB       — registry __EventFilter / __EventConsumer → T1546.003
NET_USER      — net.exe + user|localgroup|group → T1087.001/.002
SAMR          — cmdline 에 samr/EnumDomainUsers → T1087.002
WINRM_NET     — connections 5985/5986 → T1021.006
SMB_NET       — connections 445 outbound from svchost/services → T1021.002
DLLINJECT     — VirtualAllocEx/CreateRemoteThread/LoadLibrary 패턴 → T1055.001
SHARPVIEW     — sharpview.exe Get-ObjectAcl → T1087.002
SEATBELT      — seatbelt.exe → T1082
PYTHON_WS     — python.exe http.server/SimpleHTTPServer → T1059.006
CMSTP         — cmstp + .inf → T1218.003
FODHELPER     — fodhelper + ms-settings → T1548.002
PSEXEC        — services.exe parent + admin\\$\\ + remote → T1021.002
PSREMOTING    — wsmprovhost.exe child → T1021.006
NOISE         — empty features + conf=0 → benign

각 분류는 (technique_id, confidence, evidence) 반환.
confidence 가 0.9 이상이면 자동 라벨, 미만이면 pending 유지.
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MITRE_CSV  = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
LSASS_DUMP_ACCESS = {
    "0x1010", "0x1410", "0x1438", "0x143a", "0x1fffff",
    "0x143A", "0x1438", "0x101a", "0x101A",
    "0x107a", "0x107A",  # extra credential dump signatures
}
LSASS_BENIGN_ACCESS = {
    "0x1000", "0x400", "0x1400", "0x100000",
}


def load_tactic_map(csv_path: Path) -> dict[str, str]:
    tm: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = (row.get("ID") or "").strip()
            tac = (row.get("tactics") or "").split(",")[0].strip()
            if tid and tac:
                tm[tid] = tac
    return tm


def resolve_tactic(tid: str, tm: dict[str, str]) -> str:
    if tid in tm: return tm[tid]
    return tm.get(tid.split(".")[0], "")


def _norm(v) -> str:
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan","none","") else s


def _basename(p: str) -> str:
    return p.replace("\\","/").split("/")[-1].lower() if p else ""


# ---------------------------------------------------------------------------
# 패턴 분류 (각 함수: features → (tid, conf, evidence) | None)
# ---------------------------------------------------------------------------
def cls_lsass_access(features: dict) -> tuple[str, float, str] | None:
    chains = features.get("execution_context",{}).get("process_chains") or []
    lsass_high = 0; lsass_low = 0; lsass_other = 0
    sample_high = None
    for c in chains:
        target = (c.get("child_image") or "").lower()
        if "lsass" not in target: continue
        ga = (c.get("granted_access") or "").lower()
        if ga in {x.lower() for x in LSASS_DUMP_ACCESS}:
            lsass_high += 1
            if not sample_high: sample_high = (c.get("parent_image",""), ga)
        elif ga in {x.lower() for x in LSASS_BENIGN_ACCESS}:
            lsass_low += 1
        else:
            lsass_other += 1
    if lsass_high >= 1:
        return ("T1003.001", 0.92,
                f"lsass-high-access x{lsass_high} (e.g., {sample_high})")
    if lsass_low >= 1 and lsass_high == 0 and lsass_other == 0:
        # only low access → benign (Windows internal probe)
        return ("BENIGN", 0.85, f"lsass-low-access only x{lsass_low}")
    return None


def cls_cmdline(features: dict, scenario: str) -> tuple[str, float, str] | None:
    entries = features.get("command_script",{}).get("entries") or []
    if not entries: return None
    full = " ".join(
        f"{(e.get('image') or '').lower()} {(e.get('cmdline') or '').lower()}"
        for e in entries
    )
    # ── 우선순위 패턴 ──
    if "ntdsutil" in full and any(k in full for k in (" ifm ","snapshot","create full","ac i ntds"," ntds ")):
        return ("T1003.003", 0.95, "ntdsutil + ifm/snapshot")
    if "esentutl" in full and any(k in full for k in ("sam"," /y","vss","\\windows\\system32\\sam","shadowcopy")):
        return ("T1003.002", 0.93, "esentutl + sam/vss")
    if "rundll32" in full and "comsvcs" in full and ("minidump" in full or "lsass" in full):
        return ("T1003.001", 0.95, "rundll32 comsvcs MiniDump")
    if any(k in full for k in ("invoke-mimikatz","mimikatz.exe","sekurlsa","logonpasswords","mimikatz`")):
        return ("T1003.001", 0.93, "mimikatz pattern")
    if any(k in full for k in ("rubeus.exe","rubeus ","asktgt","kerberos::ask")):
        return ("T1558", 0.90, "rubeus/asktgt")
    if "schtasks" in full and "/create" in full:
        return ("T1053.005", 0.90, "schtasks /create")
    if "wevtutil" in full and any(k in full for k in (" sl ","sl security"," cl "," cl security")):
        return ("T1562.002", 0.90, "wevtutil sl/cl")
    if "reg" in full and any(k in full for k in ("eventlog","\\services\\eventlog","minint")) and any(k in full for k in (" add "," delete ")):
        return ("T1562.002", 0.88, "reg add/delete EventLog")
    if "reg" in full and any(k in full for k in ("\\run\\","\\runonce\\","currentversion\\run")) and " add " in full:
        return ("T1547.001", 0.88, "reg add Run key")
    if "set-itemproperty" in full and "\\run" in full:
        return ("T1547.001", 0.88, "Set-ItemProperty Run")
    if "net" in full and any(k in full for k in (" user "," user/"," localgroup "," group ")) and any(k in full for k in ("/domain","domain admins")):
        return ("T1087.002", 0.88, "net user/group /domain")
    if "net" in full and any(k in full for k in (" user "," localgroup "," group ")):
        return ("T1087.001", 0.85, "net user/group local")
    if "samr" in full and "enumdomainusers" in full:
        return ("T1087.002", 0.93, "SAMR EnumDomainUsers")
    if "sharpview" in full and any(k in full for k in ("get-objectacl","get-domainuser","domain admins")):
        return ("T1087.002", 0.90, "SharpView Get-ObjectAcl")
    if "seatbelt" in full:
        return ("T1082", 0.85, "Seatbelt enumeration")
    if "python" in full and any(k in full for k in ("simplehttpserver","http.server","-m http")):
        return ("T1059.006", 0.92, "python http.server")
    if "cmstp" in full and ".inf" in full:
        return ("T1218.003", 0.92, "cmstp + .inf")
    if "fodhelper" in full or "ms-settings" in full:
        return ("T1548.002", 0.90, "fodhelper / ms-settings UAC bypass")
    if "wmiprvse" in full and any(k in full for k in ("__eventfilter","__eventconsumer","activescript","scriptobj","root\\subscription")):
        return ("T1546.003", 0.92, "WMI subscription registration")
    if "set-wmiinstance" in full and any(k in full for k in ("__eventfilter","__eventconsumer","activescript")):
        return ("T1546.003", 0.92, "Set-WmiInstance subscription")
    if any(k in full for k in ("loadlibrary","createremotethread","virtualallocex")) and "powershell" in full:
        return ("T1055.001", 0.85, "PowerShell DLL injection pattern")
    if "powershell" in full and any(k in full for k in ("invoke-command","-computername","new-pssession","enter-pssession")):
        return ("T1021.006", 0.85, "PowerShell remoting")
    if "powershell" in full and ("encodedcommand" in full or "-enc " in full or "-e " in full):
        return ("T1059.001", 0.80, "PowerShell EncodedCommand")
    if any(k in full for k in ("powershell","cmd /c","cmd.exe /c")):
        return ("T1059.001", 0.55, "generic powershell/cmd")
    return None


def cls_persistence(features: dict) -> tuple[str, float, str] | None:
    p = features.get("persistence",{})
    sigs = p.get("registry_signals") or []
    noise = p.get("registry_noise") or []
    full = " ".join(str(s).lower() for s in (sigs + noise))
    if any(k in full for k in ("__eventfilter","__eventconsumer","root\\subscription","activescripteventconsumer")):
        return ("T1546.003", 0.93, "WMI subscription registry")
    if "currentversion\\run" in full and any(k in full for k in ("set","add","wrote")):
        return ("T1547.001", 0.88, "Run key write")
    if any(k in full for k in ("\\services\\eventlog","minint","wevtutil")):
        return ("T1562.002", 0.85, "EventLog service registry mod")
    if any(k in full for k in ("comhijack","clsid","inprocserver32")) and any(k in full for k in ("write","set","add")):
        return ("T1546.015", 0.80, "COM hijacking registry")
    return None


def cls_network(features: dict) -> tuple[str, float, str] | None:
    conns = features.get("network",{}).get("connections") or []
    if not conns: return None
    ports = []
    for c in conns:
        dp = str(c.get("destination_port") or c.get("dport") or "").strip()
        if dp.isdigit(): ports.append(int(dp))
    if 5985 in ports or 5986 in ports:
        return ("T1021.006", 0.85, "WinRM port 5985/5986")
    if 3389 in ports:
        return ("T1021.001", 0.85, "RDP port 3389")
    if 445 in ports:
        return ("T1021.002", 0.75, "SMB port 445")
    return None


def cls_evasion(features: dict) -> tuple[str, float, str] | None:
    e = features.get("evasion",{})
    if e.get("log_cleared"):
        return ("T1070.001", 0.92, "Windows Event Log cleared")
    if e.get("deleted_files"):
        # often also part of attack but not deterministic alone
        return None
    return None


def classify(features: dict, scenario: str) -> tuple[str, float, str] | None:
    """Run all classifiers, return highest-confidence non-None result."""
    candidates = []
    for fn in (cls_lsass_access, cls_persistence, cls_evasion, cls_network):
        r = fn(features)
        if r is not None:
            candidates.append(r)
    r = cls_cmdline(features, scenario)
    if r is not None:
        candidates.append(r)
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    tactic_map = load_tactic_map(MITRE_CSV)
    files = sorted(OUTPUT_DIR.rglob("*_annotation.json"))

    totals = defaultdict(int)
    per_scen = {}

    CONF_THRESHOLD = 0.85   # 이 이상이면 자동 라벨, 미만이면 pending 유지

    for ann in files:
        ftr = ann.with_name(ann.name.replace("_annotation.json","_feature_result.json"))
        if not ftr.exists():
            continue
        with open(ann, "r", encoding="utf-8") as f: data = json.load(f)
        with open(ftr, "r", encoding="utf-8") as f: fd = json.load(f)
        scenario = data.get("scenario", ann.parent.name)
        fmap = {g["group_id"]: g.get("features",{}) for g in fd}

        cnt = defaultdict(int)
        for g in data.get("groups", []):
            if g.get("gt_label_source") != "pending": continue
            features = fmap.get(g["group_id"], {})
            res = classify(features, scenario)
            if res is None:
                cnt["still-pending"] += 1
                continue
            tid, conf, ev = res
            if conf < CONF_THRESHOLD:
                cnt["low-conf"] += 1
                continue
            if tid == "BENIGN":
                g["gt_is_true_positive"] = False
                g["gt_technique_id"] = None
                g["gt_tactic"] = None
                g["gt_label_source"] = "auto-pattern-benign"
                g["gt_confidence"] = conf
                g["gt_notes"] = f"pattern-benign: {ev}"
                cnt["benign"] += 1
            else:
                g["gt_is_true_positive"] = True
                g["gt_technique_id"] = tid
                g["gt_tactic"] = resolve_tactic(tid, tactic_map)
                g["gt_label_source"] = "auto-pattern"
                g["gt_confidence"] = conf
                g["gt_notes"] = f"pattern: {ev}"
                cnt["pattern-tp"] += 1

        with open(ann, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        per_scen[scenario] = dict(cnt)
        for k,v in cnt.items(): totals[k] += v

    print("Stage 1.5 결과:")
    for k,v in sorted(totals.items()):
        print(f"  {k}: {v}")
    print("\nper-scenario:")
    print(f"{'scenario':<65} {'pat-tp':>7} {'benign':>7} {'low-conf':>9} {'pending':>9}")
    print("-"*100)
    for s, c in sorted(per_scen.items()):
        print(f"{s[:65]:<65} {c.get('pattern-tp',0):>7} {c.get('benign',0):>7} "
              f"{c.get('low-conf',0):>9} {c.get('still-pending',0):>9}")


if __name__ == "__main__":
    main()
