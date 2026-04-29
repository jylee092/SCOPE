"""
Auto-label GT fields in annotation JSON files based on scenario→TTP mapping.


    cd Final_Code
    python experiments/auto_label_gt.py
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = ROOT / "output"
MITRE_CSV   = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
SCENARIO_MAP: dict[str, tuple[str, list[str]]] = {
    # ── atomic/collection ──
    "msf_record_mic": (
        "T1123",
        ["record_mic", "meterpreter", "msfvenom", "metsrv",
         "audiocapture", "waveinopen", "winmm", "audiosrv",
         "audiog.exe", "audiodg.exe", "payload.exe"],
    ),
    # ── atomic/credential_access ──
    "cmd_dumping_ntds_dit_file_ntdsutil": (
        "T1003.003",
        ["ntdsutil", "ntds.dit", "ifm ", "create full", "ac i ntds", "q q"],
    ),
    "cmd_sam_copy_esentutl": (
        "T1003.002",
        ["esentutl", "\\sam", "\\system", "vss", "shadowcopy", "/y /vss"],
    ),
    "empire_mimikatz_logonpasswords": (
        "T1003.001",
        ["mimikatz", "sekurlsa", "logonpasswords", "invoke-mimikatz", "empire", "powershell"],
    ),
    "empire_shell_rubeus_asktgt_createnetonly": (
        "T1558",
        ["rubeus", "asktgt", "createnetonly", "kerberos", "empire", "powershell"],
    ),
    "psh_lsass_memory_dump_comsvcs": (
        "T1003.001",
        ["comsvcs", "minidump", "rundll32", "lsass.dmp", "powershell"],
    ),
    # ── atomic/defense_evasion ──
    # T1055.013 (Process Doppelgänging) / T1036.005 (Match Legitimate Name)
    "cmd_process_herpaderping_mimiexplorer": (
        "T1055",
        ["herpaderping", "mimiexplorer", "wardog", "processherpaderping",
         "mimi", "doppelganging"],
    ),
    "cmd_stop_event_logging_controlset001_minint_key": (
        "T1562.002",
        ["minint", "controlset001", "eventlog", "reg add", "reg.exe", "services\\eventlog"],
    ),
    "cmd_wevtutil_modify_security_eventlog_path": (
        "T1562.002",
        ["wevtutil", "sl security", "eventlog", "security.evtx"],
    ),
    "empire_dllinjection_LoadLibrary_CreateRemoteThread": (
        "T1055.001",
        ["empire", "powershell", "loadlibrary", "createremotethread",
         "invoke-dllinjection", ".dll"],
    ),
    "psh_cmstp_execution_bypassuac": (
        "T1218.003",
        ["cmstp", "bypassuac", ".inf", "powershell", "cmstp.exe"],
    ),
    # ── atomic/discovery ──
    "cmd_seatbelt_group_user": (
        "T1082",
        ["seatbelt"],
    ),
    "empire_find_localadmin_smb_svcctl_OpenSCManager": (
        "T1069.001",
        ["find-localadmin", "svcctl", "openscmanager", "empire", "powershell",
         "get-localgroup", "find-localadminaccess"],
    ),
    "empire_getsession_dcerpc_smb_srvsvc_NetSessEnum": (
        "T1049",
        ["get-netsession", "netsessionenum", "srvsvc", "empire", "powershell",
         "netsess"],
    ),
    "empire_shell_net_local_users": (
        "T1087.001",
        ["net.exe", "net user", "net localgroup", "empire", "powershell",
         "net1.exe", " user", " localgroup"],
    ),
    "empire_shell_samr_EnumDomainUsers": (
        "T1087.002",
        ["samr", "enumdomainusers", "empire", "powershell", "net.exe",
         "get-netuser", "domainuser"],
    ),
    # ── atomic/execution ──
    "cmd_sharpview_pcre_net": (
        "T1087.002",
        ["sharpview", "get-objectacl", "domain admins", "get-domainuser",
         "get-objectacl", "powerview"],
    ),
    "empire_launcher_vbs": (
        "T1059.005",
        ["empire", "launcher.vbs", "wscript", "cscript", ".vbs", "powershell",
         "launcher", "encodedcommand"],
    ),
    "psh_powershell_httplistener": (
        "T1059.001",
        ["httplistener", "system.net.httplistener", "powershell"],
    ),
    "psh_python_webserver": (
        "T1059.006",
        ["python", "simplehttpserver", "http.server", "powershell",
         "system.net.webclient"],
    ),
    # ── atomic/lateral_movement ──
    "covenant_psremoting_command": (
        "T1021.006",
        ["covenant", "grunt", "psremoting", "winrm", "invoke-command", "wsman",
         "grunthttp"],
    ),
    "empire_psexec_dcerpc_tcp_svcctl": (
        "T1021.002",
        ["empire", "psexec", "svcctl", "dcerpc", "admin$", "powershell"],
    ),
    "empire_psremoting_stager": (
        "T1021.006",
        ["empire", "psremoting", "winrm", "wsman", "invoke-command", "powershell",
         "wsmprovhost"],
    ),
    "empire_wmi_dcerpc_wmi_IWbemServices_ExecMethod": (
        "T1047",
        ["empire", "wmi", "iwbemservices", "execmethod", "wmiprvse", "dcerpc",
         "powershell"],
    ),
    "purplesharp_ad_playbook_I": (
        "T1059.001",
        ["purplesharp", "empire", "powershell", "wsmprovhost", "psremoting"],
    ),
    # ── atomic/persistence ──
    "cmd_userinitmprlogonscript_batch": (
        "T1037.001",
        ["userinitmprlogonscript", "reg.exe", "reg add",
         "hkcu\\environment", ".bat"],
    ),
    "covenant_persistwmi": (
        "T1546.003",
        ["covenant", "grunt", "persistwmi", "wbem", "wmiprvse",
         "__eventfilter", "__eventconsumer", "root\\subscription",
         "scriptobj", "rundll32", "grunthttp"],
    ),
    "empire_persistence_registry_modification_run_keys_elevated_user": (
        "T1547.001",
        ["empire", "powershell", "currentversion\\run", "reg.exe", "reg add",
         "\\run\\", "runonce", "set-itemproperty"],
    ),
    "empire_schtasks_creation_execution_elevated_user": (
        "T1053.005",
        ["empire", "powershell", "schtasks", "taskmgr", "scheduledtask",
         "schedule.service", "schtasks.exe", "/create", "register-scheduledtask",
         "taskscheduler"],
    ),
    "empire_wmi_local_event_subscriptions_elevated_user": (
        "T1546.003",
        ["empire", "powershell", "wbem", "__eventfilter", "__eventconsumer",
         "wmiprvse", "activescripteventconsumer", "root\\subscription",
         "set-wmiinstance"],
    ),
    # ── atomic/privilege_escalation ──
    "cmd_service_mod_fax": (
        "T1543.003",
        ["sc.exe", "sc config", "fax", "sc start", "services.exe",
         "binpath", "services\\fax", "scmanager"],
    ),
    "empire_uac_shellapi_fodhelper": (
        "T1548.002",
        ["fodhelper", "ms-settings", "empire", "powershell", "shellapi",
         "uac"],
    ),
    # ── compound ──
    "metasploit_logonpasswords_lsass_memory_dump": (
        "T1003.001",
        ["mimikatz", "logonpasswords", "sekurlsa", "meterpreter",
         "metsrv", "metasploit", "msfvenom", "rundll32", "comsvcs",
         "powershell", "cscript", "payload.exe"],
    ),
    "metasploit_procdump_lsass_memory_dump": (
        "T1003.001",
        ["procdump", "lsass", "meterpreter", "metsrv", "metasploit",
         "msfvenom", "sysinternals", "payload.exe", "wermgr",
         "werfault"],
    ),
}


def load_tactic_map(csv_path: Path) -> dict[str, str]:
    """tid → tactic (first listed in MITRE CSV)."""
    tm: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = (row.get("ID") or "").strip()
            tac = (row.get("tactics") or "").split(",")[0].strip()
            if tid and tac:
                tm[tid] = tac
    return tm


def resolve_tactic(tid: str, tm: dict[str, str]) -> str:
    if tid in tm:
        return tm[tid]
    parent = tid.split(".")[0]
    return tm.get(parent, "")


def gather_context(group: dict) -> str:
    """anchor + sample_logs ...context ..."""
    parts: list[str] = []
    a = group.get("anchor") or {}
    for k in ("Image", "CommandLine", "ParentImage", "TargetObject"):
        v = a.get(k)
        if v and str(v).lower() != "nan":
            parts.append(str(v))
    for s in group.get("sample_logs", []) or []:
        for k in ("Image", "CommandLine", "ParentImage", "TargetObject"):
            v = s.get(k)
            if v and str(v).lower() != "nan":
                parts.append(str(v))
    return " ".join(parts).lower()


def scenario_key(scenario: str) -> str:
    """annotation scenario ...

        metasploit_procdump_lsass_memory_dump      → metasploit_procdump_lsass_memory_dump
    """
    m = re.match(r"^(.*?)(?:_(?:20\d{2}[-T]?\d{2}[-T]?\d{2}\S*|\d{4,}))?$", scenario)
    stem = m.group(1) if m else scenario
    stem = re.sub(r"[_-]+(?:\d+)?$", "", stem)
    return stem


def pick_mapping(scenario: str) -> tuple[str, list[str]] | None:
    """SCENARIO_MAP..."""
    if scenario in SCENARIO_MAP:
        return SCENARIO_MAP[scenario]
    key = scenario_key(scenario)
    if key in SCENARIO_MAP:
        return SCENARIO_MAP[key]
    # Fuzzy: prefix match
    for k, v in SCENARIO_MAP.items():
        if scenario.startswith(k) or k in scenario:
            return v
    return None


def label_group(
    group: dict,
    expected_tid: str,
    keywords: list[str],
    tactic_map: dict[str, str],
) -> dict:
    """Return updated group dict with gt_* fields set.

    -----------
      → TP
    """
    rule_tid = group.get("rule_technique_id", "")
    confidence = float(group.get("confidence", 0) or 0)
    expected_parent = expected_tid.split(".")[0]
    rule_family_match = (
        rule_tid == expected_tid
        or rule_tid == expected_parent
        or rule_tid.startswith(expected_parent + ".")
    )
    rule_exact = rule_tid == expected_tid

    a = group.get("anchor") or {}
    anchor_text = " ".join(
        str(a.get(k, "")) for k in ("Image", "CommandLine", "ParentImage", "TargetObject")
        if a.get(k) and str(a.get(k)).lower() != "nan"
    ).lower()
    anchor_kw = [kw for kw in keywords if kw.lower() in anchor_text]

    full_text = gather_context(group)
    sample_kw = [kw for kw in keywords if kw.lower() in full_text and kw not in anchor_kw]

    is_tp = False
    reason = ""
    if anchor_kw:
        is_tp = True
        reason = f"anchor-kw({','.join(anchor_kw[:3])})"
    elif sample_kw and (rule_family_match or confidence >= 0.5):
        is_tp = True
        reason = f"sample-kw+ctx({rule_tid},c={confidence:.2f})"
    elif rule_exact and confidence >= 0.5:
        is_tp = True
        reason = f"rule-exact(c={confidence:.2f})"
    elif rule_family_match and confidence >= 0.75:
        is_tp = True
        reason = f"rule-fam-strong({rule_tid},c={confidence:.2f})"
    else:
        bits = []
        if confidence == 0: bits.append("c=0")
        if not anchor_kw and not sample_kw: bits.append("no-kw")
        if not rule_family_match: bits.append(f"rule!=fam({rule_tid})")
        reason = "noise:" + ",".join(bits) if bits else "weak"

    gt_tid = expected_tid
    gt_tactic = resolve_tactic(gt_tid, tactic_map)

    updated = dict(group)
    updated["gt_is_true_positive"] = bool(is_tp)
    updated["gt_technique_id"] = gt_tid
    updated["gt_technique_name"] = group.get("rule_technique_name", "")
    updated["gt_tactic"] = gt_tactic
    updated["gt_notes"] = "auto: " + reason
    return updated


def process_all() -> None:
    tactic_map = load_tactic_map(MITRE_CSV)
    files = sorted(OUTPUT_DIR.rglob("*_annotation.json"))
    print(f"Found {len(files)} annotation files")

    stats = {"total": 0, "tp": 0, "fp": 0, "unmapped": 0}
    unmapped_scenarios: list[str] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenario = data.get("scenario", fp.parent.name)
        mapping = pick_mapping(scenario)
        if mapping is None:
            unmapped_scenarios.append(scenario)
            stats["unmapped"] += len(data.get("groups", []))
            continue

        expected_tid, keywords = mapping
        new_groups = []
        for g in data.get("groups", []):
            ng = label_group(g, expected_tid, keywords, tactic_map)
            new_groups.append(ng)
            stats["total"] += 1
            if ng["gt_is_true_positive"]:
                stats["tp"] += 1
            else:
                stats["fp"] += 1
        data["groups"] = new_groups

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"labeled: total={stats['total']} tp={stats['tp']} fp={stats['fp']}")
    if unmapped_scenarios:
        print(f"UNMAPPED ({stats['unmapped']} groups):")
        for s in unmapped_scenarios:
            print(f"  - {s}")


if __name__ == "__main__":
    process_all()
