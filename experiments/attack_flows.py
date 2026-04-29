"""
Reference attack flow per scenario.



Mordor docs: https://github.com/OTRF/Security-Datasets
"""
from __future__ import annotations

# step = {"tid": str, "tactic": str, "alts": [str], "note": str}

ATTACK_FLOWS: dict[str, list[dict]] = {

    # ── atomic/collection ──────────────────────────────────────────────
    "msf_record_mic": [
        {"tid": "T1059", "tactic": "Execution", "alts": ["T1059.001","T1059.005"],
         "note": "Meterpreter payload exec"},
        {"tid": "T1055", "tactic": "Defense Evasion", "alts": ["T1055.001"],
         "note": "Meterpreter migrate / inject"},
        {"tid": "T1123", "tactic": "Collection", "alts": [],
         "note": "Audio capture via post/multi/manage/record_mic"},
    ],

    # ── atomic/credential_access ───────────────────────────────────────
    "cmd_dumping_ntds_dit_file_ntdsutil": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": ["T1059"],
         "note": "cmd.exe runs ntdsutil"},
        {"tid": "T1003.003", "tactic": "Credential Access", "alts": ["T1003"],
         "note": "ntdsutil ifm/snapshot dumps NTDS.dit"},
    ],
    "cmd_sam_copy_esentutl": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd.exe runs esentutl"},
        {"tid": "T1003.002", "tactic": "Credential Access", "alts": ["T1003"],
         "note": "esentutl /y /vss copies SAM/SYSTEM hive"},
    ],
    "empire_mimikatz_logonpasswords": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell launcher"},
        {"tid": "T1003.001", "tactic": "Credential Access", "alts": ["T1003"],
         "note": "Invoke-Mimikatz logonpasswords (LSASS)"},
    ],
    "empire_shell_rubeus_asktgt_createnetonly": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire shell launches Rubeus"},
        {"tid": "T1558", "tactic": "Credential Access",
         "alts": ["T1558.003","T1550.003"],
         "note": "Rubeus asktgt + createnetonly Kerberos manipulation"},
    ],
    "psh_lsass_memory_dump_comsvcs": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "PowerShell launcher"},
        {"tid": "T1003.001", "tactic": "Credential Access",
         "alts": ["T1003","T1218.011"],
         "note": "rundll32 comsvcs.dll MiniDump on lsass"},
    ],

    # ── atomic/defense_evasion ─────────────────────────────────────────
    "cmd_process_herpaderping_mimiexplorer": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd.exe runs ProcessHerpaderping"},
        {"tid": "T1036.005", "tactic": "Defense Evasion",
         "alts": ["T1055","T1055.013","T1036"],
         "note": "Process image swap on disk to masquerade"},
    ],
    "cmd_stop_event_logging_controlset001_minint_key": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd.exe runs reg add"},
        {"tid": "T1562.002", "tactic": "Defense Evasion",
         "alts": ["T1562","T1112"],
         "note": "Registry redirect EventLog to MININT key → disable logging"},
    ],
    "cmd_wevtutil_modify_security_eventlog_path": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd.exe runs wevtutil"},
        {"tid": "T1562.002", "tactic": "Defense Evasion",
         "alts": ["T1070.001","T1562"],
         "note": "wevtutil sl security /lfn changes log file path"},
    ],
    "empire_dllinjection_LoadLibrary_CreateRemoteThread": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell launcher"},
        {"tid": "T1055.001", "tactic": "Defense Evasion",
         "alts": ["T1055"],
         "note": "Invoke-DllInjection: VirtualAllocEx + LoadLibrary + CreateRemoteThread"},
    ],
    "psh_cmstp_execution_bypassuac": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "PowerShell drops .inf"},
        {"tid": "T1218.003", "tactic": "Defense Evasion",
         "alts": ["T1548.002"],
         "note": "cmstp.exe loads .inf → UAC bypass"},
    ],

    # ── atomic/discovery ───────────────────────────────────────────────
    "cmd_seatbelt_group_user": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd runs Seatbelt"},
        {"tid": "T1082", "tactic": "Discovery",
         "alts": ["T1087","T1057","T1518"],
         "note": "Seatbelt host enumeration (broad sysinfo)"},
    ],
    "empire_find_localadmin_smb_svcctl_OpenSCManager": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1069.001", "tactic": "Discovery",
         "alts": ["T1018"],
         "note": "Find-LocalAdminAccess via SVCCTL OpenSCManager"},
    ],
    "empire_getsession_dcerpc_smb_srvsvc_NetSessEnum": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1049", "tactic": "Discovery",
         "alts": [],
         "note": "Get-NetSession via SRVSVC NetSessionEnum"},
    ],
    "empire_shell_net_local_users": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1087.001", "tactic": "Discovery",
         "alts": ["T1087"],
         "note": "net.exe user/localgroup local enum"},
    ],
    "empire_shell_samr_EnumDomainUsers": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1087.002", "tactic": "Discovery",
         "alts": ["T1087"],
         "note": "SAMR EnumDomainUsers → domain accounts"},
    ],

    # ── atomic/execution ───────────────────────────────────────────────
    "cmd_sharpview_pcre_net": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd.exe runs SharpView"},
        {"tid": "T1087.002", "tactic": "Discovery",
         "alts": ["T1069.002","T1087"],
         "note": "SharpView Get-ObjectAcl Domain Admins"},
    ],
    "empire_launcher_vbs": [
        {"tid": "T1059.005", "tactic": "Execution", "alts": ["T1059"],
         "note": "wscript/cscript runs launcher.vbs"},
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "VBS spawns PowerShell stager"},
    ],
    "psh_powershell_httplistener": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "PowerShell HttpListener"},
        {"tid": "T1071.001", "tactic": "Command and Control", "alts": [],
         "note": "Listener provides HTTP C2 channel"},
    ],
    "psh_python_webserver": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "PowerShell launches python"},
        {"tid": "T1059.006", "tactic": "Execution", "alts": [],
         "note": "python http.server (file delivery / staging)"},
    ],

    # ── atomic/lateral_movement ────────────────────────────────────────
    "covenant_psremoting_command": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Covenant Grunt PowerShell"},
        {"tid": "T1021.006", "tactic": "Lateral Movement",
         "alts": ["T1021"],
         "note": "Invoke-Command via WSMan/WinRM"},
    ],
    "empire_psexec_dcerpc_tcp_svcctl": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1021.002", "tactic": "Lateral Movement",
         "alts": ["T1021"],
         "note": "PsExec via DCERPC SVCCTL on admin$ share"},
    ],
    "empire_psremoting_stager": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1021.006", "tactic": "Lateral Movement",
         "alts": ["T1021"],
         "note": "Empire PSRemoting stager via WSMan"},
    ],
    "empire_wmi_dcerpc_wmi_IWbemServices_ExecMethod": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell"},
        {"tid": "T1047", "tactic": "Execution",
         "alts": ["T1021.003"],
         "note": "WMI IWbemServices ExecMethod (lateral via DCERPC)"},
    ],
    "purplesharp_ad_playbook_I": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": ["T1059.001"],
         "note": "PurpleSharp playbook launch"},
        {"tid": "T1087.002", "tactic": "Discovery", "alts": ["T1087"],
         "note": "AD account discovery"},
        {"tid": "T1021.006", "tactic": "Lateral Movement", "alts": [],
         "note": "WinRM remoting"},
        {"tid": "T1003.001", "tactic": "Credential Access",
         "alts": ["T1003"],
         "note": "Credential dump (mimikatz)"},
    ],

    # ── atomic/persistence ─────────────────────────────────────────────
    "cmd_userinitmprlogonscript_batch": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd runs reg add"},
        {"tid": "T1037.001", "tactic": "Persistence",
         "alts": ["T1547.001"],
         "note": "HKCU\\Environment UserInitMprLogonScript = .bat"},
    ],
    "covenant_persistwmi": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Covenant Grunt"},
        {"tid": "T1546.003", "tactic": "Persistence",
         "alts": ["T1047"],
         "note": "WMI __EventFilter+__EventConsumer subscription"},
    ],
    "empire_persistence_registry_modification_run_keys_elevated_user": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell elevated"},
        {"tid": "T1547.001", "tactic": "Persistence", "alts": [],
         "note": "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run added"},
    ],
    "empire_schtasks_creation_execution_elevated_user": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell elevated"},
        {"tid": "T1053.005", "tactic": "Persistence",
         "alts": [],
         "note": "schtasks /create scheduled task"},
        {"tid": "T1053.005", "tactic": "Execution",
         "alts": [],
         "note": "Task triggers payload at logon"},
    ],
    "empire_wmi_local_event_subscriptions_elevated_user": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell elevated"},
        {"tid": "T1546.003", "tactic": "Persistence",
         "alts": ["T1047"],
         "note": "Set-WmiInstance __EventFilter/__EventConsumer subscription"},
    ],

    # ── atomic/privilege_escalation ────────────────────────────────────
    "cmd_service_mod_fax": [
        {"tid": "T1059.003", "tactic": "Execution", "alts": [],
         "note": "cmd runs sc"},
        {"tid": "T1543.003", "tactic": "Persistence",
         "alts": ["T1574.011"],
         "note": "sc config Fax binPath rebind"},
        {"tid": "T1543.003", "tactic": "Privilege Escalation",
         "alts": [],
         "note": "Service starts SYSTEM payload"},
    ],
    "empire_uac_shellapi_fodhelper": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell user-context"},
        {"tid": "T1548.002", "tactic": "Privilege Escalation",
         "alts": ["T1218"],
         "note": "fodhelper.exe ms-settings UAC bypass"},
    ],
    "empire_invoke_runas": [
        {"tid": "T1059.001", "tactic": "Execution", "alts": [],
         "note": "Empire PowerShell stager (encoded -enc launcher)"},
        {"tid": "T1134.002", "tactic": "Privilege Escalation",
         "alts": ["T1134", "T1078"],
         "note": "Invoke-RunAs uses CreateProcessWithLogonW → EID 4648 explicit-creds logon, spawn as alt user"},
    ],

    # ── compound ───────────────────────────────────────────────────────
    "metasploit_logonpasswords_lsass_memory_dump": [
        {"tid": "T1059.005", "tactic": "Execution",
         "alts": ["T1059"],
         "note": "Initial cscript/payload (Metasploit launcher)"},
        {"tid": "T1055", "tactic": "Defense Evasion",
         "alts": ["T1055.001"],
         "note": "Meterpreter migrate/inject"},
        {"tid": "T1003.001", "tactic": "Credential Access",
         "alts": ["T1003"],
         "note": "Mimikatz logonpasswords on LSASS"},
    ],
    "metasploit_procdump_lsass_memory_dump": [
        {"tid": "T1059", "tactic": "Execution",
         "alts": ["T1059.005","T1059.001"],
         "note": "Metasploit payload exec"},
        {"tid": "T1055", "tactic": "Defense Evasion",
         "alts": ["T1055.001"],
         "note": "Meterpreter inject"},
        {"tid": "T1003.001", "tactic": "Credential Access",
         "alts": ["T1003"],
         "note": "procdump.exe lsass.exe lsass.dmp"},
    ],
}


def get_flow(scenario: str) -> list[dict]:
    """...attack flow. ...prefix → contains ..."""
    if scenario in ATTACK_FLOWS:
        return ATTACK_FLOWS[scenario]
    # try strip timestamp suffix
    import re
    stem = re.sub(r"_(?:20\d{2}[-T]?\d{2}[-T]?\d{2}\S*|\d{4,})$", "", scenario)
    if stem in ATTACK_FLOWS:
        return ATTACK_FLOWS[stem]
    for k, v in ATTACK_FLOWS.items():
        if scenario.startswith(k) or k in scenario:
            return v
    return []


def all_acceptable_tids(flow: list[dict]) -> set[str]:
    """flow ...tid (primary + alts)."""
    tids: set[str] = set()
    for step in flow:
        tids.add(step["tid"])
        for a in step.get("alts", []):
            tids.add(a)
    return tids


if __name__ == "__main__":
    print(f"Total scenarios: {len(ATTACK_FLOWS)}")
    for name, flow in sorted(ATTACK_FLOWS.items()):
        print(f"\n{name} ({len(flow)} steps):")
        for i, step in enumerate(flow, 1):
            alts = f" alts={step['alts']}" if step.get("alts") else ""
            print(f"  {i}. {step['tid']:<12} ({step['tactic']:<22}) -- {step['note']}{alts}")
