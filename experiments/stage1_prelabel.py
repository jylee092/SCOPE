"""
Stage 1: 자명 그룹 사전 라벨링.

다음 두 부류는 LLM 검토 없이 자동 결정:

1. **trivial-noise**: confidence==0 AND anchor 의 Image/CommandLine/ParentImage/
   TargetObject 가 모두 nan/null AND sample_logs 중 의미 있는 텍스트 없음
   → gt_is_true_positive=False, gt_label_source="auto-trivial"

2. **anchor-attacker-tool**: anchor.Image 에 시나리오의 명확한 attacker tool 이
   포함 (e.g., esentutl.exe, ntdsutil.exe, mimikatz.exe, sharpview.exe)
   → 임시로 시나리오의 primary technique 부여, gt_label_source="auto-anchor-tool"
   (Stage 2 에서 LLM 이 검증·세분화 가능)

나머지 그룹은 gt_*=None 으로 두고 Stage 2 (Claude LLM) 가 처리.

산출
----
- annotation.json 의 gt_* 필드를 위 결과로 갱신
- gt_label_source, gt_step_index, gt_confidence 메타 필드 추가
- output/labeling_state.json: 시나리오별 (n_trivial, n_anchor_tool, n_pending) 통계

사용:
    cd Final_Code
    python experiments/stage1_prelabel.py
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
# 시나리오 → primary attacker-tool anchor (실행 파일 이름) 매핑
#   - anchor.Image basename 이 매치되면 high-confidence TP
#   - 시나리오의 expected primary technique 으로 라벨
# ---------------------------------------------------------------------------
ATTACKER_TOOLS: dict[str, list[tuple[str, str]]] = {
    # (basename, technique_id) — anchor 매치 시 부여
    "msf_record_mic": [
        ("payload.exe",   "T1123"),
    ],
    "cmd_dumping_ntds_dit_file_ntdsutil": [
        ("ntdsutil.exe",  "T1003.003"),
    ],
    "cmd_sam_copy_esentutl": [
        ("esentutl.exe",  "T1003.002"),
    ],
    "empire_mimikatz_logonpasswords": [
        ("mimikatz.exe",  "T1003.001"),
        ("powershell.exe","T1059.001"),  # Empire launcher / Invoke-Mimikatz
    ],
    "empire_shell_rubeus_asktgt_createnetonly": [
        ("rubeus.exe",    "T1558"),
        ("powershell.exe","T1059.001"),
    ],
    "psh_lsass_memory_dump_comsvcs": [
        ("rundll32.exe",  "T1003.001"),
        ("powershell.exe","T1059.001"),
    ],
    "cmd_process_herpaderping_mimiexplorer": [
        ("processherpaderping.exe","T1036.005"),
        ("mimiexplorer.exe",       "T1055"),
        ("wardog.exe",             "T1055"),
    ],
    "cmd_stop_event_logging_controlset001_minint_key": [
        ("reg.exe",       "T1562.002"),
        ("payload.exe",   "T1059"),
    ],
    "cmd_wevtutil_modify_security_eventlog_path": [
        ("wevtutil.exe",  "T1562.002"),
    ],
    "empire_dllinjection_LoadLibrary_CreateRemoteThread": [
        ("powershell.exe","T1055.001"),
    ],
    "psh_cmstp_execution_bypassuac": [
        ("cmstp.exe",     "T1218.003"),
        ("powershell.exe","T1059.001"),
    ],
    "cmd_seatbelt_group_user": [
        ("seatbelt.exe",  "T1082"),
    ],
    "empire_find_localadmin_smb_svcctl_OpenSCManager": [
        ("powershell.exe","T1069.001"),
    ],
    "empire_getsession_dcerpc_smb_srvsvc_NetSessEnum": [
        ("powershell.exe","T1049"),
    ],
    "empire_shell_net_local_users": [
        ("net.exe",       "T1087.001"),
        ("net1.exe",      "T1087.001"),
        ("powershell.exe","T1059.001"),
    ],
    "empire_shell_samr_EnumDomainUsers": [
        ("powershell.exe","T1087.002"),
        ("net.exe",       "T1087.002"),
    ],
    "cmd_sharpview_pcre_net": [
        ("sharpview.exe", "T1087.002"),
    ],
    "empire_launcher_vbs": [
        ("wscript.exe",   "T1059.005"),
        ("cscript.exe",   "T1059.005"),
        ("powershell.exe","T1059.001"),
    ],
    "psh_powershell_httplistener": [
        ("powershell.exe","T1059.001"),
    ],
    "psh_python_webserver": [
        ("python.exe",    "T1059.006"),
        ("powershell.exe","T1059.001"),
    ],
    "covenant_psremoting_command": [
        ("grunthttp.exe", "T1059.001"),
        ("wsmprovhost.exe","T1021.006"),
        ("powershell.exe","T1059.001"),
    ],
    "empire_psexec_dcerpc_tcp_svcctl": [
        ("powershell.exe","T1059.001"),
    ],
    "empire_psremoting_stager": [
        ("wsmprovhost.exe","T1021.006"),
        ("powershell.exe","T1059.001"),
    ],
    "empire_wmi_dcerpc_wmi_IWbemServices_ExecMethod": [
        ("powershell.exe","T1059.001"),
        ("wmiprvse.exe",  "T1047"),
    ],
    "purplesharp_ad_playbook_I": [
        ("purplesharp.exe","T1059.003"),
        ("wsmprovhost.exe","T1021.006"),
        ("powershell.exe", "T1059.001"),
    ],
    "cmd_userinitmprlogonscript_batch": [
        ("reg.exe",       "T1037.001"),
    ],
    "covenant_persistwmi": [
        ("grunthttp.exe", "T1059.001"),
        ("wmiprvse.exe",  "T1546.003"),
    ],
    "empire_persistence_registry_modification_run_keys_elevated_user": [
        ("powershell.exe","T1547.001"),
        ("reg.exe",       "T1547.001"),
    ],
    "empire_schtasks_creation_execution_elevated_user": [
        ("schtasks.exe",  "T1053.005"),
        ("powershell.exe","T1059.001"),
    ],
    "empire_wmi_local_event_subscriptions_elevated_user": [
        ("powershell.exe","T1546.003"),
        ("wmiprvse.exe",  "T1546.003"),
    ],
    "cmd_service_mod_fax": [
        ("sc.exe",        "T1543.003"),
        ("powershell.exe","T1059.001"),
    ],
    "empire_uac_shellapi_fodhelper": [
        ("fodhelper.exe", "T1548.002"),
        ("powershell.exe","T1059.001"),
    ],
    "metasploit_logonpasswords_lsass_memory_dump": [
        ("payload.exe",   "T1059"),
        ("cscript.exe",   "T1059.005"),
        ("rundll32.exe",  "T1003.001"),
    ],
    "metasploit_procdump_lsass_memory_dump": [
        ("procdump.exe",  "T1003.001"),
        ("payload.exe",   "T1059"),
    ],
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
    if not tid:
        return ""
    if tid in tm:
        return tm[tid]
    return tm.get(tid.split(".")[0], "")


def scenario_key(scenario: str) -> str:
    """annotation scenario 이름에서 timestamp 제거."""
    m = re.match(r"^(.*?)(?:_(?:20\d{2}[-T]?\d{2}[-T]?\d{2}\S*|\d{4,}))?$", scenario)
    stem = m.group(1) if m else scenario
    return re.sub(r"[_-]+(?:\d+)?$", "", stem)


def find_attacker_tools(scenario: str) -> list[tuple[str, str]]:
    if scenario in ATTACKER_TOOLS:
        return ATTACKER_TOOLS[scenario]
    key = scenario_key(scenario)
    if key in ATTACKER_TOOLS:
        return ATTACKER_TOOLS[key]
    for k, v in ATTACKER_TOOLS.items():
        if scenario.startswith(k) or k in scenario:
            return v
    return []


def _norm(v) -> str:
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def is_trivial_noise(group: dict) -> bool:
    """confidence==0 + anchor 텅 + sample_logs 무내용."""
    if float(group.get("confidence") or 0) > 0:
        return False
    a = group.get("anchor") or {}
    has_anchor_text = any(_norm(a.get(k)) for k in
                          ("Image", "CommandLine", "ParentImage", "TargetObject"))
    if has_anchor_text:
        return False
    for s in group.get("sample_logs", []) or []:
        for k in ("Image", "CommandLine", "ParentImage", "TargetObject"):
            if _norm(s.get(k)):
                return False
    return True


def matched_attacker_tool(group: dict, tools: list[tuple[str, str]]) -> tuple[str, str] | None:
    """anchor.Image basename 이 attacker tool list 와 매치되면 (tool, tid) 반환."""
    a = group.get("anchor") or {}
    img = _norm(a.get("Image"))
    if not img:
        return None
    base = img.replace("\\", "/").split("/")[-1].lower()
    for tool, tid in tools:
        if tool.lower() == base:
            return (tool, tid)
    return None


def reset_gt_fields(group: dict) -> None:
    for k in ("gt_is_true_positive", "gt_technique_id", "gt_technique_name",
              "gt_tactic", "gt_notes", "gt_label_source",
              "gt_confidence", "gt_step_index"):
        group[k] = None


def stage1_label(group: dict, scenario: str, tools: list[tuple[str, str]],
                 tactic_map: dict[str, str]) -> str:
    """Returns one of: 'trivial', 'anchor-tool', 'pending'."""
    reset_gt_fields(group)
    if is_trivial_noise(group):
        group["gt_is_true_positive"] = False
        group["gt_label_source"] = "auto-trivial"
        group["gt_confidence"] = 1.0
        group["gt_notes"] = "auto-trivial: conf=0 + anchor empty + samples empty"
        return "trivial"

    tool_match = matched_attacker_tool(group, tools)
    if tool_match:
        tool, tid = tool_match
        group["gt_is_true_positive"] = True
        group["gt_technique_id"] = tid
        group["gt_tactic"] = resolve_tactic(tid, tactic_map)
        group["gt_label_source"] = "auto-anchor-tool"
        group["gt_confidence"] = 0.85   # high prior, may be refined by LLM
        group["gt_notes"] = f"auto-anchor-tool: anchor={tool} → {tid}"
        return "anchor-tool"

    group["gt_label_source"] = "pending"
    group["gt_confidence"] = None
    group["gt_notes"] = "pending Stage 2 (LLM)"
    return "pending"


def main() -> None:
    tactic_map = load_tactic_map(MITRE_CSV)
    files = sorted(OUTPUT_DIR.rglob("*_annotation.json"))
    print(f"Found {len(files)} annotation files")

    state: dict[str, dict] = {}
    totals = defaultdict(int)

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenario = data.get("scenario", fp.parent.name)
        tools = find_attacker_tools(scenario)

        cnt = defaultdict(int)
        for g in data.get("groups", []):
            cat = stage1_label(g, scenario, tools, tactic_map)
            cnt[cat] += 1
            totals[cat] += 1

        state[scenario] = {
            "n_total": len(data.get("groups", [])),
            "n_trivial": cnt["trivial"],
            "n_anchor_tool": cnt["anchor-tool"],
            "n_pending": cnt["pending"],
            "annotation_path": str(fp.relative_to(OUTPUT_DIR)),
        }

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    state_path = OUTPUT_DIR / "labeling_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"per_scenario": state, "totals": dict(totals)},
                  f, ensure_ascii=False, indent=2)

    print()
    print(f"{'scenario':<65} {'total':>6} {'triv':>6} {'tool':>6} {'pend':>6}")
    print("-" * 100)
    for s, v in sorted(state.items()):
        print(f"{s[:65]:<65} {v['n_total']:>6} {v['n_trivial']:>6} "
              f"{v['n_anchor_tool']:>6} {v['n_pending']:>6}")
    print("-" * 100)
    print(f"{'TOTAL':<65} {sum(v['n_total']      for v in state.values()):>6} "
          f"{totals['trivial']:>6} {totals['anchor-tool']:>6} {totals['pending']:>6}")
    print(f"\nState saved: {state_path}")


if __name__ == "__main__":
    main()
