"""
Section 3-(2). LLM 호출 전 feature dict 정제

- 특정 값(경로, IP, 레지스트리, 파일명) → 추상 레이블
- 노이즈(정상 시스템 프로세스, 임시 파일) 제거
- 원본 불변, deepcopy로 반환
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────
_NOISE_PROCESSES = {
    "backgroundtaskhost.exe", "runtimebroker.exe", "browser_broker.exe",
    "textinputhost.exe",
    "windowsinternal.composableshell.experiences.textinput.inputapp.exe",
    "dwm.exe", "smss.exe", "wininit.exe", "services.exe", "csrss.exe",
}

_NOISE_ACCESS_SOURCES = {
    "svchost.exe", "csrss.exe", "wininit.exe", "services.exe",
    "msmpseng.exe", "senseir.exe",
}

_NOISE_FILE_PATTERNS = [
    "__psscriptpolicytest_",
    "moduleanalysiscache",
    "\\appdata\\local\\temp\\__pss",
]

_NOISE_DROP_IMAGES = {
    "svchost.exe", "sysmon.exe", "sysmon64.exe",
    "msiexec.exe", "tiworker.exe", "trustedinstaller.exe",
}


# ──────────────────────────────────────────────────────────────────────────────
# 추상화 치환
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize_path(path: Optional[str]) -> Optional[str]:
    """사용자 경로 → 환경변수. 확장자별 파일명 치환."""
    if not path:
        return path
    p = path
    p = re.sub(r"C:\\Users\\[^\\]+\\AppData\\Local\\Temp",
               "%TEMP%", p, flags=re.IGNORECASE)
    p = re.sub(r"C:\\Users\\[^\\]+\\AppData\\Roaming",
               "%APPDATA%", p, flags=re.IGNORECASE)
    p = re.sub(r"C:\\Users\\[^\\]+\\Desktop",
               "%USERPROFILE%" + "\\\\" + "Desktop", p, flags=re.IGNORECASE)
    p = re.sub(r"C:\\Users\\[^\\]+",
               "%USERPROFILE%", p, flags=re.IGNORECASE)
    p = re.sub(r"\b\w[\w\-\.]*\.dmp\b", "[memory_dump].dmp", p, flags=re.IGNORECASE)
    p = re.sub(r"\b\w[\w\-\.]*\.bat\b", "[batch_script].bat", p, flags=re.IGNORECASE)
    p = re.sub(r"\b\w[\w\-\.]*\.ps1\b", "[powershell_script].ps1", p, flags=re.IGNORECASE)
    p = re.sub(r"\b\w[\w\-\.]*\.vbs\b", "[vbscript].vbs", p, flags=re.IGNORECASE)
    p = re.sub(r"\b\w[\w\-\.]*\.hta\b", "[hta_script].hta", p, flags=re.IGNORECASE)
    return p


def _sanitize_cmdline(cmdline: Optional[str]) -> Optional[str]:
    """경로 치환 + 3~6자리 PID → [PID]."""
    if not cmdline:
        return cmdline
    c = _sanitize_path(cmdline)
    c = re.sub(r"(?<![/\\\w])(\d{3,6})(?![/\\\w\.\-])", "[PID]", c)
    return c


def _sanitize_ip(ip: Optional[str]) -> Optional[str]:
    """IP → 클래스 레이블."""
    if not ip:
        return ip
    if re.match(r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)", ip):
        return "[internal_ip]"
    if ip.startswith("127.") or ip == "::1":
        return "[loopback]"
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
        return "[external_ip]"
    return ip


def _sanitize_registry(key: Optional[str]) -> Optional[str]:
    """HKU\\S-... → HKCU\\..."""
    if not key:
        return key
    return re.sub(r"HKU\\S-\d+-\d+-\d+-\d+-\d+-\d+-\d+\\",
                  "HKCU" + "\\\\", key, flags=re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# 카테고리별 정제
# ──────────────────────────────────────────────────────────────────────────────
def _clean_execution_context(ctx: dict) -> dict:
    cleaned: list[dict] = []
    for c in ctx.get("process_chains", []):
        parent = (c.get("parent_image") or "").lower()
        child  = (c.get("child_image")  or "").lower()
        rel    = c.get("relation", "spawn")

        if rel == "access":
            if "lsass" in child:
                cleaned.append(c)
                continue
            if parent in _NOISE_ACCESS_SOURCES:
                continue
        else:
            if parent in _NOISE_PROCESSES and child in _NOISE_PROCESSES:
                continue

        entry = dict(c)
        if entry.get("cmdline"):
            entry["cmdline"] = _sanitize_cmdline(entry["cmdline"])
        cleaned.append(entry)

    return {**ctx, "process_chains": cleaned, "chain_depth": len(cleaned)}


def _clean_command_script(cmd: dict) -> dict:
    cleaned: list[dict] = []
    for e in cmd.get("entries", []):
        entry = dict(e)
        if entry.get("cmdline"):
            entry["cmdline"] = _sanitize_cmdline(entry["cmdline"])
        cleaned.append(entry)
    return {**cmd, "entries": cleaned}


def _clean_network(net: dict) -> dict:
    cleaned: list[dict] = []
    for c in net.get("connections", []):
        entry = dict(c)
        entry["src_ip"]     = _sanitize_ip(entry.get("src_ip"))
        entry["dst_ip"]     = _sanitize_ip(entry.get("dst_ip"))
        entry["query_name"] = "[queried_domain]" if entry.get("query_name") else None
        cleaned.append(entry)
    return {**net, "connections": cleaned}


def _clean_persistence(per: dict) -> dict:
    cleaned_signals: list[dict] = []
    for s in per.get("registry_signals", []):
        entry = dict(s)
        entry["target_object"] = _sanitize_registry(entry.get("target_object"))
        entry["details"]       = _sanitize_path(entry.get("details"))
        cleaned_signals.append(entry)

    cleaned_files: list[dict] = []
    for f in per.get("dropped_files", []):
        img  = (f.get("image") or "").lower()
        path = (f.get("path")  or "").lower()
        if img in _NOISE_DROP_IMAGES:
            continue
        if any(pat in path for pat in _NOISE_FILE_PATTERNS):
            continue
        entry = dict(f)
        entry["path"] = _sanitize_path(entry.get("path"))
        cleaned_files.append(entry)

    return {**per, "registry_signals": cleaned_signals, "dropped_files": cleaned_files}


def _clean_evasion(eva: dict) -> dict:
    cleaned_obf: list[dict] = []
    for e in eva.get("obfuscated_cmdlines", []):
        entry = dict(e)
        entry["cmdline"] = _sanitize_cmdline(entry.get("cmdline"))
        cleaned_obf.append(entry)

    cleaned_deleted: list[dict] = []
    for f in eva.get("deleted_files", []):
        entry = dict(f)
        entry["path"] = _sanitize_path(entry.get("path"))
        cleaned_deleted.append(entry)

    return {**eva, "obfuscated_cmdlines": cleaned_obf, "deleted_files": cleaned_deleted}


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────
def sanitize(features: dict) -> dict:
    """feature dict 정제 복사본 반환 (원본 불변)."""
    f    = deepcopy(features)
    feat = f["features"]

    feat["execution_context"] = _clean_execution_context(feat["execution_context"])
    feat["command_script"]    = _clean_command_script(feat["command_script"])
    feat["network"]           = _clean_network(feat["network"])
    feat["persistence"]       = _clean_persistence(feat["persistence"])
    feat["evasion"]           = _clean_evasion(feat["evasion"])
    # identity / temporal은 구조 값만 있어 추상화 불필요
    return f
