"""
Technique-level In/Out Entity Type Signatures

MITRE ATT&CK CSV의 Log Sources (Data Component) 필드에서
각 technique이 소비(In)하고 생산(Out)하는 entity type을 자동 추출한다.

Entity types: {process, file, registry, network, service, user}

공개 API
--------
build_technique_io(mitre_csv_path) -> dict[str, dict]
load_or_build_technique_io(mitre_csv_path, cache_path) -> dict[str, dict]
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Data Component → (entity_type, direction) 매핑
#
# direction: "in"  = 기존 아티팩트를 읽거나 사용
#            "out" = 새 아티팩트를 생성·변경·삭제
#            "both"= 실행 과정에서 양쪽 모두
# ──────────────────────────────────────────────────────────────────────────────
_DC_MAP: dict[str, tuple[str, str]] = {
    # Process
    "Process Creation":       ("process", "out"),
    "Process Access":         ("process", "in"),
    "Process Metadata":       ("process", "in"),
    "Process Modification":   ("process", "out"),
    "Process Termination":    ("process", "out"),
    "Command Execution":      ("process", "both"),
    "Script Execution":       ("process", "both"),
    "OS API Execution":       ("process", "in"),
    "Module Load":            ("process", "in"),

    # File
    "File Access":            ("file", "in"),
    "File Creation":          ("file", "out"),
    "File Deletion":          ("file", "out"),
    "File Metadata":          ("file", "in"),
    "File Modification":      ("file", "out"),
    "Firmware Modification":  ("file", "out"),

    # Registry
    "Windows Registry Key Access":       ("registry", "in"),
    "Windows Registry Key Creation":     ("registry", "out"),
    "Windows Registry Key Modification": ("registry", "out"),

    # Network
    "Network Connection Creation": ("network", "out"),
    "Network Traffic Content":     ("network", "both"),
    "Network Traffic Flow":        ("network", "both"),
    "Network Share Access":        ("network", "in"),
    "Firewall Rule Modification":  ("network", "out"),
    "Domain Registration":         ("network", "out"),

    # Service
    "Service Creation":            ("service", "out"),
    "Service Metadata":            ("service", "in"),
    "Service Modification":        ("service", "out"),
    "Scheduled Job Creation":      ("service", "out"),
    "Scheduled Job Metadata":      ("service", "in"),
    "Scheduled Job Modification":  ("service", "out"),
    "WMI Creation":                ("service", "out"),
    "Driver Load":                 ("service", "in"),

    # User / Identity
    "User Account Authentication": ("user", "in"),
    "User Account Creation":       ("user", "out"),
    "User Account Deletion":       ("user", "out"),
    "User Account Metadata":       ("user", "in"),
    "User Account Modification":   ("user", "out"),
    "Logon Session Creation":      ("user", "out"),
    "Logon Session Metadata":      ("user", "in"),
    "Web Credential Creation":     ("user", "out"),

    # Active Directory → user
    "Active Directory Credential Request": ("user", "in"),
    "Active Directory Object Access":      ("user", "in"),
    "Active Directory Object Creation":    ("user", "out"),
    "Active Directory Object Deletion":    ("user", "out"),
    "Active Directory Object Modification":("user", "out"),

    # Storage / Volume → file
    "Drive Access":         ("file", "in"),
    "Drive Creation":       ("file", "out"),
    "Drive Modification":   ("file", "out"),
    "Volume Creation":      ("file", "out"),

    # Misc
    "Named Pipe Metadata":      ("process", "in"),
    "Application Log Content":  ("file", "in"),
    "Host Status":              ("process", "in"),
    "Cloud Service Metadata":   ("network", "in"),
    "Cloud Storage Access":     ("network", "in"),
}


def _parse_dc_name(dc_str: str) -> str:
    """'Process Creation (DC0032)' → 'Process Creation'"""
    idx = dc_str.find("(DC")
    if idx > 0:
        return dc_str[:idx].strip()
    return dc_str.strip()


def build_technique_io(mitre_csv_path: str | Path) -> dict[str, dict]:
    """
    MITRE CSV → technique별 In/Out entity type 매핑.

    Returns:
        {
            "T1059.001": {"in": ["process", "file"], "out": ["process"]},
            ...
        }
    """
    df = pd.read_csv(mitre_csv_path)

    id_col = next(
        (c for c in df.columns if c.lower() in ("technique_id", "id", "techniqueid")),
        None,
    )
    ls_col = next((c for c in df.columns if "log" in c.lower() and "source" in c.lower()), None)

    if not id_col or not ls_col:
        raise ValueError(f"필요 컬럼 없음: {list(df.columns)}")

    result: dict[str, dict] = {}
    unmapped: set[str] = set()

    for _, row in df.iterrows():
        tid = str(row[id_col]).strip()
        raw = row.get(ls_col)
        if pd.isna(raw) or not raw:
            continue

        try:
            sources = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        in_set: set[str] = set()
        out_set: set[str] = set()

        for src in sources:
            dc = src.get("Data Component", "")
            if not dc:
                continue
            dc_clean = _parse_dc_name(dc)
            mapping = _DC_MAP.get(dc_clean)
            if mapping is None:
                unmapped.add(dc_clean)
                continue

            entity, direction = mapping
            if direction == "in":
                in_set.add(entity)
            elif direction == "out":
                out_set.add(entity)
            else:  # both
                in_set.add(entity)
                out_set.add(entity)

        if in_set or out_set:
            result[tid] = {
                "in": sorted(in_set),
                "out": sorted(out_set),
            }

    if unmapped:
        print(f"  [technique_io] 미매핑 Data Component {len(unmapped)}개: {unmapped}")

    print(f"  [technique_io] {len(result)}개 technique In/Out 추출 완료")
    return result


def load_or_build_technique_io(
    mitre_csv_path: str | Path,
    cache_path: str | Path | None = None,
) -> dict[str, dict]:
    """캐시가 있으면 로드, 없으면 빌드 후 저장."""
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  [technique_io] 캐시 로드: {cache_path} ({len(data)}개)")
            return data

    result = build_technique_io(mitre_csv_path)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  [technique_io] 캐시 저장: {cache_path}")

    return result
