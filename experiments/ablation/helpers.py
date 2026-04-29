"""
Ablation helpers — no_grouping / no_llm 모드 전용 유틸.

- no_grouping: 룰 매칭을 우회하여 의미 있는 EID(Process Create, PowerShell ScriptBlock 등)
  마다 단일 이벤트 그룹 생성. technique_id는 "UNKNOWN"으로 지정되어 LLM+FAISS가 TTP를 결정.

- no_llm: LLM description 생성을 건너뛰고 feature dict를 그대로 문자열로 직렬화하여
  FAISS 쿼리에 사용.
"""
from __future__ import annotations

import pandas as pd


# 그룹핑 우회 시 anchor로 쓸 의미 있는 EID 집합
_SOLO_ANCHOR_EIDS = {1, 4688, 4104, 10, 11, 12, 13, 22, 23, 8, 4698, 7045}


def build_solo_groups(df: pd.DataFrame, max_groups: int = 200,
                       anchor_idxs: list[int] | None = None) -> list[dict]:
    """룰 매칭을 우회. 각 의미 이벤트를 1개짜리 그룹으로 구성.

    If `anchor_idxs` is given, use exactly those event row indices as solo
    anchors (one solo group per index) — used for the "anchor-only" ablation
    that isolates grouping from anchor selection. Otherwise fall back to
    uniform sampling over the meaningful-EID candidate pool.
    """
    if anchor_idxs is not None:
        groups: list[dict] = []
        for idx in anchor_idxs:
            if idx not in df.index:
                continue
            row = df.loc[idx]
            try:
                eid = int(row["EventID"])
            except (KeyError, TypeError, ValueError):
                eid = 0
            groups.append({
                "group_id":        f"ANCHOR_SOLO_{idx}",
                "technique_id":    "UNKNOWN",
                "technique_name":  f"Anchor-only solo (EID {eid})",
                "anchor_idx":      int(idx),
                "anchor_eid":      eid,
                "core_idxs":       [int(idx)],
                "supporting_idxs": [],
                "all_idxs":        [int(idx)],
                "supporting_def":  [],
                "supporting_hit":  [eid],
                "confidence":      1.0,
                "filter_passed":   True,
            })
        return groups

    mask = df["EventID"].isin(_SOLO_ANCHOR_EIDS)
    candidates = df[mask]
    if len(candidates) > max_groups:
        # 균등 샘플링
        step = len(candidates) // max_groups
        candidates = candidates.iloc[::step].head(max_groups)

    groups: list[dict] = []
    for idx, row in candidates.iterrows():
        eid = int(row["EventID"])
        groups.append({
            "group_id":        f"SOLO_{idx}",
            "technique_id":    "UNKNOWN",
            "technique_name":  f"Solo event (EID {eid})",
            "anchor_idx":      int(idx),
            "anchor_eid":      eid,
            "core_idxs":       [int(idx)],
            "supporting_idxs": [],
            "all_idxs":        [int(idx)],
            "supporting_def":  [],
            "supporting_hit":  [eid],
            "confidence":      1.0,
            "filter_passed":   True,
        })
    return groups


def feature_to_text(feat: dict) -> str:
    """Feature dict → 자연어-유사 요약 텍스트 (LLM 우회 모드).

    FAISS가 쿼리로 받아 MITRE 설명과 코사인 유사도 비교.
    LLM description 스타일을 모방하되 사실만 나열.
    """
    f = feat["features"]
    parts: list[str] = []

    # Process chains
    chains = (f.get("execution_context") or {}).get("process_chains", [])
    for c in chains[:10]:
        parent = c.get("parent_image") or "?"
        child = c.get("child_image") or "?"
        rel = c.get("relation", "spawn")
        if rel == "access":
            parts.append(f"{parent} accessed memory of {child}")
        else:
            parts.append(f"{parent} spawned {child}")
        cl = c.get("cmdline")
        if cl:
            parts.append(f"Command: {cl[:200]}")

    # Command lines
    cmd = f.get("command_script") or {}
    for e in (cmd.get("entries") or [])[:5]:
        cl = e.get("cmdline") or ""
        if cl:
            parts.append(f"Executed: {cl[:200]}")
    if cmd.get("has_obfuscation"):
        parts.append("Obfuscated command detected.")

    # Registry
    reg = (f.get("persistence") or {}).get("registry_signals", [])
    for s in reg[:8]:
        to = s.get("target_object") or ""
        det = s.get("details") or ""
        if to:
            parts.append(f"Registry modified: {to} = {det}")

    # Files
    files = (f.get("persistence") or {}).get("dropped_files", [])
    for fi in files[:5]:
        p = fi.get("path") or ""
        if p:
            parts.append(f"File created: {p}")

    # Network
    conns = (f.get("network") or {}).get("connections", [])
    for c in conns[:5]:
        d = c.get("direction", "")
        if d == "dns_query":
            parts.append(f"DNS query: {c.get('query_name', '')}")
        elif d == "listen":
            parts.append(f"Listening on port {c.get('listen_port', '')}")
        else:
            parts.append(f"Network connection to {c.get('dst_ip', '')}:{c.get('dst_port', '')}")

    # Evasion
    eva = f.get("evasion") or {}
    if eva.get("log_cleared"):
        parts.append("Event logs cleared.")
    if eva.get("deleted_files"):
        parts.append(f"{len(eva['deleted_files'])} files deleted.")

    # Identity
    idn = f.get("identity") or {}
    if idn.get("user"):
        parts.append(f"User: {idn.get('domain','')}\\{idn['user']}".strip("\\"))
    if idn.get("integrity_level"):
        parts.append(f"Integrity level: {idn['integrity_level']}")
    if idn.get("privilege_list"):
        parts.append(f"Privileges: {str(idn['privilege_list'])[:200]}")

    if not parts:
        return "No significant evidence observed in this event group."
    return ". ".join(parts)
