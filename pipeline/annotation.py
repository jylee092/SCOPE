"""
Annotation Template Generator

파이프라인 실행 후 그룹별 요약을 JSON으로 덤프하여
수동 라벨링(Ground Truth)을 위한 템플릿을 생성한다.

공개 API
--------
generate_template(groups, df, output_path)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _summarize_log(row: pd.Series) -> dict:
    return {
        "idx": int(row.name),
        "EventID": int(row.get("EventID", 0)),
        "TimeCreated": str(row.get("TimeCreated", "")),
        "Image": str(row.get("Image", row.get("NewProcessName", "")))[:120],
        "CommandLine": str(row.get("CommandLine", ""))[:200],
        "ParentImage": str(row.get("ParentImage", row.get("ParentProcessName", "")))[:120],
        "TargetObject": str(row.get("TargetObject", ""))[:150],
    }


def _summarize_group(group: dict, df: pd.DataFrame) -> dict:
    all_idxs = group.get("all_idxs", [])
    valid = [i for i in all_idxs if i in df.index]
    logs = df.loc[valid].sort_values("TimeCreated") if valid else pd.DataFrame()

    anchor_idx = group.get("anchor_idx")
    anchor_row = df.loc[anchor_idx] if anchor_idx in df.index else None

    anchor_summary = None
    if anchor_row is not None:
        anchor_summary = {
            "idx": int(anchor_idx),
            "EventID": int(anchor_row.get("EventID", 0)),
            "Image": str(anchor_row.get("Image", ""))[:120],
            "CommandLine": str(anchor_row.get("CommandLine", ""))[:200],
            "TimeCreated": str(anchor_row.get("TimeCreated", "")),
        }

    eid_dist = {}
    if not logs.empty:
        for eid, cnt in logs["EventID"].value_counts().items():
            eid_dist[str(int(eid))] = int(cnt)

    sample_logs = [_summarize_log(row) for _, row in logs.head(15).iterrows()]

    return {
        "group_id": group["group_id"],
        "rule_technique_id": group["technique_id"],
        "rule_technique_name": group["technique_name"],
        "confidence": group.get("confidence", 0.0),
        "anchor": anchor_summary,
        "num_events": len(valid),
        "event_id_distribution": eid_dist,
        "sample_logs": sample_logs,

        "gt_technique_id": "",
        "gt_technique_name": "",
        "gt_tactic": "",
        "gt_is_true_positive": None,
        "gt_notes": "",
    }


_GT_FIELDS = ("gt_technique_id", "gt_technique_name", "gt_tactic",
              "gt_is_true_positive", "gt_notes")


def _load_existing_labels(path: Path) -> dict[str, dict]:
    """기존 annotation 파일에서 group_id → gt_* 매핑을 추출.

    파일이 없거나 파싱 실패 시 빈 dict 반환. gt_is_true_positive가 None인
    그룹은 "미라벨"로 간주해 매핑에서 제외 (blank 템플릿은 덮어쓰기 가능)."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    labels: dict[str, dict] = {}
    for g in data.get("groups", []):
        if g.get("gt_is_true_positive") is None:
            continue
        gid = g.get("group_id")
        if gid:
            labels[gid] = {k: g.get(k) for k in _GT_FIELDS}
    return labels


def generate_template(
    groups: list[dict],
    df: pd.DataFrame,
    output_path: str | Path,
    scenario_name: str = "",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 GT 라벨 보존: 재실행 시 수작업 라벨이 빈 템플릿으로 덮어쓰이지 않도록.
    existing = _load_existing_labels(output_path)

    entries = [_summarize_group(g, df) for g in groups if g.get("filter_passed", True)]
    preserved = 0
    for entry in entries:
        lbl = existing.get(entry["group_id"])
        if lbl is not None:
            for k in _GT_FIELDS:
                entry[k] = lbl[k]
            preserved += 1

    template = {
        "scenario": scenario_name,
        "total_groups": len(entries),
        "instruction": (
            "각 그룹의 sample_logs와 anchor를 확인한 뒤 gt_ 필드를 채워주세요. "
            "gt_is_true_positive: 해당 그룹이 실제 공격 행위인지 (true/false). "
            "gt_technique_id: 실제 MITRE technique ID (예: T1059.001). "
            "gt_tactic: 실제 tactic 이름 (예: Execution). "
            "gt_notes: 판단 근거나 메모."
        ),
        "groups": entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    msg = f"  Annotation 템플릿 저장: {output_path} ({len(entries)}개 그룹"
    if preserved:
        msg += f", GT 라벨 {preserved}개 유지"
    msg += ")"
    print(msg)
    return output_path
