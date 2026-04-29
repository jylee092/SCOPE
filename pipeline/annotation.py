"""
Annotation Template Generator


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
    """Load any existing annotation file and return a {group_id: gt_*}
    mapping. Groups marked as ``...`` are treated as blank templates and
    excluded from the mapping so a fresh template overwrites them."""
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
            "...sample_logs...anchor...gt_ ..."
            "gt_is_true_positive: ...true/false). "
            "gt_technique_id: ...MITRE technique ID (...: T1059.001). "
            "gt_tactic: ...tactic ...: Execution). "
            "gt_notes: ..."
        ),
        "groups": entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    msg = f"  Annotation ...: {output_path} ({len(entries)}..."
    if preserved:
        msg += f", GT ...{preserved}..."
    msg += ")"
    print(msg)
    return output_path
