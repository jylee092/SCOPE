"""
LLM-라벨링 결과를 annotation.json 에 적용.

사용:
    from experiments.apply_labels import apply_decisions
    apply_decisions("scenario_name_substring", {
        "group_id_1": {"tid": "T1003.001", "is_attack": True, "step": 2, "reason": "..."},
        "group_id_2": {"tid": None, "is_attack": False, "reason": "noise"},
        ...
    })

`step` 은 attack_flows 의 step index (1-based) — chain alignment 평가 용.
None 이면 attack flow 와 무관한 부수 활동.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MITRE_CSV  = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"


def _load_tactic_map() -> dict[str, str]:
    tm: dict[str, str] = {}
    with open(MITRE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = (row.get("ID") or "").strip()
            tac = (row.get("tactics") or "").split(",")[0].strip()
            if tid and tac:
                tm[tid] = tac
    return tm


_TACTIC_MAP = _load_tactic_map()


def _resolve_tactic(tid: str | None) -> str:
    if not tid:
        return ""
    if tid in _TACTIC_MAP:
        return _TACTIC_MAP[tid]
    return _TACTIC_MAP.get(tid.split(".")[0], "")


def apply_decisions(
    scenario_substr: str,
    decisions: dict[str, dict],
    source_label: str = "claude-llm",
) -> dict:
    """Apply decisions dict to matching scenario's annotation file.

    decisions 형식:
        { group_id: {
            "is_attack": bool,
            "tid": str | None,        # None if benign
            "tactic": str | None,     # auto-derived from tid if None
            "step": int | None,       # attack flow step index (1-based)
            "reason": str,
            "confidence": float,      # default 0.85
          }, ... }
    """
    files = list(OUTPUT_DIR.rglob(f"*{scenario_substr}*_annotation.json"))
    if not files:
        return {"error": f"no scenario matching: {scenario_substr}"}
    if len(files) > 1:
        # use exact match if possible
        exact = [f for f in files if scenario_substr in f.parent.name]
        if len(exact) == 1:
            files = exact
        else:
            return {"error": f"ambiguous: {len(files)} files match",
                    "files": [str(f) for f in files]}

    fp = files[0]
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    applied = 0; skipped_pre = 0; missed = []
    for g in data.get("groups", []):
        gid = g["group_id"]
        if gid not in decisions:
            continue
        if g.get("gt_is_true_positive") is not None and \
           g.get("gt_label_source", "").startswith("auto-anchor-tool"):
            # anchor-tool 라벨이 있어도 LLM 결정이 우선 (anchor-tool 은 prior)
            pass
        d = decisions[gid]
        is_atk = bool(d.get("is_attack", False))
        tid = d.get("tid")
        tac = d.get("tactic") or _resolve_tactic(tid) if tid else None

        g["gt_is_true_positive"] = is_atk
        g["gt_technique_id"] = tid if is_atk else None
        g["gt_tactic"] = tac if is_atk else None
        g["gt_label_source"] = source_label
        g["gt_confidence"] = float(d.get("confidence", 0.85))
        g["gt_step_index"] = d.get("step")
        g["gt_notes"] = d.get("reason", "")
        applied += 1

    seen = set(decisions.keys())
    actual = {g["group_id"] for g in data.get("groups", [])}
    missed = sorted(seen - actual)

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "scenario": data.get("scenario"),
        "file": str(fp.relative_to(OUTPUT_DIR)),
        "applied": applied,
        "missed_group_ids": missed,
    }


def status() -> None:
    """현재 모든 시나리오의 라벨링 상태 요약."""
    from collections import Counter
    print(f"{'scenario':<60} {'tot':>4} {'TP':>4} {'FP':>4} {'PEND':>5}")
    print("-" * 80)
    tot_pend = 0
    tot_total = 0
    for fp in sorted(OUTPUT_DIR.rglob("*_annotation.json")):
        with open(fp,"r",encoding="utf-8") as f: a = json.load(f)
        c = Counter()
        for g in a["groups"]:
            v = g.get("gt_is_true_positive")
            if v is True: c["TP"] += 1
            elif v is False: c["FP"] += 1
            else: c["PEND"] += 1
        rel = "/".join(fp.relative_to(OUTPUT_DIR).as_posix().split("/")[:3])
        print(f"{rel[-60:]:<60} {sum(c.values()):>4} {c['TP']:>4} {c['FP']:>4} {c['PEND']:>5}")
        tot_pend += c["PEND"]
        tot_total += sum(c.values())
    print("-" * 80)
    print(f"TOTAL: {tot_total}, PENDING: {tot_pend}")


if __name__ == "__main__":
    status()
