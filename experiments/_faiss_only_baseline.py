"""
FAISS-only chain metrics baseline.

목적: Viterbi 를 완전히 끄고 (각 그룹 = FAISS top-1 candidate) chain metrics
측정. Viterbi 가 실제로 가치를 더하는지 비교하기 위한 냉정한 baseline.

구현: viterbi_*.json 을 "each step = FAISS top-1" 로 덮어쓴 뒤 기존 eval 스크립트
실행. 다른 파이프라인 변경 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.attack_chain import (
    sort_results_by_time, load_tactic_map,
)
import pandas as pd

from experiments.run_eval_post_viterbi import _run as eval_post_run
from experiments.run_eval_v2 import main as eval_v2


def run_one(dataset_path: Path):
    config.configure_dataset(dataset_path)
    ttp_fp = config.TTP_MAPPING_JSON_PATH
    fcsv_fp = config.FINALE_CSV_PATH
    if not (ttp_fp.exists() and fcsv_fp.exists()):
        return False
    with open(ttp_fp, encoding="utf-8") as f:
        ttp = json.load(f)
    df = pd.read_csv(fcsv_fp)
    df["TimeCreated"] = pd.to_datetime(df["TimeCreated"], errors="coerce")
    sorted_res = sort_results_by_time(ttp, df)

    # Build breakdown as if each step picked FAISS top-1.
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    def resolve_tactic(tid):
        ts = tactic_map.get(tid, [])
        if ts:
            return ts[0]
        parent = tid.split(".", 1)[0] if "." in tid else None
        if parent:
            ts = tactic_map.get(parent, [])
            if ts:
                return ts[0]
        return "Unknown"

    breakdown = []
    for idx, r in enumerate(sorted_res):
        cands = r.get("similar_techniques", [])
        if not cands:
            continue
        top1 = cands[0]
        breakdown.append({
            "step":             idx + 1,
            "group_idx":        idx,
            "group_id":         r.get("group_id", ""),
            "technique_id":     top1.get("technique_id", ""),
            "technique_name":   top1.get("technique_name", ""),
            "tactic":           top1.get("tactic") or resolve_tactic(top1.get("technique_id", "")),
            "similarity":       round(float(top1.get("similarity", 0.0)), 4),
            "log_emission":     0.0,
            "transition_from":  None,
            "transition_weight": None,
            "transition_rule":   None,
            "transition_note":   None,
            "log_transition":    None,
            "skip_distance":     0,
        })
    with open(config.VITERBI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, ensure_ascii=False, indent=2)
    return True


def main():
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    ok = 0
    for ds in datasets:
        try:
            if run_one(ds):
                ok += 1
        except Exception as e:
            print(f"  ERROR {ds.name}: {type(e).__name__}: {e}")
    print(f"FAISS-only breakdown written: {ok}/{len(datasets)}")

    eval_post_run(strong_only=False, label="ALL TPs")
    eval_v2()


if __name__ == "__main__":
    main()
