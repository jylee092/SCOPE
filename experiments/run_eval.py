"""
Standalone evaluation driver for 1차 실험.

main.py evaluate는 pandas가 필요한 전체 파이프라인을 import 하므로
별도로 evaluator 모듈만 사용하는 드라이버.

추가 기능
---------
- candidate `tactic` 필드가 비어있으면 MITRE CSV에서 채움
- **lenient Hit@K**: pred TID 가 GT TID 의 부모/자식이면 hit 인정
  (e.g., pred=T1087, GT=T1087.001 → 인정)
- per-scenario TTP Hit/MRR + chain F1 표 출력

사용:
    cd Final_Code
    python experiments/run_eval.py
"""
from __future__ import annotations
import csv, io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MITRE_CSV  = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"

sys.path.insert(0, str(ROOT))
from pipeline.evaluator import (
    load_ground_truth, evaluate_tactic_chain, aggregate_report, print_report,
)


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


def patch_candidate_tactics(ttp_results: list[dict], tm: dict[str, str]) -> None:
    for r in ttp_results:
        for c in r.get("similar_techniques", []) or []:
            if not c.get("tactic"):
                c["tactic"] = resolve_tactic(c.get("technique_id", ""), tm)


def tid_match(pred: str, gt: str, lenient: bool) -> bool:
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    if not lenient:
        return False
    p_root = pred.split(".")[0]
    g_root = gt.split(".")[0]
    return p_root == g_root  # parent/child or sibling-of-parent


def evaluate_ttp_lenient(
    gt: dict,
    ttp_results: list[dict],
    k: int = 5,
    confidence_filter: float | None = None,
) -> dict:
    """Strict + lenient Hit/MRR. 두 metric 동시 계산."""
    metrics = []
    for r in ttp_results:
        gid = r["group_id"]
        if gid not in gt:
            continue
        truth = gt[gid]
        if not truth["is_tp"]:
            continue

        # confidence_filter: rule confidence 컷오프 (auto_label 의 group confidence)
        # ttp_results 에는 confidence 가 없으므로 group_id 로 매칭하기 어려움.
        # 별도 처리 필요 시 호출자가 ttp_results 를 사전 필터.

        gt_tid = truth["technique_id"]
        candidates = r.get("similar_techniques", [])
        ranked = [c["technique_id"] for c in candidates]

        hit1_s = int(bool(ranked) and tid_match(ranked[0], gt_tid, False))
        hit1_l = int(bool(ranked) and tid_match(ranked[0], gt_tid, True))
        hitk_s = int(any(tid_match(t, gt_tid, False) for t in ranked[:k]))
        hitk_l = int(any(tid_match(t, gt_tid, True)  for t in ranked[:k]))

        rr_s = rr_l = 0.0
        for rank, t in enumerate(ranked, start=1):
            if rr_s == 0 and tid_match(t, gt_tid, False):
                rr_s = 1.0 / rank
            if rr_l == 0 and tid_match(t, gt_tid, True):
                rr_l = 1.0 / rank
            if rr_s > 0 and rr_l > 0:
                break

        pred_tactic = candidates[0].get("tactic", "") if candidates else ""

        metrics.append({
            "group_id": gid,
            "gt_technique_id": gt_tid,
            "gt_tactic": truth["tactic"],
            "pred_technique_id": ranked[0] if ranked else "",
            "pred_tactic": pred_tactic,
            "hit_at_1": hit1_s,        # strict
            f"hit_at_{k}": hitk_s,
            "reciprocal_rank": rr_s,
            "hit_at_1_lenient": hit1_l,
            f"hit_at_{k}_lenient": hitk_l,
            "rr_lenient": rr_l,
        })

    n = len(metrics)
    if n == 0:
        return {"n": 0, "hit_at_1": 0, f"hit_at_{k}": 0, "mrr": 0,
                "hit_at_1_lenient": 0, f"hit_at_{k}_lenient": 0,
                "mrr_lenient": 0, "details": []}

    return {
        "n": n,
        "hit_at_1":          sum(m["hit_at_1"]          for m in metrics) / n,
        f"hit_at_{k}":       sum(m[f"hit_at_{k}"]       for m in metrics) / n,
        "mrr":               sum(m["reciprocal_rank"]   for m in metrics) / n,
        "hit_at_1_lenient":  sum(m["hit_at_1_lenient"]  for m in metrics) / n,
        f"hit_at_{k}_lenient": sum(m[f"hit_at_{k}_lenient"] for m in metrics) / n,
        "mrr_lenient":       sum(m["rr_lenient"]       for m in metrics) / n,
        "details": metrics,
    }


def filter_ttp_by_confidence(
    ttp_results: list[dict],
    annotation: dict,
    min_conf: float,
) -> list[dict]:
    """annotation 의 confidence 필드로 ttp_results 필터."""
    conf_map = {g["group_id"]: float(g.get("confidence") or 0)
                for g in annotation.get("groups", [])}
    return [r for r in ttp_results if conf_map.get(r["group_id"], 0) >= min_conf]


def run_one(min_conf: float, label: str) -> None:
    tactic_map = load_tactic_map(MITRE_CSV)

    per_scenario_rows: list[dict] = []
    eval_results: list[dict] = []

    for ann in sorted(OUTPUT_DIR.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt:
            continue
        with open(ann, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")

        rec = {"scenario": ann.parent.name}
        if ttp_fp.exists():
            with open(ttp_fp, "r", encoding="utf-8") as f:
                ttp_results = json.load(f)
            patch_candidate_tactics(ttp_results, tactic_map)
            if min_conf > 0:
                ttp_results = filter_ttp_by_confidence(ttp_results, ann_data, min_conf)
            rec["ttp"] = evaluate_ttp_lenient(gt, ttp_results)
        if vit_fp.exists():
            with open(vit_fp, "r", encoding="utf-8") as f:
                viterbi = json.load(f)
            rec["chain"] = evaluate_tactic_chain(gt, viterbi)

        eval_results.append(rec)

        t = rec.get("ttp", {})
        c = rec.get("chain", {})
        per_scenario_rows.append({
            "scenario": rec["scenario"],
            "n_tp": t.get("n", 0),
            "hit@1": round(t.get("hit_at_1", 0), 3),
            "hit@5": round(t.get("hit_at_5", 0), 3),
            "mrr":   round(t.get("mrr", 0), 3),
            "hit@1_l": round(t.get("hit_at_1_lenient", 0), 3),
            "hit@5_l": round(t.get("hit_at_5_lenient", 0), 3),
            "mrr_l":   round(t.get("mrr_lenient", 0), 3),
            "chain_jaccard":  c.get("tactic_set_jaccard", 0),
            "chain_edit_sim": c.get("normalized_edit_similarity", 0),
            "chain_f1":       c.get("f1", 0),
        })

    # Aggregate (간이 계산)
    ttps = [r["ttp"] for r in eval_results if r.get("ttp", {}).get("n", 0) > 0]
    chains = [r["chain"] for r in eval_results if "chain" in r]
    n_groups = sum(t["n"] for t in ttps)

    def _avg(rows, key):
        return sum(r.get(key, 0) for r in rows) / len(rows) if rows else 0

    print()
    print("═" * 90)
    print(f"  AGGREGATE  (label={label}, min_confidence={min_conf})")
    print("═" * 90)
    print(f"  scenarios={len(eval_results)}  TP-groups-evaluated={n_groups}")
    print(f"  TTP   strict   : H@1={_avg(ttps,'hit_at_1'):.4f}  "
          f"H@5={_avg(ttps,'hit_at_5'):.4f}  MRR={_avg(ttps,'mrr'):.4f}")
    print(f"  TTP   lenient  : H@1={_avg(ttps,'hit_at_1_lenient'):.4f}  "
          f"H@5={_avg(ttps,'hit_at_5_lenient'):.4f}  MRR={_avg(ttps,'mrr_lenient'):.4f}")
    print(f"  Chain          : Jacc={_avg(chains,'tactic_set_jaccard'):.4f}  "
          f"EditSim={_avg(chains,'normalized_edit_similarity'):.4f}  "
          f"P={_avg(chains,'precision'):.4f}  R={_avg(chains,'recall'):.4f}  "
          f"F1={_avg(chains,'f1'):.4f}")

    # 콘솔: 시나리오별 표 (strict + lenient)
    print()
    print("─" * 130)
    print(f"  {'scenario':<55s} {'nTP':>4s}  "
          f"{'H@1':>5s} {'H@5':>5s} {'MRR':>5s}  "
          f"{'H1L':>5s} {'H5L':>5s} {'MRRL':>5s}  "
          f"{'Jac':>5s} {'Edit':>5s} {'F1':>5s}")
    print("─" * 130)
    for row in per_scenario_rows:
        print(f"  {row['scenario'][:55]:<55s} {row['n_tp']:>4d}  "
              f"{row['hit@1']:>5.2f} {row['hit@5']:>5.2f} {row['mrr']:>5.2f}  "
              f"{row['hit@1_l']:>5.2f} {row['hit@5_l']:>5.2f} {row['mrr_l']:>5.2f}  "
              f"{row['chain_jaccard']:>5.2f} {row['chain_edit_sim']:>5.2f} "
              f"{row['chain_f1']:>5.2f}")
    print("─" * 130)

    # 저장
    suffix = label.replace(" ", "_")
    out_json = OUTPUT_DIR / f"per_scenario_eval_{suffix}.json"
    out_csv  = OUTPUT_DIR / f"per_scenario_eval_{suffix}.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    if per_scenario_rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_scenario_rows[0].keys()))
            w.writeheader(); w.writerows(per_scenario_rows)
    print(f"  saved: {out_json}, {out_csv}")


def main() -> None:
    run_one(min_conf=0.0,  label="all")
    run_one(min_conf=0.5,  label="conf-ge-0p5")
    run_one(min_conf=1.0,  label="conf-eq-1")


if __name__ == "__main__":
    main()
