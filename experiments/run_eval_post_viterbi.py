"""
Post-Viterbi per-group TTP evaluation.

Stage 1 (FAISS top-K)와 Stage 2 (Viterbi가 top-K 중 고른 최종 TID) 각각에 대해
per-group TTP plausibility를 측정해 비교한다.

목적: "Viterbi 전이확률이 실제로 top-1을 더 정답에 가깝게 rerank하는가?"
  - FAISS_top1 plausibility   : FAISS 그대로의 top-1
  - Viterbi_pick plausibility : Viterbi가 해당 그룹에서 최종 선택한 TID
  - improvement = Viterbi - FAISS_top1
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT))

from pipeline.evaluator import load_ground_truth
from experiments.run_eval import load_tactic_map, patch_candidate_tactics
from experiments.attack_flows import get_flow, all_acceptable_tids
from experiments.run_eval_plausible import _is_strong_tp, tid_family_match

MITRE_CSV = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"


def anchor_tid_from_gid(gid: str) -> str:
    """GID format like 'T1003_3948' or 'T1021_003_5094' → 'T1003' / 'T1021.003'.

    Convention: the GID is built from the anchor rule's TID and an event index.
    Sub-techniques are encoded only when there are 3 underscore-separated parts
    AND the middle part is a 3-digit zero-padded ID (e.g., 'T1021_003_5094').
    A 2-part GID like 'T1112_128' is always 'T1112' (128 is just the event index).
    """
    if not gid or not gid.startswith("T"):
        return ""
    parts = gid.split("_")
    head = parts[0]
    if len(parts) >= 3 and len(parts[1]) == 3 and parts[1].isdigit():
        return f"{head}.{parts[1]}"
    return head


def _match_any(pred: str, acceptable: set[str]) -> bool:
    return any(tid_family_match(pred, a) for a in acceptable)


def _run(strong_only: bool, label: str):
    tm = load_tactic_map(MITRE_CSV)
    rows = []
    n_total = n_faiss = n_viterbi = n_improved = n_regressed = 0

    for ann in sorted(OUTPUT_DIR.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt:
            continue
        with open(ann, encoding="utf-8") as f:
            ad = json.load(f)
        scenario = ad.get("scenario", ann.parent.name)
        flow = get_flow(scenario)
        if not flow:
            continue
        scenario_acceptable = set(all_acceptable_tids(flow))
        for a in list(scenario_acceptable):
            scenario_acceptable.add(a.split(".")[0])

        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")
        if not ttp_fp.exists() or not vit_fp.exists():
            continue

        with open(ttp_fp, encoding="utf-8") as f:
            ttp = json.load(f)
        with open(vit_fp, encoding="utf-8") as f:
            vit = json.load(f)
        patch_candidate_tactics(ttp, tm)

        ttp_by_gid = {r["group_id"]: r for r in ttp}
        vit_tid_by_gid = {b["group_id"]: b["technique_id"] for b in vit}

        for gid, truth in gt.items():
            if not truth["is_tp"]:
                continue
            if strong_only and not _is_strong_tp(truth.get("notes", "")):
                continue
            r = ttp_by_gid.get(gid)
            if not r:
                continue
            cands = r.get("similar_techniques", [])[:5]
            top5_tids = [c["technique_id"] for c in cands]
            if not top5_tids:
                continue

            faiss_top1 = top5_tids[0]
            viterbi_pick = vit_tid_by_gid.get(gid, "")

            # Per-group acceptable = scenario reference flow
            #                       ∪ scenario auto-labeled GT for the group
            #                       ∪ anchor rule's TID (extracted from GID).
            # Rationale: a behavior group is a TP if any of these is recovered.
            # Auto-labeling assigns the scenario's main TID to every TP group, which
            # over-penalizes groups that legitimately realize a supporting technique
            # (e.g., T1033 discovery as part of a T1003 credential-dump scenario).
            # Adding the anchor TID treats agreement with the rule that triggered the
            # group as an acceptable hit.
            acceptable = set(scenario_acceptable)
            gt_tid = truth.get("technique_id", "")
            if gt_tid:
                acceptable.add(gt_tid)
                acceptable.add(gt_tid.split(".")[0])
            anchor_tid = anchor_tid_from_gid(gid)
            if anchor_tid:
                acceptable.add(anchor_tid)
                acceptable.add(anchor_tid.split(".")[0])

            # Viterbi가 skip해서 해당 group이 chain에 없으면 viterbi_pick=""
            faiss_hit   = _match_any(faiss_top1, acceptable)
            top5_hit    = any(_match_any(t, acceptable) for t in top5_tids)
            viterbi_hit = _match_any(viterbi_pick, acceptable) if viterbi_pick else False

            n_total += 1
            if faiss_hit:   n_faiss += 1
            if viterbi_hit: n_viterbi += 1
            if viterbi_hit and not faiss_hit: n_improved += 1
            if faiss_hit and not viterbi_hit: n_regressed += 1

            rows.append({
                "scenario": scenario[:50],
                "gid": gid,
                "acceptable": ",".join(sorted(acceptable))[:60],
                "faiss_top1": faiss_top1,
                "viterbi_pick": viterbi_pick,
                "top5": ",".join(top5_tids),
                "top5_hit": int(top5_hit),
                "faiss_hit": int(faiss_hit),
                "viterbi_hit": int(viterbi_hit),
                "in_chain": int(bool(viterbi_pick)),
                "delta": int(viterbi_hit) - int(faiss_hit),
            })

    if n_total == 0:
        print(f"[{label}] no groups")
        return

    print("=" * 100)
    print(f"  {label}  (per-group TTP: FAISS vs Viterbi rerank)")
    print("=" * 100)
    print(f"  total TP groups evaluated : {n_total}")
    print()
    print(f"  FAISS top-1 plausibility  : {n_faiss/n_total:.4f}  ({n_faiss}/{n_total})")
    print(f"  Viterbi pick plausibility : {n_viterbi/n_total:.4f}  ({n_viterbi}/{n_total})")
    print(f"  Δ (Viterbi - FAISS top-1) : {(n_viterbi - n_faiss)/n_total:+.4f}")
    print()
    n_in_chain = sum(r["in_chain"] for r in rows)
    print(f"  Groups IN Viterbi chain   : {n_in_chain}/{n_total} ({100*n_in_chain/n_total:.1f}%)")
    print(f"  Groups skipped by Viterbi : {n_total - n_in_chain}/{n_total}")
    print()
    print(f"  Improved (FAISS miss → Vit hit) : {n_improved}")
    print(f"  Regressed (FAISS hit → Vit miss): {n_regressed}")
    print()

    # save
    suffix = "strong" if strong_only else "all"
    out_csv = OUTPUT_DIR / f"eval_post_viterbi_{suffix}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  saved: {out_csv}\n")


def main():
    _run(strong_only=False, label="ALL TPs")
    _run(strong_only=True,  label="STRONG TPs ONLY")


if __name__ == "__main__":
    main()
