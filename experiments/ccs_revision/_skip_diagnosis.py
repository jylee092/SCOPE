"""
Why does hole-bridging (D=2) HURT on complete data (0.68 -> 0.49)?

Hypothesis: on complete logs every group is a legitimate chain step; the skip
operator routes around locally-weak-but-correct intermediate groups, producing
a SHORTER recovered chain that drops reference techniques -> recall-normalized
chain-LCS falls.

This re-runs the cached main pipeline at D=0 and D=2 (in memory, no API, no
canonical overwrite) and reports, per scenario: chain length, #skips fired,
tech-LCS; then dumps the worst-hurt scenarios' chains to show what got skipped.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
import pandas as pd
from collections import Counter as _C
from pipeline.attack_chain import (
    sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer,
    MultiDimTransitionScorer, load_campaign_library, topk_viterbi,
    apply_emission_confidence_bypass,
)
from pipeline.technique_io import load_or_build_technique_io
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment


def _prep(ttp_results, final_df):
    sr = sort_results_by_time(ttp_results, final_df)
    fb = float(getattr(config, "FAMILY_BOOST", 0.0)); fw = int(getattr(config, "FAMILY_BOOST_WIDTH", 10))
    if fb > 0 and fw > 0:
        for r in sr:
            cands = r.get("similar_techniques", [])
            if len(cands) < 2: continue
            wide = min(fw, len(cands))
            parents = [c["technique_id"].split(".", 1)[0] for c in cands[:wide]]
            pc = _C(parents); scored = []
            for i, c in enumerate(cands[:wide]):
                boosted = c.get("p_ttp", c.get("similarity", 0)) * (1.0 + fb * (pc[parents[i]] - 1))
                scored.append((boosted, i, c))
            scored.sort(key=lambda x: -x[0])
            rr = [c for _, _, c in scored] + cands[wide:]
            for nr, c in enumerate(rr, 1): c["rank"] = nr
            r["similar_techniques"] = rr
            if len(rr) >= 2:
                r["confidence_margin"] = float(rr[0].get("p_ttp", 0) - rr[1].get("p_ttp", 0))
    return sr


def run(group_nodes, multi, campaigns, bypass_thr, d):
    vit = topk_viterbi(group_nodes, multi, beam_k=config.VITERBI_BEAM_K,
                       max_skip=d, skip_penalty=config.VITERBI_SKIP_PENALTY,
                       transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=campaigns)
    if bypass_thr is not None:
        vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass_thr))
    return vit.score_breakdown


def main():
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    sem = get_semantic_scorer(getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
                              backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
                              calibration=getattr(config, "SEM_CALIBRATION", "linear")) if config.USE_SEMANTIC_SCORING else None
    cau = None
    if config.USE_CAUSAL_SCORING:
        tio = load_or_build_technique_io(str(config.MITRE_CSV_PATH),
                                         cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json")
        cau = CausalScorer(technique_io=tio)
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
    bypass_thr = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)

    recs = []
    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists()
                and config.FINALE_CSV_PATH.exists()):
            continue
        stem = ds.stem
        flow = get_flow(stem)
        if not flow: continue
        ttp = json.load(open(config.TTP_MAPPING_JSON_PATH, encoding="utf-8"))
        feats = json.load(open(config.FEATURE_RESULT_JSON_PATH, encoding="utf-8"))
        fdf = pd.read_csv(config.FINALE_CSV_PATH, low_memory=False)
        fdf["TimeCreated"] = pd.to_datetime(fdf["TimeCreated"], errors="coerce")
        sr = _prep(ttp, fdf)
        gnodes = build_group_nodes(sr, tactic_map, {f["group_id"]: f for f in feats})
        if not gnodes: continue
        tac = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
        multi = MultiDimTransitionScorer(tac_scorer=tac, sem_scorer=sem, cau_scorer=cau,
                                         w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
                                         self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0))
        bd0 = run(gnodes, multi, campaigns, bypass_thr, 0)
        bd2 = run(gnodes, multi, campaigns, bypass_thr, 2)
        lcs0 = evaluate_chain_alignment(stem, bd0, ref_flow=flow).get("technique_lcs_norm", 0.0) if bd0 else 0.0
        lcs2 = evaluate_chain_alignment(stem, bd2, ref_flow=flow).get("technique_lcs_norm", 0.0) if bd2 else 0.0
        nsk2 = sum(1 for n in bd2 if int(n.get("skip_distance", 0)) > 0)
        skipped_groups2 = sum(int(n.get("skip_distance", 0)) for n in bd2)
        recs.append({"stem": stem, "n_groups": len(gnodes),
                     "len0": len(bd0), "len2": len(bd2),
                     "nskips2": nsk2, "skipped_groups2": skipped_groups2,
                     "lcs0": lcs0, "lcs2": lcs2, "delta": lcs2 - lcs0,
                     "bd0": bd0, "bd2": bd2, "flow": flow})

    print(f"\n=== skip diagnosis on {len(recs)} scenarios ===")
    print(f"  mean chain len:  D0={mean(r['len0'] for r in recs):.1f}  D2={mean(r['len2'] for r in recs):.1f}")
    print(f"  mean #skips fired (D2): {mean(r['nskips2'] for r in recs):.2f}")
    print(f"  mean groups skipped-over (D2): {mean(r['skipped_groups2'] for r in recs):.2f}")
    print(f"  mean LCS: D0={mean(r['lcs0'] for r in recs):.4f}  D2={mean(r['lcs2'] for r in recs):.4f}")
    hurt = [r for r in recs if r["delta"] < -1e-6]
    print(f"  scenarios where D2 HURT: {len(hurt)}/{len(recs)}")
    # correlation: does len reduction predict LCS drop?
    import statistics as st
    if len(recs) > 2:
        xs = [r["len0"] - r["len2"] for r in recs]
        ys = [r["delta"] for r in recs]
        mx, my = mean(xs), mean(ys)
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
        corr = cov / ((vx*vy)**0.5) if vx > 0 and vy > 0 else 0
        print(f"  corr(len_reduction, LCS_delta) = {corr:+.3f}  (negative => skipping shortens chain & drops LCS)")

    print("\n=== 3 worst-hurt scenarios ===")
    for r in sorted(recs, key=lambda r: r["delta"])[:3]:
        print(f"\n--- {r['stem']}  LCS {r['lcs0']:.2f}->{r['lcs2']:.2f} (d{r['delta']:+.2f}); "
              f"len {r['len0']}->{r['len2']}; skips={r['nskips2']} ---")
        ref_tids = set()
        for st_ in r["flow"]:
            ref_tids.add(st_.get("tid") or st_.get("technique_id"))
            for a in (st_.get("alts") or []): ref_tids.add(a)
        seq0 = [(n.get("group_idx"), n.get("technique_id")) for n in r["bd0"]]
        seq2 = [(n.get("group_idx"), n.get("technique_id")) for n in r["bd2"]]
        gi2 = {n.get("group_idx") for n in r["bd2"]}
        dropped = [(gi, tid) for gi, tid in seq0 if gi not in gi2]
        dropped_ref = [(gi, tid) for gi, tid in dropped if tid in ref_tids]
        print(f"   reference TIDs: {sorted(ref_tids)}")
        print(f"   D0 chain TIDs: {[t for _,t in seq0]}")
        print(f"   D2 chain TIDs: {[t for _,t in seq2]}")
        print(f"   groups in D0 but skipped in D2: {len(dropped)}; of which REFERENCE techniques: {len(dropped_ref)} -> {dropped_ref[:6]}")


if __name__ == "__main__":
    main()
