"""
Skip-consistency check (read-only, no API, no canonical-output overwrite).

Question: does the MAIN-comparison technique-LCS (reported ~0.68, run with the
config default VITERBI_MAX_SKIP=0) change if hole-bridging is engaged
(VITERBI_MAX_SKIP=2) on the *complete* (undropped) data?

If 0.68 is preserved at D=2, the per-experiment skip setting is cosmetic (no
holes to bridge on complete logs) and we can unify the whole paper at D=2.
If it changes materially, that must be disclosed.

This reuses the cached ttp_mapping + features (same inputs as the canonical
main run) and the exact attack_chain Viterbi + scoring. It writes NOTHING to
canonical paths -- everything stays in memory.
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

SKIPS = [0, 2]


def _prep_results(ttp_results, final_df):
    sorted_results = sort_results_by_time(ttp_results, final_df)
    fam_boost = float(getattr(config, "FAMILY_BOOST", 0.0))
    fam_width = int(getattr(config, "FAMILY_BOOST_WIDTH", 10))
    if fam_boost > 0 and fam_width > 0:
        for r in sorted_results:
            cands = r.get("similar_techniques", [])
            if len(cands) < 2:
                continue
            wide = min(fam_width, len(cands))
            parents = [c["technique_id"].split(".", 1)[0] for c in cands[:wide]]
            pcount = _C(parents)
            scored = []
            for i, c in enumerate(cands[:wide]):
                shared = pcount[parents[i]] - 1
                boosted = c.get("p_ttp", c.get("similarity", 0)) * (1.0 + fam_boost * shared)
                scored.append((boosted, i, c))
            scored.sort(key=lambda x: -x[0])
            reranked = [c for _, _, c in scored] + cands[wide:]
            for nr, c in enumerate(reranked, 1):
                c["rank"] = nr
            r["similar_techniques"] = reranked
            if len(reranked) >= 2:
                r["confidence_margin"] = float(
                    reranked[0].get("p_ttp", 0) - reranked[1].get("p_ttp", 0))
    return sorted_results


def main():
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    sem = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
        calibration=getattr(config, "SEM_CALIBRATION", "linear"),
    ) if config.USE_SEMANTIC_SCORING else None
    cau = None
    if config.USE_CAUSAL_SCORING:
        technique_io = load_or_build_technique_io(
            str(config.MITRE_CSV_PATH),
            cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json")
        cau = CausalScorer(technique_io=technique_io)
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
    bypass_thr = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)

    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    per_skip = {d: [] for d in SKIPS}
    n = 0
    for ds in datasets:
        config.configure_dataset(ds)
        ttp_fp, feat_fp, fcsv_fp = (config.TTP_MAPPING_JSON_PATH,
                                    config.FEATURE_RESULT_JSON_PATH,
                                    config.FINALE_CSV_PATH)
        if not (ttp_fp.exists() and feat_fp.exists() and fcsv_fp.exists()):
            continue
        stem = ds.stem
        flow = get_flow(stem)
        if not flow:
            continue
        ttp_results = json.load(open(ttp_fp, encoding="utf-8"))
        all_features = json.load(open(feat_fp, encoding="utf-8"))
        final_df = pd.read_csv(fcsv_fp)
        final_df["TimeCreated"] = pd.to_datetime(final_df["TimeCreated"], errors="coerce")
        sorted_results = _prep_results(ttp_results, final_df)
        features_by_gid = {f["group_id"]: f for f in all_features}
        group_nodes = build_group_nodes(sorted_results, tactic_map, features_by_gid)
        if not group_nodes:
            for d in SKIPS:
                per_skip[d].append(0.0)
            n += 1
            continue
        tac = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
        multi = MultiDimTransitionScorer(
            tac_scorer=tac, sem_scorer=sem, cau_scorer=cau,
            w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
            self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0))
        for d in SKIPS:
            vit = topk_viterbi(
                group_nodes, multi, beam_k=config.VITERBI_BEAM_K,
                max_skip=d, skip_penalty=config.VITERBI_SKIP_PENALTY,
                transition_weight=config.VITERBI_TRANSITION_WEIGHT,
                campaigns=campaigns)
            if bypass_thr is not None:
                vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass_thr))
            bd = vit.score_breakdown
            lcs = evaluate_chain_alignment(stem, bd, ref_flow=flow).get("technique_lcs_norm", 0.0) if bd else 0.0
            per_skip[d].append(lcs)
        n += 1

    print(f"\n=== skip-consistency on {n} scenarios (main/complete data) ===")
    for d in SKIPS:
        vals = per_skip[d]
        print(f"  VITERBI_MAX_SKIP={d}: mean technique-LCS = {mean(vals):.4f}  (n={len(vals)})")
    if all(per_skip[d] for d in SKIPS):
        diff = mean(per_skip[2]) - mean(per_skip[0])
        print(f"  delta (D=2 - D=0) = {diff:+.4f}")


if __name__ == "__main__":
    main()
