"""
Re-run Viterbi only using cached ttp_mapping + features. Skip LLM/FAISS stages.

"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.attack_chain import (
    sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer,
    MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi,
)
from pipeline.technique_io import load_or_build_technique_io
import pandas as pd


def run_one(dataset_path: Path):
    config.configure_dataset(dataset_path)
    ttp_fp = config.TTP_MAPPING_JSON_PATH
    feat_fp = config.FEATURE_RESULT_JSON_PATH
    fcsv_fp = config.FINALE_CSV_PATH
    if not (ttp_fp.exists() and feat_fp.exists() and fcsv_fp.exists()):
        print(f"  [skip] {config.DATASET_NAME}: missing cached files")
        return False

    with open(ttp_fp, encoding="utf-8") as f:
        ttp_results = json.load(f)
    with open(feat_fp, encoding="utf-8") as f:
        all_features = json.load(f)
    final_df = pd.read_csv(fcsv_fp)
    final_df["TimeCreated"] = pd.to_datetime(final_df["TimeCreated"], errors="coerce")

    sorted_results = sort_results_by_time(ttp_results, final_df)

    # (A1) Family-consensus rerank on cached similar_techniques.
    fam_boost = float(getattr(config, "FAMILY_BOOST", 0.0))
    fam_width = int(getattr(config, "FAMILY_BOOST_WIDTH", 10))
    if fam_boost > 0 and fam_width > 0:
        from collections import Counter as _C
        for r in sorted_results:
            cands = r.get("similar_techniques", [])
            if len(cands) < 2:
                continue
            wide = min(fam_width, len(cands))
            parents = [c["technique_id"].split(".", 1)[0] for c in cands[:wide]]
            pcount = _C(parents)
            # multiplicative boost on p_ttp by family consensus count (excluding self).
            scored = []
            for i, c in enumerate(cands[:wide]):
                p = parents[i]
                shared = pcount[p] - 1
                boosted = c.get("p_ttp", c.get("similarity", 0)) * (1.0 + fam_boost * shared)
                scored.append((boosted, i, c))
            scored.sort(key=lambda x: -x[0])
            reranked = [c for _, _, c in scored] + cands[wide:]
            for new_rank, c in enumerate(reranked, 1):
                c["rank"] = new_rank
            r["similar_techniques"] = reranked
            if len(reranked) >= 2:
                r["confidence_margin"] = float(
                    reranked[0].get("p_ttp", 0) - reranked[1].get("p_ttp", 0)
                )

    # Confidence gate
    min_sim = getattr(config, "VITERBI_MIN_SIM_GATE", 0.0)
    max_after = getattr(config, "VITERBI_MAX_GROUPS_AFTER_GATE", 0)
    if min_sim > 0:
        n_before = len(sorted_results)
        filtered = []
        for r in sorted_results:
            cands = r.get("similar_techniques", [])
            top1 = float(cands[0].get("similarity", 0)) if cands else 0.0
            if top1 >= min_sim:
                filtered.append(r)
        if max_after > 0 and len(filtered) > max_after:
            scored = [(i, float(r.get("similar_techniques",[{}])[0].get("similarity",0)))
                      for i, r in enumerate(filtered)]
            scored.sort(key=lambda x: -x[1])
            keep_idx = sorted(i for i, _ in scored[:max_after])
            filtered = [filtered[i] for i in keep_idx]
        print(f"  gate: {n_before} -> {len(filtered)}")
        sorted_results = filtered

    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    features_by_gid = {f["group_id"]: f for f in all_features}
    group_nodes = build_group_nodes(sorted_results, tactic_map, features_by_gid)

    if not group_nodes:
        print(f"  [empty] {config.DATASET_NAME}")
        return False

    tac = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    sem = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
        calibration=getattr(config, "SEM_CALIBRATION", "linear"),
        sigmoid_center=getattr(config, "SEM_SIGMOID_CENTER", 0.5),
        sigmoid_scale=getattr(config, "SEM_SIGMOID_SCALE", 8.0),
    ) if config.USE_SEMANTIC_SCORING else None
    cau = None
    if config.USE_CAUSAL_SCORING:
        technique_io = load_or_build_technique_io(
            str(config.MITRE_CSV_PATH),
            cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
        )
        cau = CausalScorer(technique_io=technique_io)

    multi = MultiDimTransitionScorer(
        tac_scorer=tac, sem_scorer=sem, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0),
    )
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
    vit = topk_viterbi(
        group_nodes, multi,
        beam_k=config.VITERBI_BEAM_K,
        max_skip=config.VITERBI_MAX_SKIP,
        skip_penalty=config.VITERBI_SKIP_PENALTY,
        transition_weight=config.VITERBI_TRANSITION_WEIGHT,
        campaigns=campaigns,
        margin_gated=getattr(config, "VITERBI_MARGIN_GATED_ALPHA", False),
        margin_low=getattr(config, "VITERBI_MARGIN_LOW", 0.05),
        margin_high=getattr(config, "VITERBI_MARGIN_HIGH", 0.20),
        alpha_low_margin=getattr(config, "VITERBI_ALPHA_LOW_MARGIN", 0.40),
        alpha_high_margin=getattr(config, "VITERBI_ALPHA_HIGH_MARGIN", 0.05),
        sim_gated=getattr(config, "VITERBI_SIM_GATED_ALPHA", False),
        sim_margin_low=getattr(config, "VITERBI_SIM_MARGIN_LOW", 0.03),
        sim_margin_high=getattr(config, "VITERBI_SIM_MARGIN_HIGH", 0.10),
        alpha_low_sim=getattr(config, "VITERBI_ALPHA_LOW_SIM", 0.5),
        alpha_high_sim=getattr(config, "VITERBI_ALPHA_HIGH_SIM", 0.1),
        hard_tactic_filter=getattr(config, "VITERBI_HARD_TACTIC_FILTER", False),
    )

    # v20: emission-confidence bypass (X+Z hybrid).
    bypass_thr = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)
    if bypass_thr is not None:
        from pipeline.attack_chain import apply_emission_confidence_bypass
        vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass_thr))

    with open(config.VITERBI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(vit.score_breakdown, f, ensure_ascii=False, indent=2)
    print(f"  {config.DATASET_NAME}: chain len={len(vit.score_breakdown)}")
    return True


def main():
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    ok = 0
    for i, ds in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {ds.relative_to(config.DATASET_FOLDER)}")
        try:
            if run_one(ds):
                ok += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"\nDone: {ok}/{len(datasets)} Viterbi recomputed")


if __name__ == "__main__":
    main()
