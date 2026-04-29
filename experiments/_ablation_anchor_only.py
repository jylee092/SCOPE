"""
"anchor_only" ablation -- replace each Full SCOPE behavior group with a
single-event group anchored at exactly that group's anchor row. Same group
count, same anchor selection, no supporting-event aggregation.

This isolates the contribution of supporting-event aggregation from
upstream factors (anchor selection, group count).

Output → output/ablation_anchor_only/<scenario>/<scenario>_viterbi.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from pipeline.data_loader       import load_and_normalize
from pipeline.feature_extractor import extract_all
from pipeline.feature_sanitizer import sanitize
from pipeline.mitre_mapper      import analyze
from pipeline.attack_chain      import (
    sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer,
    MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi, apply_emission_confidence_bypass,
)
from pipeline.technique_io      import load_or_build_technique_io
from experiments.ablation.helpers import build_solo_groups


def _output_dir(rel: Path) -> Path:
    return config.OUTPUT_BASE_DIR / "ablation_anchor_only" / rel


def run_one(dataset_path: Path) -> dict | None:
    rel = dataset_path.relative_to(config.DATASET_FOLDER).with_suffix("")
    stem = dataset_path.stem
    out_dir = _output_dir(rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load full SCOPE feature_result.json to get the anchor indices
    full_feat_path = (config.OUTPUT_BASE_DIR / rel
                      / f"{stem}_feature_result.json")
    if not full_feat_path.exists():
        print(f"  [skip] {stem}: no full feature_result.json")
        return None
    with open(full_feat_path, encoding="utf-8") as f:
        full_groups = json.load(f)
    if not isinstance(full_groups, list) or not full_groups:
        print(f"  [skip] {stem}: empty full feature_result")
        return None

    anchor_idxs = []
    for g in full_groups:
        a = g.get("anchor_idx")
        if a is None and g.get("all_idxs"):
            a = g["all_idxs"][0]
        if a is not None:
            anchor_idxs.append(int(a))
    if not anchor_idxs:
        print(f"  [skip] {stem}: no anchors in full feature_result")
        return None

    # 2) Load and normalize the scenario; rebuild solo groups at anchors
    final_df = load_and_normalize(str(dataset_path))
    groups = build_solo_groups(final_df, anchor_idxs=anchor_idxs)
    print(f"  [{stem}] anchor-only solo groups: {len(groups)} "
          f"(orig full groups={len(full_groups)})")
    if not groups:
        return None

    # 3) Feature extract → mapper (LLM ON, same as Full SCOPE)
    all_features = extract_all(groups, final_df)
    all_features_sanitized = [sanitize(f) for f in all_features]

    # No per-TID cap -- anchor count is already small
    sampled = all_features_sanitized

    results = analyze(
        sampled,
        str(config.MITRE_CSV_PATH),
        config.GEMINI_API_KEY,
        cache_dir=config.CACHE_DIR,
        use_llm=True,
    )
    ttp_path = out_dir / f"{stem}_ttp_mapping.json"
    with open(ttp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 4) Viterbi (same hyperparameters as Full SCOPE)
    sorted_results = sort_results_by_time(results, final_df)
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    features_by_gid = {f["group_id"]: f for f in all_features}
    group_nodes = build_group_nodes(sorted_results, tactic_map, features_by_gid)

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
    )
    bypass_thr = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)
    if bypass_thr is not None:
        vit = apply_emission_confidence_bypass(vit, multi,
                                                  sim_threshold=float(bypass_thr))

    vit_path = out_dir / f"{stem}_viterbi.json"
    with open(vit_path, "w", encoding="utf-8") as f:
        json.dump(vit.score_breakdown, f, ensure_ascii=False, indent=2)
    print(f"    chain_length={len(vit.score_breakdown)}")

    # also save feature_result.json so downstream tools can look up groups
    feat_out = out_dir / f"{stem}_feature_result.json"
    with open(feat_out, "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)

    return {"scenario": stem, "n_groups": len(groups),
            "chain_length": len(vit.score_breakdown)}


def main() -> None:
    paths = sorted(config.DATASET_FOLDER.rglob("*.json"))
    print(f"Anchor-only solo ablation on {len(paths)} scenarios ...")
    t0 = time.time()
    ok = 0
    for ds in paths:
        try:
            r = run_one(ds)
            if r:
                ok += 1
        except Exception as e:
            print(f"  [FAIL] {ds.stem}: {type(e).__name__}: {e}")
    print(f"\nCompleted: {ok}/{len(paths)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
