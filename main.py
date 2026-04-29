"""
EDR Agent Prototype -- End-to-End Pipeline


    cd "Final Code"
    export GEMINI_API_KEY=...
    python main.py
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

import config
from pipeline.data_loader       import load_and_normalize
from pipeline.rule_matcher      import (
    load_rules, run_grouping, merge_same_anchor,
    merge_shared_supporting, print_groups,
)
from pipeline.feature_extractor import extract_all
from pipeline.feature_sanitizer import sanitize
from pipeline.mitre_mapper      import analyze
from pipeline.attack_chain      import (
    sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer,
    MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi, print_viterbi_report,
)
from pipeline.technique_io      import load_or_build_technique_io
from pipeline.annotation        import generate_template
from pipeline.evaluator         import (
    load_ground_truth, evaluate_ttp_mapping, evaluate_tactic_chain,
    aggregate_report, print_report,
)


def run_pipeline():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if (config.TTP_MAPPING_JSON_PATH.exists() and
            config.VITERBI_JSON_PATH.exists()):
        print(f"  [skip] ...: {config.DATASET_NAME}")
        return

    print("\n" + "═" * 75)
    print("  [1/5] ...")
    print("═" * 75)
    final_df = load_and_normalize(str(config.DATASET_FILE))
    final_df.to_csv(config.FINALE_CSV_PATH, index=False)
    print(f"\n  ...: {config.FINALE_CSV_PATH}")

    print("\n" + "═" * 75)
    print("  [2/5] Rule ...")
    print("═" * 75)
    rule_list = load_rules(config.RULE_FOLDER)
    groups = run_grouping(
        df                   = final_df,
        rule_list            = rule_list,
        before_sec           = config.GROUPING_BEFORE_SEC,
        after_sec            = config.GROUPING_AFTER_SEC,
        hop_up               = config.GROUPING_HOP_UP,
        hop_down             = config.GROUPING_HOP_DOWN,
        apply_filters        = config.GROUPING_APPLY_FILTER,
        use_shared_entity    = config.GROUPING_USE_SHARED_ENTITY,
        max_anchors_per_rule = config.GROUPING_MAX_ANCHORS_PER_RULE,
    )
    n_before = len(groups)
    groups = merge_same_anchor(groups)
    print(f"  same-anchor ...: {n_before} → {len(groups)}")

    groups = merge_shared_supporting(
        groups, final_df,
        overlap_threshold=config.MERGE_OVERLAP_THRESHOLD,
    )

    if getattr(config, "DROP_FILTER_FAILED_GROUPS", False):
        n_before = len(groups)
        groups = [g for g in groups if g.get("filter_passed", True)]
        if len(groups) < n_before:
            print(f"  filter_passed=False ...: {n_before} → {len(groups)}")

    if len(groups) > config.MAX_GROUPS_PER_SCENARIO:
        by_tid: dict[str, list] = defaultdict(list)
        for g in groups:
            by_tid[g["technique_id"]].append(g)
        for tid in by_tid:
            by_tid[tid].sort(key=lambda g: g.get("confidence", 0), reverse=True)

        truncated: list = []
        i = 0
        while len(truncated) < config.MAX_GROUPS_PER_SCENARIO:
            added = False
            for tid, glist in by_tid.items():
                if i < len(glist):
                    truncated.append(glist[i])
                    added = True
                    if len(truncated) >= config.MAX_GROUPS_PER_SCENARIO:
                        break
            if not added:
                break
            i += 1
        print(f"\n  ⚠ ...{len(groups)} → {len(truncated)}..."
              f"(cap={config.MAX_GROUPS_PER_SCENARIO})")
        groups = truncated

    print(f"\n  ...: {len(groups)}")
    print_groups(groups)

    generate_template(
        groups, final_df,
        config.ANNOTATION_JSON_PATH,
        scenario_name=config.DATASET_NAME,
    )

    print("\n" + "═" * 75)
    print("  [3/5] Feature ...+ ...")
    print("═" * 75)
    all_features = extract_all(groups, final_df)
    with open(config.FEATURE_RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    print(f"  ...: {config.FEATURE_RESULT_JSON_PATH}")

    all_features_sanitized = [sanitize(f) for f in all_features]

    print("\n" + "═" * 75)
    print("  [4/5] LLM description + FAISS MITRE ...")
    print("═" * 75)
    if not config.GEMINI_API_KEY:
        print("  ⚠ GEMINI_API_KEY ...-- 4...")
        return

    cap = config.SAMPLE_PER_TECHNIQUE
    if cap <= 0:
        sampled = list(all_features_sanitized)
    else:
        sampled = []
        count_by_tid: dict[str, int] = defaultdict(int)
        for f in all_features_sanitized:
            tid = f["technique_id"]
            if count_by_tid[tid] < cap:
                sampled.append(f)
                count_by_tid[tid] += 1
    print(f"  ...{len(all_features_sanitized)}...{len(sampled)}...cap={cap})")

    ce_for_rerank = None
    if getattr(config, "USE_CE_EMISSION_RERANK", True):
        sem_scorer_for_ce = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
    )
        ce_for_rerank = getattr(sem_scorer_for_ce, "_model", None)
        if ce_for_rerank is None:
            print("  [warn] CE ...-- emission rerank ...")

    results = analyze(
        sampled,
        str(config.MITRE_CSV_PATH),
        config.GEMINI_API_KEY,
        cache_dir=config.CACHE_DIR,
        cross_encoder=ce_for_rerank,
        ce_rerank_width=getattr(config, "CE_RERANK_WIDTH", 20),
        ce_weight=getattr(config, "CE_RERANK_WEIGHT", 0.0),
        bm25_weight=getattr(config, "BM25_WEIGHT", 0.3),
        bm25_rerank_width=getattr(config, "BM25_RERANK_WIDTH", 30),
        tid_prior=getattr(config, "RULE_TID_PRIOR", 1.15),
        tactic_prior=getattr(config, "RULE_TACTIC_PRIOR", 1.05),
        signature_weight=getattr(config, "SIGNATURE_WEIGHT", 0.0),
        signature_rerank_width=getattr(config, "SIGNATURE_RERANK_WIDTH", 10),
        family_boost=getattr(config, "FAMILY_BOOST", 0.0),
        family_boost_width=getattr(config, "FAMILY_BOOST_WIDTH", 10),
    )
    with open(config.TTP_MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  ...: {config.TTP_MAPPING_JSON_PATH}")

    # 5. Top-K Viterbi + Hole-Bridging ─────────────────────────────────────
    print("\n" + "═" * 75)
    print("  [5/5] Attack Chain (Top-K Viterbi + Hole-Bridging)")
    print("═" * 75)
    sorted_results = sort_results_by_time(results, final_df)

    # ── Confidence gate ───────────────────────────────────────────────────
    min_sim_gate = getattr(config, "VITERBI_MIN_SIM_GATE", 0.0)
    max_after_gate = getattr(config, "VITERBI_MAX_GROUPS_AFTER_GATE", 0)
    if min_sim_gate > 0:
        n_before = len(sorted_results)
        filtered = []
        for r in sorted_results:
            cands = r.get("similar_techniques", [])
            top1_sim = float(cands[0].get("similarity", 0)) if cands else 0.0
            if top1_sim >= min_sim_gate:
                filtered.append(r)
        print(f"  Confidence gate (sim>={min_sim_gate}): {n_before} → {len(filtered)}")
        if max_after_gate > 0 and len(filtered) > max_after_gate:
            scored = [(i, float(r.get("similar_techniques",[{}])[0].get("similarity",0)))
                      for i, r in enumerate(filtered)]
            scored.sort(key=lambda x: -x[1])
            keep_idx = sorted(i for i, _ in scored[:max_after_gate])
            filtered = [filtered[i] for i in keep_idx]
            print(f"  Hard cap after gate (top-{max_after_gate}): → {len(filtered)}")
        sorted_results = filtered
    tactic_map     = load_tactic_map(str(config.MITRE_CSV_PATH))

    features_by_gid = {f["group_id"]: f for f in all_features}
    group_nodes = build_group_nodes(sorted_results, tactic_map, features_by_gid)

    tac_scorer = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)

    sem_scorer = None
    if config.USE_SEMANTIC_SCORING:
        sem_scorer = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
    )

    cau_scorer = None
    if config.USE_CAUSAL_SCORING:
        technique_io = load_or_build_technique_io(
            str(config.MITRE_CSV_PATH),
            cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
        )
        cau_scorer = CausalScorer(technique_io=technique_io)

    multi_scorer = MultiDimTransitionScorer(
        tac_scorer=tac_scorer,
        sem_scorer=sem_scorer,
        cau_scorer=cau_scorer,
        w_tac=config.W_TAC,
        w_sem=config.W_SEM,
        w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0),
    )

    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    viterbi_result = topk_viterbi(
        group_nodes, multi_scorer,
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
    _bypass_thr = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)
    if _bypass_thr is not None:
        from pipeline.attack_chain import apply_emission_confidence_bypass
        viterbi_result = apply_emission_confidence_bypass(
            viterbi_result, multi_scorer, sim_threshold=float(_bypass_thr)
        )

    print_viterbi_report(viterbi_result)

    with open(config.VITERBI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(viterbi_result.score_breakdown, f, ensure_ascii=False, indent=2)
    print(f"  ...: {config.VITERBI_JSON_PATH}")


def run_all() -> None:
    """Iterate every *.json scenario under DATASET_FOLDER and run the
    end-to-end pipeline. Failures are collected and reported at the end."""
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    if not datasets:
        print(f"...: {config.DATASET_FOLDER}")
        return

    total = len(datasets)
    failed: list[tuple[str, str]] = []

    for i, ds_path in enumerate(datasets, start=1):
        rel_label = ds_path.relative_to(config.DATASET_FOLDER).as_posix()
        print("\n" + "#" * 75)
        print(f"#  [{i}/{total}] DATASET: {rel_label}")
        print("#" * 75)

        config.configure_dataset(ds_path)
        try:
            run_pipeline()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"\n  ✗ '{rel_label}' ...-- {err}")
            traceback.print_exc()
            failed.append((rel_label, err))

    print("\n" + "═" * 75)
    print("  ...")
    print("═" * 75)
    print(f"  ...{total}...{total - len(failed)}...{len(failed)}...")
    if failed:
        print("\n  ...:")
        for name, err in failed:
            print(f"    - {name}  ({err})")


def run_evaluate() -> None:
    """...annotation + ..."""
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    if not datasets:
        print(f"...: {config.DATASET_FOLDER}")
        return

    eval_results: list[dict] = []

    for ds_path in datasets:
        config.configure_dataset(ds_path)
        ann_path = config.ANNOTATION_JSON_PATH
        ttp_path = config.TTP_MAPPING_JSON_PATH
        vit_path = config.VITERBI_JSON_PATH

        if not ann_path.exists():
            continue

        gt = load_ground_truth(ann_path)
        if not gt:
            print(f"  [SKIP] {config.DATASET_NAME}: annotation ...")
            continue

        scenario_eval = {"scenario": config.DATASET_NAME}

        if ttp_path.exists():
            with open(ttp_path, "r", encoding="utf-8") as f:
                ttp_results = json.load(f)
            scenario_eval["ttp"] = evaluate_ttp_mapping(gt, ttp_results)

        if vit_path.exists():
            with open(vit_path, "r", encoding="utf-8") as f:
                viterbi_breakdown = json.load(f)
            scenario_eval["chain"] = evaluate_tactic_chain(gt, viterbi_breakdown)

        eval_results.append(scenario_eval)

        with open(config.EVAL_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(scenario_eval, f, ensure_ascii=False, indent=2)
        print(f"  {config.DATASET_NAME}: ...{config.EVAL_JSON_PATH}")

    if eval_results:
        agg = aggregate_report(
            eval_results,
            output_path=config.OUTPUT_BASE_DIR / "aggregate_eval.json",
        )
        print_report(agg)
    else:
        print("...annotation ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDR Agent Pipeline")
    parser.add_argument(
        "mode", nargs="?", default="run",
        choices=["run", "evaluate"],
        help="run: ...evaluate: ...GT...",
    )
    args = parser.parse_args()

    if args.mode == "evaluate":
        run_evaluate()
    else:
        run_all()
