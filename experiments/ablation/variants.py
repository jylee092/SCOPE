"""
Ablation 변종 실행 함수들.

각 변종은 main.py의 파이프라인을 재구성하지 않고 pipeline/* 모듈을 재조합하여
독립적으로 실행한다. 출력은 variant별 subfolder로 분리되어 full과 충돌하지 않음.

Variants:
  full         : 전체 프레임워크 (baseline) — 참고용, 실제로는 main.py로 실행 권장
  no_grouping  : 룰 우회, 단일 이벤트 그룹 → LLM → FAISS → Viterbi
  no_llm       : LLM description 생략, feature_to_text로 FAISS 쿼리
  top1_only    : Viterbi beam_k=1, max_skip=0 (greedy)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Final_Code/를 sys.path에 추가
_FINAL_CODE = Path(__file__).resolve().parent.parent.parent
if str(_FINAL_CODE) not in sys.path:
    sys.path.insert(0, str(_FINAL_CODE))

import config
from pipeline.data_loader       import load_and_normalize
from pipeline.rule_matcher      import load_rules, run_grouping, merge_shared_supporting
from pipeline.feature_extractor import extract_all
from pipeline.feature_sanitizer import sanitize
from pipeline.mitre_mapper      import analyze
from pipeline.attack_chain      import (
    sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer,
    MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi,
)
from pipeline.technique_io      import load_or_build_technique_io
from experiments.ablation.helpers import build_solo_groups


VARIANT_NAMES = ["full", "no_grouping", "no_llm", "top1_only", "no_shared_entity"]


def _variant_output_dir(variant: str, dataset_rel: Path) -> Path:
    """변종별 출력 폴더: output/ablation_<variant>/<scenario_path>/."""
    if variant == "full":
        return config.OUTPUT_BASE_DIR / dataset_rel
    return config.OUTPUT_BASE_DIR / f"ablation_{variant}" / dataset_rel


def _make_groups(variant: str, final_df, rule_list) -> list[dict]:
    """variant에 따라 그룹 생성 방식 분기."""
    if variant == "no_grouping":
        print("  [ablation] no_grouping — 단일 이벤트 그룹 생성")
        return build_solo_groups(final_df)

    use_shared = (variant != "no_shared_entity") and config.GROUPING_USE_SHARED_ENTITY
    if variant == "no_shared_entity":
        print("  [ablation] no_shared_entity — lineage-only grouping")

    groups = run_grouping(
        df                = final_df,
        rule_list         = rule_list,
        before_sec        = config.GROUPING_BEFORE_SEC,
        after_sec         = config.GROUPING_AFTER_SEC,
        hop_up            = config.GROUPING_HOP_UP,
        hop_down          = config.GROUPING_HOP_DOWN,
        apply_filters     = config.GROUPING_APPLY_FILTER,
        use_shared_entity = use_shared,
    )
    return merge_shared_supporting(
        groups, final_df,
        overlap_threshold=config.MERGE_OVERLAP_THRESHOLD,
    )


def run_variant_on_scenario(variant: str, dataset_path: Path) -> dict:
    """하나의 (variant, scenario) 조합을 실행하고 결과 경로들을 dict로 반환."""
    assert variant in VARIANT_NAMES

    rel = dataset_path.relative_to(config.DATASET_FOLDER).with_suffix("")
    stem = dataset_path.stem
    out_dir = _variant_output_dir(variant, rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    ttp_path     = out_dir / f"{stem}_ttp_mapping.json"
    viterbi_path = out_dir / f"{stem}_viterbi.json"

    print(f"\n{'─'*75}\n  VARIANT={variant}  DATASET={rel}\n{'─'*75}")

    # 1) Load
    final_df = load_and_normalize(str(dataset_path))

    # 2) Group
    rule_list = load_rules(config.RULE_FOLDER) if variant != "no_grouping" else []
    groups = _make_groups(variant, final_df, rule_list)
    print(f"  그룹 수: {len(groups)}")

    if not groups:
        print("  (그룹 없음 — 스킵)")
        return {"variant": variant, "scenario": stem, "groups": 0, "skipped": True}

    # 3) Feature
    all_features = extract_all(groups, final_df)
    all_features_sanitized = [sanitize(f) for f in all_features]

    # 4) TTP mapping
    if not config.GEMINI_API_KEY:
        print("  ⚠ GEMINI_API_KEY 미설정 — TTP 매핑 스킵")
        return {"variant": variant, "scenario": stem, "groups": len(groups), "skipped": True}

    # no_grouping 모드는 그룹 수가 많아 샘플 캡 무시
    cap = config.SAMPLE_PER_TECHNIQUE
    if cap > 0 and variant != "no_grouping":
        from collections import defaultdict
        sampled: list[dict] = []
        count_by_tid: dict[str, int] = defaultdict(int)
        for f in all_features_sanitized:
            tid = f["technique_id"]
            if count_by_tid[tid] < cap:
                sampled.append(f)
                count_by_tid[tid] += 1
    else:
        sampled = list(all_features_sanitized)

    # `no_grouping` is forced to use feature-text→FAISS instead of LLM
    # description: at solo-group cardinality (one anchor per event) the LLM
    # cost is prohibitive (≈30 K calls / scenario set). This conflates
    # "no grouping" with "no LLM polish"; we note this in the paper §7.5.
    use_llm = (variant not in ("no_llm", "no_grouping"))
    results = analyze(
        sampled,
        str(config.MITRE_CSV_PATH),
        config.GEMINI_API_KEY,
        cache_dir=config.CACHE_DIR,
        use_llm=use_llm,
    )
    with open(ttp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 5) Viterbi
    sorted_results = sort_results_by_time(results, final_df)
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    features_by_gid = {f["group_id"]: f for f in all_features}
    group_nodes = build_group_nodes(sorted_results, tactic_map, features_by_gid)

    tac_scorer = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    sem_scorer = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
        calibration=getattr(config, "SEM_CALIBRATION", "linear"),
        sigmoid_center=getattr(config, "SEM_SIGMOID_CENTER", 0.5),
        sigmoid_scale=getattr(config, "SEM_SIGMOID_SCALE", 8.0),
    ) if config.USE_SEMANTIC_SCORING else None
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
        w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
    )
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    # top1_only override
    beam_k   = 1 if variant == "top1_only" else config.VITERBI_BEAM_K
    max_skip = 0 if variant == "top1_only" else config.VITERBI_MAX_SKIP

    viterbi_result = topk_viterbi(
        group_nodes, multi_scorer,
        beam_k=beam_k,
        max_skip=max_skip,
        skip_penalty=config.VITERBI_SKIP_PENALTY,
        transition_weight=config.VITERBI_TRANSITION_WEIGHT,
        campaigns=campaigns,
    )
    with open(viterbi_path, "w", encoding="utf-8") as f:
        json.dump(viterbi_result.score_breakdown, f, ensure_ascii=False, indent=2)

    return {
        "variant": variant,
        "scenario": stem,
        "groups": len(groups),
        "chain_length": len(viterbi_result.best_path),
        "best_score": viterbi_result.best_score,
        "novelty": viterbi_result.novelty_score,
        "ttp_mapping": str(ttp_path),
        "viterbi": str(viterbi_path),
    }
