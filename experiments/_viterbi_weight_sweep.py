"""
Transition weight + self-loop penalty sweep. Re-runs Viterbi per config,
then evaluates per-group post-Viterbi plausibility.
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
    MultiDimTransitionScorer, load_campaign_library, topk_viterbi,
)
from pipeline.technique_io import load_or_build_technique_io
from pipeline.evaluator import load_ground_truth
from experiments.run_eval import load_tactic_map as _tm, patch_candidate_tactics
from experiments.attack_flows import get_flow, all_acceptable_tids
from experiments.run_eval_plausible import _is_strong_tp, tid_family_match
import pandas as pd


def rerun_viterbi(trans_w: float, self_loop: float, max_skip: int = 0):
    """Re-run Viterbi for all scenarios with given weights. In-memory; no disk write."""
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    technique_io = load_or_build_technique_io(
        str(config.MITRE_CSV_PATH),
        cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
    )
    sem = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
        calibration=getattr(config, "SEM_CALIBRATION", "linear"),
        sigmoid_center=getattr(config, "SEM_SIGMOID_CENTER", 0.5),
        sigmoid_scale=getattr(config, "SEM_SIGMOID_SCALE", 8.0),
    ) if config.USE_SEMANTIC_SCORING else None
    tac = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    cau = CausalScorer(technique_io=technique_io) if config.USE_CAUSAL_SCORING else None
    multi = MultiDimTransitionScorer(
        tac_scorer=tac, sem_scorer=sem, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
        self_loop_tid_penalty=self_loop,
    )
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    scenario_viterbi = {}  # scenario_name -> list of step breakdowns

    for ds in datasets:
        config.configure_dataset(ds)
        ttp_fp = config.TTP_MAPPING_JSON_PATH
        feat_fp = config.FEATURE_RESULT_JSON_PATH
        fcsv_fp = config.FINALE_CSV_PATH
        if not (ttp_fp.exists() and feat_fp.exists() and fcsv_fp.exists()):
            continue
        with open(ttp_fp, encoding="utf-8") as f:
            ttp = json.load(f)
        with open(feat_fp, encoding="utf-8") as f:
            feat = json.load(f)
        df = pd.read_csv(fcsv_fp)
        df["TimeCreated"] = pd.to_datetime(df["TimeCreated"], errors="coerce")

        sorted_res = sort_results_by_time(ttp, df)
        features_by_gid = {f["group_id"]: f for f in feat}
        nodes = build_group_nodes(sorted_res, tactic_map, features_by_gid)
        if not nodes:
            continue
        vit = topk_viterbi(
            nodes, multi,
            beam_k=config.VITERBI_BEAM_K,
            max_skip=max_skip,
            skip_penalty=config.VITERBI_SKIP_PENALTY,
            transition_weight=trans_w,
            campaigns=campaigns,
            sim_gated=getattr(config, "VITERBI_SIM_GATED_ALPHA", False),
            sim_margin_low=getattr(config, "VITERBI_SIM_MARGIN_LOW", 0.03),
            sim_margin_high=getattr(config, "VITERBI_SIM_MARGIN_HIGH", 0.10),
            alpha_low_sim=getattr(config, "VITERBI_ALPHA_LOW_SIM", 0.5),
            alpha_high_sim=getattr(config, "VITERBI_ALPHA_HIGH_SIM", 0.1),
        )
        scenario_viterbi[config.DATASET_NAME] = vit.score_breakdown
    return scenario_viterbi


def evaluate(scenario_viterbi: dict, strong_only: bool):
    tm = _tm(config.MITRE_CSV_PATH)
    n_total = n_faiss = n_viterbi = 0
    imp = reg = 0

    for ann in sorted(config.OUTPUT_BASE_DIR.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt: continue
        with open(ann, encoding="utf-8") as f: ad = json.load(f)
        scenario = ad.get("scenario", ann.parent.name)
        flow = get_flow(scenario)
        if not flow: continue
        acc = set(all_acceptable_tids(flow))
        for a in list(acc):
            acc.add(a.split(".")[0])

        stem = ann.name.replace("_annotation.json","")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        if not ttp_fp.exists(): continue
        with open(ttp_fp, encoding="utf-8") as f: ttp = json.load(f)
        patch_candidate_tactics(ttp, tm)

        vit = scenario_viterbi.get(scenario, [])
        vit_by_gid = {b["group_id"]: b["technique_id"] for b in vit}

        def match(p): return any(tid_family_match(p, a) for a in acc) if p else False

        ttp_by_gid = {r["group_id"]: r for r in ttp}
        for gid, truth in gt.items():
            if not truth["is_tp"]: continue
            if strong_only and not _is_strong_tp(truth.get("notes","")): continue
            r = ttp_by_gid.get(gid)
            if not r: continue
            cands = r.get("similar_techniques", [])[:5]
            if not cands: continue
            top1 = cands[0]["technique_id"]
            vpick = vit_by_gid.get(gid, "")
            n_total += 1
            fh = match(top1); vh = match(vpick)
            if fh: n_faiss += 1
            if vh: n_viterbi += 1
            if vh and not fh: imp += 1
            if fh and not vh: reg += 1

    return {
        "n": n_total,
        "faiss_plausible": n_faiss/n_total if n_total else 0,
        "viterbi_plausible": n_viterbi/n_total if n_total else 0,
        "delta": (n_viterbi - n_faiss)/n_total if n_total else 0,
        "improved": imp,
        "regressed": reg,
    }


def main():
    combos = [
        (0.0, 1.0, 0, "pure_emission (FAISS top-1 baseline)"),
        (0.1, 1.0, 0, "trans=0.1 no-selfloop"),
        (0.2, 1.0, 0, "trans=0.2 no-selfloop"),
        (0.3, 1.0, 0, "trans=0.3 no-selfloop"),
        (0.5, 1.0, 0, "trans=0.5 no-selfloop"),
        (0.3, 0.3, 0, "trans=0.3 selfloop=0.3 (current)"),
        (0.5, 0.3, 0, "trans=0.5 selfloop=0.3 (current v5)"),
    ]

    print(f"{'config':<45s} {'n':>5s} {'FAISS':>8s} {'Vit':>8s} {'Δ':>7s} {'+imp':>5s} {'-reg':>5s}")
    print("-" * 90)
    for trans_w, sl, skip, label in combos:
        sv = rerun_viterbi(trans_w, sl, skip)
        for strong in (False, True):
            r = evaluate(sv, strong_only=strong)
            suffix = " [strong]" if strong else " [all]"
            print(f"{label + suffix:<45s} {r['n']:>5d} {r['faiss_plausible']:>8.3f} "
                  f"{r['viterbi_plausible']:>8.3f} {r['delta']:>+7.3f} "
                  f"{r['improved']:>5d} {r['regressed']:>5d}")


if __name__ == "__main__":
    main()
