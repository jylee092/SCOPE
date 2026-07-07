"""M_tac perturbation sensitivity.

Perturbs EVERY tactical-matrix value (the rule weights + self-loop/wildcard-in
overrides that materialize the 14x14 M_tac) by multiplicative noise +-eps and
re-runs the deterministic Viterbi over cached emissions (no LLM, no re-map).
Measures technique-LCS stability -> defends "results are not knife-edge tuned
to hand-picked matrix values".

Group nodes are built ONCE per scenario (CSV read once); N trials reuse them.
"""
from __future__ import annotations
import json, sys, random, copy
from pathlib import Path
from statistics import mean, pstdev
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config, pandas as pd
from pipeline.attack_chain import (sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, CausalScorer, MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi, apply_emission_confidence_bypass)
from pipeline.technique_io import load_or_build_technique_io
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment

tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json"))
bypass = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)

# ---- pass 1: build group nodes once ----
scenarios = []
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists()
            and config.FINALE_CSV_PATH.exists()):
        continue
    flow = get_flow(ds.stem)
    if not flow:
        continue
    ttp = json.load(open(config.TTP_MAPPING_JSON_PATH, encoding="utf-8"))
    ft = json.load(open(config.FEATURE_RESULT_JSON_PATH, encoding="utf-8"))
    fdf = pd.read_csv(config.FINALE_CSV_PATH, low_memory=False)
    fdf["TimeCreated"] = pd.to_datetime(fdf["TimeCreated"], errors="coerce")
    gn = build_group_nodes(sort_results_by_time(ttp, fdf), tm, {f["group_id"]: f for f in ft})
    if gn:
        scenarios.append((ds.stem, gn, flow))
print(f"[mtac-pert] {len(scenarios)} scenarios cached in memory")


def make_scorer(eps: float, rng: random.Random | None):
    """TacticalScorer whose rule weights + overrides are jittered by +-eps."""
    s = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    if rng is None:
        return s
    def jit(w):
        return max(0.0, min(1.0, w * (1.0 + rng.uniform(-eps, eps))))
    s._rules = {rid: {**r, "weight": jit(r["weight"])} for rid, r in s._rules.items()}
    s._self_loop_w = {k: jit(v) for k, v in s._self_loop_w.items()}
    s._wildcard_in_w = {k: jit(v) for k, v in s._wildcard_in_w.items()}
    s._overrides = {k: (rid, jit(w), note) for k, (rid, w, note) in s._overrides.items()}
    return s


def tech_lcs_macro(tac_scorer) -> float:
    vals = []
    multi = MultiDimTransitionScorer(tac_scorer=tac_scorer, sem_scorer=None, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0))
    for stem, gn, flow in scenarios:
        vit = topk_viterbi(gn, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
            skip_penalty=config.VITERBI_SKIP_PENALTY,
            transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
        if bypass is not None:
            vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
        bd = vit.score_breakdown
        vals.append(evaluate_chain_alignment(stem, bd, ref_flow=flow).get("technique_lcs_norm", 0) if bd else 0)
    return mean(vals)


base = tech_lcs_macro(make_scorer(0.0, None))
print(f"baseline technique-LCS: {base:.4f}")
for eps in (0.10, 0.20, 0.30):
    trials = [tech_lcs_macro(make_scorer(eps, random.Random(1000 + i))) for i in range(20)]
    print(f"eps=+-{eps:.0%}: mean {mean(trials):.4f}  std {pstdev(trials):.4f}  "
          f"min {min(trials):.4f}  max {max(trials):.4f}  (n=20)")
