"""Does UNIFYING the self-loop weight (drop the Exec=0.4 / DefEvasion=0.6
distinction -> all self-loops = default 0.5) actually change results?
Reads cached emissions, re-runs deterministic Viterbi. No LLM.
Reports ALL four chain metrics, not just technique-LCS.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean
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

scen = []
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
        scen.append((ds.stem, gn, flow))
print(f"{len(scen)} scenarios")


def metrics(unify_selfloop: bool):
    s = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    if unify_selfloop:
        s._self_loop_w = {}          # all self-loops fall back to default R1 = 0.5
    multi = MultiDimTransitionScorer(tac_scorer=s, sem_scorer=None, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0))
    te, ta, st, od = [], [], [], []
    for stem, gn, flow in scen:
        vit = topk_viterbi(gn, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
            skip_penalty=config.VITERBI_SKIP_PENALTY,
            transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
        if bypass is not None:
            vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
        bd = vit.score_breakdown
        r = evaluate_chain_alignment(stem, bd, ref_flow=flow) if bd else {}
        te.append(r.get("technique_lcs_norm", 0)); ta.append(r.get("tactic_lcs_norm", 0))
        st.append(r.get("step_coverage", 0)); od.append(r.get("order_accuracy", 0))
    return mean(te), mean(ta), mean(st), mean(od)


b = metrics(False)
u = metrics(True)
print(f"{'':22}{'tech':>8}{'tac':>8}{'step':>8}{'order':>8}")
print(f"{'baseline(Exec.4/DE.6)':22}{b[0]:>8.4f}{b[1]:>8.4f}{b[2]:>8.4f}{b[3]:>8.4f}")
print(f"{'unified(all 0.5)':22}{u[0]:>8.4f}{u[1]:>8.4f}{u[2]:>8.4f}{u[3]:>8.4f}")
print(f"{'delta':22}{u[0]-b[0]:>+8.4f}{u[1]-b[1]:>+8.4f}{u[2]-b[2]:>+8.4f}{u[3]-b[3]:>+8.4f}")
