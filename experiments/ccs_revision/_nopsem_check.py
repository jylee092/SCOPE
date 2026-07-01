"""Read-only: does removing P_sem change the main (complete-data, D=0) score?
Reuses cached canonical ttp_mapping+features; rebuilds transition scoring with
sem on vs sem off. No API, no canonical overwrite."""
import json, sys
from pathlib import Path
from statistics import mean
from collections import Counter as _C
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config, pandas as pd
from pipeline.attack_chain import (sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer, MultiDimTransitionScorer,
    load_campaign_library, topk_viterbi, apply_emission_confidence_bypass)
from pipeline.technique_io import load_or_build_technique_io
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment

tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json")) if config.USE_CAUSAL_SCORING else None
sem = get_semantic_scorer(config.SEMANTIC_MODEL, backend=config.SEMANTIC_BACKEND,
                          calibration=getattr(config,"SEM_CALIBRATION","linear"))
bypass = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)

def prep(ttp, fdf):
    sr = sort_results_by_time(ttp, fdf); fb=float(getattr(config,"FAMILY_BOOST",0.0)); fw=int(getattr(config,"FAMILY_BOOST_WIDTH",10))
    if fb>0:
        for r in sr:
            cd=r.get("similar_techniques",[])
            if len(cd)<2: continue
            wide=min(fw,len(cd)); par=[c["technique_id"].split(".",1)[0] for c in cd[:wide]]; pc=_C(par)
            sc=sorted([(c.get("p_ttp",c.get("similarity",0))*(1.0+fb*(pc[par[i]]-1)),i,c) for i,c in enumerate(cd[:wide])],key=lambda x:-x[0])
            r["similar_techniques"]=[c for _,_,c in sc]+cd[wide:]
    return sr

def run(gnodes, use_sem):
    multi = MultiDimTransitionScorer(
        tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
        sem_scorer=(sem if use_sem else None), cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=(config.W_SEM if use_sem else 0.0), w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))
    vit = topk_viterbi(gnodes, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
                       skip_penalty=config.VITERBI_SKIP_PENALTY,
                       transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
    if bypass is not None:
        vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
    return vit.score_breakdown

on, off = [], []
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists() and config.FINALE_CSV_PATH.exists()):
        continue
    flow = get_flow(ds.stem)
    if not flow: continue
    ttp = json.load(open(config.TTP_MAPPING_JSON_PATH, encoding="utf-8"))
    ft = json.load(open(config.FEATURE_RESULT_JSON_PATH, encoding="utf-8"))
    fdf = pd.read_csv(config.FINALE_CSV_PATH, low_memory=False); fdf["TimeCreated"]=pd.to_datetime(fdf["TimeCreated"],errors="coerce")
    gn = build_group_nodes(prep(ttp,fdf), tm, {f["group_id"]:f for f in ft})
    if not gn: continue
    for lst, use in ((on,True),(off,False)):
        bd = run(gn, use)
        lst.append(evaluate_chain_alignment(ds.stem, bd, ref_flow=flow).get("technique_lcs_norm",0.0) if bd else 0.0)
print(f"\n=== main (D=0, complete) P_sem on vs off, n={len(on)} ===", file=sys.stderr)
print(f"  P_sem ON : tech-LCS = {mean(on):.4f}", file=sys.stderr)
print(f"  P_sem OFF: tech-LCS = {mean(off):.4f}", file=sys.stderr)
print(f"  delta = {mean(off)-mean(on):+.4f}", file=sys.stderr)
