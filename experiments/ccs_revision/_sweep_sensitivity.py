"""Recompute the q5 sensitivity sweeps at the NEW config (D=0, P_sem OFF):
Panel A: alpha sweep (no bypass); Panel B: bypass-threshold sweep (alpha=0.5).
Read-only, cached mappings, no API."""
import json, sys
from pathlib import Path
from statistics import mean
from collections import Counter as _C
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

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]
THRS   = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.00]

def prep(sr):
    fb=float(getattr(config,"FAMILY_BOOST",0.0)); fw=int(getattr(config,"FAMILY_BOOST_WIDTH",10))
    if fb>0:
        for r in sr:
            cd=r.get("similar_techniques",[])
            if len(cd)<2: continue
            wide=min(fw,len(cd)); par=[c["technique_id"].split(".",1)[0] for c in cd[:wide]]; pc=_C(par)
            sc=sorted([(c.get("p_ttp",c.get("similarity",0))*(1.0+fb*(pc[par[i]]-1)),i,c) for i,c in enumerate(cd[:wide])],key=lambda x:-x[0])
            r["similar_techniques"]=[c for _,_,c in sc]+cd[wide:]
    return sr

# build (group_nodes, flow) per scenario once
scen = []
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists() and config.FINALE_CSV_PATH.exists()):
        continue
    flow = get_flow(ds.stem)
    if not flow: continue
    ttp=json.load(open(config.TTP_MAPPING_JSON_PATH,encoding="utf-8"))
    ft=json.load(open(config.FEATURE_RESULT_JSON_PATH,encoding="utf-8"))
    fdf=pd.read_csv(config.FINALE_CSV_PATH,low_memory=False); fdf["TimeCreated"]=pd.to_datetime(fdf["TimeCreated"],errors="coerce")
    gn=build_group_nodes(prep(sort_results_by_time(ttp,fdf)),tm,{f["group_id"]:f for f in ft})
    if gn: scen.append((ds.stem, gn, flow))

def multi():
    return MultiDimTransitionScorer(tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
        sem_scorer=None, cau_scorer=cau, w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))

def run(alpha, thr):
    vals=[]
    m=multi()
    for stem, gn, flow in scen:
        vit=topk_viterbi(gn, m, beam_k=config.VITERBI_BEAM_K, max_skip=0,
            skip_penalty=config.VITERBI_SKIP_PENALTY, transition_weight=alpha, campaigns=camp)
        if thr is not None:
            vit=apply_emission_confidence_bypass(vit, m, sim_threshold=thr)
        bd=vit.score_breakdown
        vals.append(evaluate_chain_alignment(stem, bd, ref_flow=flow).get("technique_lcs_norm",0.0) if bd else 0.0)
    return mean(vals)

print(f"n={len(scen)}", file=sys.stderr)
a_y=[round(run(a, None),4) for a in ALPHAS]
print("PanelA alpha(no bypass):", dict(zip(ALPHAS,a_y)), file=sys.stderr)
b_y=[round(run(0.5, thr),4) for thr in THRS]
b_van=round(run(0.5, None),4)
print("PanelB bypass(alpha=0.5):", dict(zip(THRS,b_y)), file=sys.stderr)
print("no-bypass ref (alpha=0.5):", b_van, file=sys.stderr)
