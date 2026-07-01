"""Recompute SCOPE tab:main chain metrics at the NEW config (D=0, P_sem OFF),
compared to reported (tech 0.68 / tac 0.77 / step 0.71 / order 0.60).
Read-only, cached mappings, no API. Also chain-novelty (chain-dependent)."""
import json, sys
from pathlib import Path
from statistics import mean
from collections import Counter as _C
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config, pandas as pd
from pipeline.attack_chain import (sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, CausalScorer, MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi, apply_emission_confidence_bypass, compute_novelty)
from pipeline.technique_io import load_or_build_technique_io
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment

tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json"))
bypass = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)
TAU2 = 0.30  # chain-novel threshold

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

tech=[]; tac=[]; step=[]; order=[]; chain_novel=[]; coherent=[]
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists() and config.FINALE_CSV_PATH.exists()):
        continue
    flow = get_flow(ds.stem)
    if not flow: continue
    ttp=json.load(open(config.TTP_MAPPING_JSON_PATH,encoding="utf-8"))
    ft=json.load(open(config.FEATURE_RESULT_JSON_PATH,encoding="utf-8"))
    fdf=pd.read_csv(config.FINALE_CSV_PATH,low_memory=False); fdf["TimeCreated"]=pd.to_datetime(fdf["TimeCreated"],errors="coerce")
    gn=build_group_nodes(prep(ttp,fdf),tm,{f["group_id"]:f for f in ft})
    if not gn: continue
    multi=MultiDimTransitionScorer(tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
        sem_scorer=None, cau_scorer=cau, w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))
    vit=topk_viterbi(gn, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
        skip_penalty=config.VITERBI_SKIP_PENALTY, transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
    if bypass is not None:
        vit=apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
    bd=vit.score_breakdown
    if not bd:
        tech.append(0); tac.append(0); step.append(0); order.append(0); chain_novel.append(0); coherent.append(0); continue
    r=evaluate_chain_alignment(ds.stem, bd, ref_flow=flow)
    tl=r.get("technique_lcs_norm",0)
    tech.append(tl); tac.append(r.get("tactic_lcs_norm",0)); step.append(r.get("step_coverage",0)); order.append(r.get("order_accuracy",0))
    nov = vit.novelty_score
    cn = 1.0 if nov >= TAU2 else 0.0
    chain_novel.append(cn)
    coherent.append(1.0 if (cn and tl >= 0.5) else 0.0)

print(f"\n=== SCOPE @ D=0, P_sem OFF  (n={len(tech)}) ===", file=sys.stderr)
print(f"  technique-LCS : {mean(tech):.4f}  (reported 0.68)", file=sys.stderr)
print(f"  tactic-LCS    : {mean(tac):.4f}  (reported 0.77)", file=sys.stderr)
print(f"  step coverage : {mean(step):.4f}  (reported 0.71)", file=sys.stderr)
print(f"  order accuracy: {mean(order):.4f}  (reported 0.60)", file=sys.stderr)
print(f"  chain-novel   : {mean(chain_novel):.4f}  (reported 0.91)", file=sys.stderr)
print(f"  coherent-novel: {mean(coherent):.4f}  (reported 0.83)", file=sys.stderr)
