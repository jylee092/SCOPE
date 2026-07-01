"""Verify SCOPE's recovered chains are IDENTICAL with P_sem on vs off (D=0),
so all chain-derived tables (tech/tac-LCS, step-cov, novelty, strict) are
unchanged by P_sem removal. Read-only, cached mappings, no API."""
import json, sys
from pathlib import Path
from collections import Counter as _C
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config, pandas as pd
from pipeline.attack_chain import (sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer, MultiDimTransitionScorer,
    load_campaign_library, topk_viterbi, apply_emission_confidence_bypass)
from pipeline.technique_io import load_or_build_technique_io

tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json"))
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

def chain(gn, use_sem):
    multi = MultiDimTransitionScorer(
        tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
        sem_scorer=(sem if use_sem else None), cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=(0.3 if use_sem else 0.0), w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))
    vit = topk_viterbi(gn, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
                       skip_penalty=config.VITERBI_SKIP_PENALTY,
                       transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
    if bypass is not None:
        vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
    return [(n.get("group_idx"), n.get("technique_id"), n.get("tactic")) for n in vit.score_breakdown]

n=0; diff=0; difflist=[]
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists() and config.FINALE_CSV_PATH.exists()):
        continue
    ttp=json.load(open(config.TTP_MAPPING_JSON_PATH,encoding="utf-8"))
    ft=json.load(open(config.FEATURE_RESULT_JSON_PATH,encoding="utf-8"))
    fdf=pd.read_csv(config.FINALE_CSV_PATH,low_memory=False); fdf["TimeCreated"]=pd.to_datetime(fdf["TimeCreated"],errors="coerce")
    gn=build_group_nodes(prep(ttp,fdf),tm,{f["group_id"]:f for f in ft})
    if not gn: continue
    n+=1
    c_on=chain(gn,True); c_off=chain(gn,False)
    if c_on != c_off:
        diff+=1; difflist.append(ds.stem)
print(f"\n=== chain identity check: P_sem ON vs OFF (D=0), n={n} ===", file=sys.stderr)
print(f"  chains differ in {diff}/{n} scenarios", file=sys.stderr)
if difflist: print("  differing:", difflist[:10], file=sys.stderr)
print("  => "+("IDENTICAL: all SCOPE chain-derived tables unchanged." if diff==0 else "SOME DIFFER: recompute affected SCOPE tables."), file=sys.stderr)
