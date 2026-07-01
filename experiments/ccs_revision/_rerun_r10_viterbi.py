"""Regenerate R10 (LLM-swap) gpt & template viterbi at the NEW config
(D=0, P_sem OFF) from their cached ttp_mapping + feature_result. No API.
Then tab:llmswap (via r10_eval) reflects the new framework."""
import json, sys
from pathlib import Path
from collections import Counter as _C
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config, pandas as pd
from pipeline.attack_chain import (sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, CausalScorer, MultiDimTransitionScorer, load_campaign_library,
    topk_viterbi, apply_emission_confidence_bypass)
from pipeline.technique_io import load_or_build_technique_io

R10 = config.OUTPUT_BASE_DIR / "_ccs_revision" / "R10_llm_swap"
# stem -> canonical Dataset scenario path (for the correct time frame / FINALE_CSV)
STEM2DS = {p.stem: p for p in config.DATASET_FOLDER.rglob("*.json")}
tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json"))
bypass = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)

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

for base in ["gpt", "template"]:
    root = R10 / base
    if not root.exists():
        print(f"[skip] {base}: no dir"); continue
    n=0
    for mp in root.rglob("*_ttp_mapping.json"):
        d = mp.parent
        stem = mp.name.replace("_ttp_mapping.json","")
        fp = d / f"{stem}_feature_result.json"
        vp = d / f"{stem}_viterbi.json"
        if not fp.exists():
            continue
        ttp = json.load(open(mp, encoding="utf-8"))
        ft = json.load(open(fp, encoding="utf-8"))
        # time frame from the canonical scenario (same events as R10)
        ds = STEM2DS.get(stem)
        if ds is None:
            continue
        config.configure_dataset(ds)
        fdf = pd.read_csv(config.FINALE_CSV_PATH, low_memory=False)
        fdf["TimeCreated"] = pd.to_datetime(fdf["TimeCreated"], errors="coerce")
        sr = prep(sort_results_by_time(ttp, fdf))
        gn = build_group_nodes(sr, tm, {f["group_id"]: f for f in ft})
        if not gn:
            json.dump([], open(vp,"w")); n+=1; continue
        multi = MultiDimTransitionScorer(
            tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
            sem_scorer=None, cau_scorer=cau, w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
            self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))
        vit = topk_viterbi(gn, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
            skip_penalty=config.VITERBI_SKIP_PENALTY, transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
        if bypass is not None:
            vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
        json.dump(vit.score_breakdown, open(vp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        n+=1
    print(f"[{base}] regenerated {n} viterbi at new config")
