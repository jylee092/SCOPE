"""Isolate the runtime/memory delta of the framework change for Q5 (section eff).

The only pipeline change that touches Q5 is P_sem removal (skip was already 0 in
the timed main pipeline). Grouping / feature / mapping stages are unchanged, so we
re-time ONLY the Viterbi stage (transition scoring + Top-K + bypass) under:
  NEW  : sem_scorer = None            (current framework)
  OLD  : sem_scorer = ATTACK-BERT     (w_sem=0.3, previous framework)
across all 35 scenarios, from cached ttp_mapping + feature_result (no API).
Also reports the peak-RSS cost of loading the ATTACK-BERT bi-encoder.
"""
import json, sys, os, time, gc
from pathlib import Path
from statistics import median
from collections import Counter as _C
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config, pandas as pd, psutil
from pipeline.attack_chain import (sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, CausalScorer, get_semantic_scorer, MultiDimTransitionScorer,
    load_campaign_library, topk_viterbi, apply_emission_confidence_bypass)
from pipeline.technique_io import load_or_build_technique_io

proc = psutil.Process(os.getpid())
def rss_mb(): return proc.memory_info().rss / (1024*1024)

tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json"))
BYPASS = float(getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", 0.75))
ALPHA  = float(getattr(config, "VITERBI_TRANSITION_WEIGHT", 0.5))
BEAM   = int(getattr(config, "VITERBI_BEAM_K", 5))
SP     = float(getattr(config, "VITERBI_SKIP_PENALTY", 0.25))
SLP    = float(getattr(config, "SELF_LOOP_TID_PENALTY", 1.0))

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

# ---- build (group_nodes) per scenario once (shared by both configs) ----
scen=[]
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists() and config.FINALE_CSV_PATH.exists()):
        continue
    ttp=json.load(open(config.TTP_MAPPING_JSON_PATH,encoding="utf-8"))
    ft=json.load(open(config.FEATURE_RESULT_JSON_PATH,encoding="utf-8"))
    fdf=pd.read_csv(config.FINALE_CSV_PATH,low_memory=False); fdf["TimeCreated"]=pd.to_datetime(fdf["TimeCreated"],errors="coerce")
    gn=build_group_nodes(prep(sort_results_by_time(ttp,fdf)),tm,{f["group_id"]:f for f in ft})
    if gn: scen.append((ds.stem, gn))
print(f"scenarios={len(scen)}  total_groups={sum(len(g) for _,g in scen)}", file=sys.stderr)

def run_all(sem):
    m=MultiDimTransitionScorer(tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
        sem_scorer=sem, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=(0.3 if sem is not None else 0.0), w_cau=config.W_CAU,
        self_loop_tid_penalty=SLP)
    t0=time.perf_counter()
    for _stem,gn in scen:
        vit=topk_viterbi(gn, m, beam_k=BEAM, max_skip=0, skip_penalty=SP,
            transition_weight=ALPHA, campaigns=camp)
        apply_emission_confidence_bypass(vit, m, sim_threshold=BYPASS)
    return time.perf_counter()-t0

# ---- NEW config: sem=None ----
gc.collect(); base_rss=rss_mb()
new_times=[run_all(None) for _ in range(5)]
new_rss=rss_mb()

# ---- OLD config: load ATTACK-BERT bi-encoder, measure its memory cost ----
pre_load=rss_mb()
sem=get_semantic_scorer(config.SEMANTIC_MODEL, backend="bi-encoder")
# warm one encode so lazy weights/graph are realized before timing
try: sem.score("a","b")
except Exception as e: print("warm err:",e,file=sys.stderr)
post_load=rss_mb()
old_times=[run_all(sem) for _ in range(5)]
old_rss=rss_mb()

print("\n==== Q5 Viterbi-stage timing delta (5 reps, cached data, no API) ====")
print(f"scenarios={len(scen)}  total_groups={sum(len(g) for _,g in scen)}")
print(f"NEW (P_sem off): median {median(new_times):.3f}s  min {min(new_times):.3f}s  reps={[round(x,3) for x in new_times]}")
print(f"OLD (P_sem on):  median {median(old_times):.3f}s  min {min(old_times):.3f}s  reps={[round(x,3) for x in old_times]}")
d=median(old_times)-median(new_times)
print(f"Viterbi-stage delta (OLD-NEW): {d:.3f}s  ({100*d/median(old_times):.1f}% of old viterbi)")
print(f"\nATTACK-BERT load RSS cost: {post_load-pre_load:.0f} MB (pre {pre_load:.0f} -> post {post_load:.0f})")
print(f"base RSS {base_rss:.0f} MB | after-new {new_rss:.0f} | after-old {old_rss:.0f}")
print(f"\n-- context: paper reports total pipeline 1034s, viterbi stage 17-18% (~180s) --")
