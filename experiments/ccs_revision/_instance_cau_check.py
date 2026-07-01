"""Read-only validation: instance-level P_cau vs current type-level P_cau.
Main data, D=0, P_sem OFF. Reuses cached ttp_mapping+features. No overwrite, no API.

Instance P_cau: Out(b_i) = concrete artifacts the group PRODUCES (dropped files,
written registry keys, spawned/created processes); In(b_j) = artifacts it
CONSUMES (executed images, accessed/loaded processes/modules, parent context,
read files). Directional flow score = saturating fn of |Out(i) ∩ In(j)|, with a
small floor so a missing link discounts rather than disconnects (geometric-mean safe).
"""
import json, sys, math
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

_FLOOR = 0.15   # missing-link floor


def _norm(s):
    return str(s).strip().lower()


def extract_inout(feat: dict):
    """Return (out_set, in_set) of normalized instance strings from a group feature dict."""
    f = feat.get("features", feat)
    out, inn = set(), set()
    ec = f.get("execution_context") or {}
    for pc in ec.get("process_chains", []) or []:
        rel = pc.get("relation")
        ci = pc.get("child_image"); pi = pc.get("parent_image")
        sg = pc.get("source_guid"); tg = pc.get("target_guid")
        if rel == "create":
            if ci: out.add("proc:" + _norm(ci))
            if tg: out.add("proc:" + _norm(tg))
            if pi: inn.add("proc:" + _norm(pi))
            if sg: inn.add("proc:" + _norm(sg))
        elif rel == "access":
            # actor (source) consumes target
            if tg: inn.add("proc:" + _norm(tg))
            if ci: inn.add("proc:" + _norm(ci))
            if sg: out.add("proc:" + _norm(sg))
            if pi: out.add("proc:" + _norm(pi))
        else:
            for v in (ci, pi):
                if v: out.add("proc:" + _norm(v)); inn.add("proc:" + _norm(v))
            for v in (sg, tg):
                if v: out.add("proc:" + _norm(v)); inn.add("proc:" + _norm(v))
    per = f.get("persistence") or {}
    for df_ in per.get("dropped_files", []) or []:
        p = df_ if isinstance(df_, str) else (df_.get("path") or df_.get("file") or "")
        if p: out.add("file:" + _norm(p))
    for rs in per.get("registry_signals", []) or []:
        k = rs if isinstance(rs, str) else (rs.get("key") or rs.get("target") or "")
        if k: out.add("reg:" + _norm(k))
    cs = f.get("command_script") or {}
    for e in cs.get("entries", []) or []:
        img = e if isinstance(e, str) else (e.get("image") or e.get("cmd") or "")
        if img: inn.add("file:" + _norm(img))
    net = f.get("network") or {}
    for c in net.get("connections", []) or []:
        ep = c if isinstance(c, str) else (str(c.get("dest_ip", "")) + ":" + str(c.get("dest_port", "")))
        if ep and ep != ":":
            out.add("net:" + _norm(ep)); inn.add("net:" + _norm(ep))
    return out, inn


def inst_cau(out_i, in_j):
    if not out_i or not in_j:
        return _FLOOR
    ov = out_i & in_j
    if not ov:
        return _FLOOR
    # saturating: 1 shared artifact already strong evidence
    return min(1.0, _FLOOR + (1.0 - _FLOOR) * (1.0 - math.exp(-1.5 * len(ov))))


class InstanceMultiScorer(MultiDimTransitionScorer):
    """Same as MultiDim but replaces causal with instance-flow using node._io sets."""
    def score(self, cand_i, cand_j, node_i, node_j):
        tac_result = self.tac.score(cand_i.tactic, cand_j.tactic)
        p_tac = max(tac_result.weight, 1e-6)
        oi, _ = getattr(node_i, "_io", (set(), set()))
        _, ij = getattr(node_j, "_io", (set(), set()))
        p_cau = max(inst_cau(oi, ij), 1e-6)
        wt, wc = self._w_tac, self._w_cau
        tot = wt + wc
        wt, wc = wt / tot, wc / tot
        fused = math.exp(wt * math.log(p_tac) + wc * math.log(p_cau))
        if cand_i.technique_id == cand_j.technique_id and self._self_loop_tid_penalty < 1.0:
            fused *= self._self_loop_tid_penalty
        return fused, tac_result


tm = load_tactic_map(str(config.MITRE_CSV_PATH))
camp = load_campaign_library(str(config.CAMPAIGN_FOLDER), tm)
cau = CausalScorer(technique_io=load_or_build_technique_io(
    str(config.MITRE_CSV_PATH), cache_path=config.OUTPUT_BASE_DIR/"technique_io_cache.json"))
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


def runvit(multi, gn):
    vit = topk_viterbi(gn, multi, beam_k=config.VITERBI_BEAM_K, max_skip=0,
                       skip_penalty=config.VITERBI_SKIP_PENALTY,
                       transition_weight=config.VITERBI_TRANSITION_WEIGHT, campaigns=camp)
    if bypass is not None:
        vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=float(bypass))
    return vit.score_breakdown


type_s, inst_s = [], []
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    config.configure_dataset(ds)
    if not (config.TTP_MAPPING_JSON_PATH.exists() and config.FEATURE_RESULT_JSON_PATH.exists() and config.FINALE_CSV_PATH.exists()):
        continue
    flow = get_flow(ds.stem)
    if not flow: continue
    ttp = json.load(open(config.TTP_MAPPING_JSON_PATH, encoding="utf-8"))
    ft = json.load(open(config.FEATURE_RESULT_JSON_PATH, encoding="utf-8"))
    fbg = {f["group_id"]: f for f in ft}
    fdf = pd.read_csv(config.FINALE_CSV_PATH, low_memory=False); fdf["TimeCreated"]=pd.to_datetime(fdf["TimeCreated"],errors="coerce")
    gn = build_group_nodes(prep(ttp, fdf), tm, fbg)
    if not gn: continue
    # attach instance io
    for node in gn:
        node._io = extract_inout(fbg.get(node.group_id, {}))
    # type-level (current), D=0, no P_sem
    mt = MultiDimTransitionScorer(tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
                                  sem_scorer=None, cau_scorer=cau,
                                  w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
                                  self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))
    bd = runvit(mt, gn)
    type_s.append(evaluate_chain_alignment(ds.stem, bd, ref_flow=flow).get("technique_lcs_norm",0.0) if bd else 0.0)
    # instance-level
    mi = InstanceMultiScorer(tac_scorer=TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD),
                             sem_scorer=None, cau_scorer=cau,
                             w_tac=config.W_TAC, w_sem=0.0, w_cau=config.W_CAU,
                             self_loop_tid_penalty=getattr(config,"SELF_LOOP_TID_PENALTY",1.0))
    bd = runvit(mi, gn)
    inst_s.append(evaluate_chain_alignment(ds.stem, bd, ref_flow=flow).get("technique_lcs_norm",0.0) if bd else 0.0)

print(f"\n=== main D=0 noPsem: type-cau vs instance-cau, n={len(type_s)} ===", file=sys.stderr)
print(f"  type-level P_cau (current):  tech-LCS = {mean(type_s):.4f}", file=sys.stderr)
print(f"  instance-level P_cau (new):  tech-LCS = {mean(inst_s):.4f}", file=sys.stderr)
print(f"  delta = {mean(inst_s)-mean(type_s):+.4f}", file=sys.stderr)
