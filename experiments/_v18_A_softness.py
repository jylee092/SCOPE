"""
v18 A-softness sweep.

Task A (per-tactic self-loop / per-target wildcard-IN) 의 공격성을 단계별로 완화해
tech/tac-LCS + Viterbi micro 의 sweet spot 찾기.

baseline_legacy    : A 비활성 (self-loop 일괄 0.5, wildcard-IN 일괄 0.8)
A_aggressive       : 현재 config (Exec 0.20, DE self 0.70, DE-in 0.55)
A_mild             : Exec 0.35, DE self 0.60, DE-in 0.65
A_soft             : Exec 0.40, DE self 0.55, DE-in 0.70
A_exec_only        : Exec self-loop 만 0.35 로, 나머지 legacy default
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
    _SEM_SCORER_CACHE,
)
from pipeline.technique_io import load_or_build_technique_io
import pandas as pd

from experiments.run_eval_post_viterbi import _run as eval_post_run
from experiments.run_eval_v2 import main as eval_v2


def make_tac_scorer(self_loop: dict, wildcard_in: dict) -> TacticalScorer:
    t = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    t._self_loop_w = dict(self_loop)
    t._wildcard_in_w = dict(wildcard_in)
    return t


def run_viterbi_all(self_loop: dict, wildcard_in: dict):
    _SEM_SCORER_CACHE.clear()
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    technique_io = load_or_build_technique_io(
        str(config.MITRE_CSV_PATH),
        cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
    )
    sem = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
    ) if config.USE_SEMANTIC_SCORING else None
    tac = make_tac_scorer(self_loop, wildcard_in)
    cau = CausalScorer(technique_io=technique_io) if config.USE_CAUSAL_SCORING else None
    multi = MultiDimTransitionScorer(
        tac_scorer=tac, sem_scorer=sem, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0),
    )
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
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
            max_skip=config.VITERBI_MAX_SKIP,
            skip_penalty=config.VITERBI_SKIP_PENALTY,
            transition_weight=config.VITERBI_TRANSITION_WEIGHT,
            campaigns=campaigns,
        )
        with open(config.VITERBI_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(vit.score_breakdown, f, ensure_ascii=False, indent=2)


def read_metrics():
    with open(config.OUTPUT_BASE_DIR / "eval_v2_results.json", encoding="utf-8") as f:
        v2 = json.load(f)
    chains = [r["chain"] for r in v2 if "chain" in r and "error" not in r["chain"]]
    def _avg(rows, key):
        vals = [r.get(key, 0) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0
    import csv
    tp_hits = vit_hits = n = 0
    with open(config.OUTPUT_BASE_DIR / "eval_post_viterbi_all.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            tp_hits += int(row.get("faiss_hit", 0) or 0)
            vit_hits += int(row.get("viterbi_hit", 0) or 0)
    return {
        "tech_lcs":    _avg(chains, "technique_lcs_norm"),
        "tac_lcs":     _avg(chains, "tactic_lcs_norm"),
        "step_cov":    _avg(chains, "step_coverage"),
        "order":       _avg(chains, "order_accuracy"),
        "viterbi_mic": vit_hits / max(n, 1),
        "faiss_mic":   tp_hits / max(n, 1),
    }


VARIANTS = [
    # (name, self_loop_dict, wildcard_in_dict)
    ("baseline_legacy", {}, {}),  # 전통적 uniform
    ("A_aggressive", {
        "Execution": 0.20, "Defense Evasion": 0.70, "Discovery": 0.60,
        "Credential Access": 0.60, "Collection": 0.55, "Persistence": 0.40,
        "Privilege Escalation": 0.30, "Lateral Movement": 0.40, "Command and Control": 0.55,
    }, {"Defense Evasion": 0.55, "Privilege Escalation": 0.70}),
    ("A_mild", {
        "Execution": 0.35, "Defense Evasion": 0.60, "Discovery": 0.55,
        "Credential Access": 0.55, "Collection": 0.50, "Persistence": 0.45,
        "Privilege Escalation": 0.40, "Lateral Movement": 0.45, "Command and Control": 0.55,
    }, {"Defense Evasion": 0.65, "Privilege Escalation": 0.75}),
    ("A_soft", {
        "Execution": 0.40, "Defense Evasion": 0.55, "Discovery": 0.52,
        "Credential Access": 0.52, "Collection": 0.50, "Persistence": 0.48,
        "Privilege Escalation": 0.45, "Lateral Movement": 0.48, "Command and Control": 0.52,
    }, {"Defense Evasion": 0.70, "Privilege Escalation": 0.78}),
    ("A_exec_only_35", {"Execution": 0.35}, {}),
    ("A_exec_only_40", {"Execution": 0.40}, {}),
    ("A_exec_40_de_60", {"Execution": 0.40, "Defense Evasion": 0.60},
                        {"Defense Evasion": 0.70}),
]


def main():
    results = []
    for name, sl, wi in VARIANTS:
        print(f"\n═══ {name}  self_loop={sl}  wildcard_in={wi} ═══")
        run_viterbi_all(sl, wi)
        eval_post_run(strong_only=False, label="ALL TPs")
        eval_v2()
        m = read_metrics()
        print(f"  tech_lcs={m['tech_lcs']:.4f}  tac_lcs={m['tac_lcs']:.4f}  "
              f"step_cov={m['step_cov']:.4f}  order={m['order']:.4f}  "
              f"vit_mic={m['viterbi_mic']:.4f}  faiss_mic={m['faiss_mic']:.4f}")
        results.append({"variant": name,
                        "self_loop": sl, "wildcard_in": wi, **m})

    out = config.OUTPUT_BASE_DIR / "v18_A_softness_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")
    print("\n" + "=" * 110)
    print(f"{'variant':<22} {'tech':>8} {'tac':>8} {'step':>8} {'order':>8} {'vit_mic':>10} {'faiss_mic':>10}")
    for r in results:
        print(f"{r['variant']:<22} {r['tech_lcs']:>8.4f} {r['tac_lcs']:>8.4f} "
              f"{r['step_cov']:>8.4f} {r['order']:>8.4f} "
              f"{r['viterbi_mic']:>10.4f} {r['faiss_mic']:>10.4f}")


if __name__ == "__main__":
    main()
