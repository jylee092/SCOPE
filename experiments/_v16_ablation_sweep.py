"""
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


def run_viterbi_all(sem_cal, sim_gate, sigmoid_center=0.5, sigmoid_scale=8.0):
    """Run Viterbi over all 35 scenarios with the given variant flags.

    Overwrites *_viterbi.json in-place so downstream eval_* scripts pick them up.
    """
    # Flush sem scorer cache so calibration change takes effect
    _SEM_SCORER_CACHE.clear()

    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    technique_io = load_or_build_technique_io(
        str(config.MITRE_CSV_PATH),
        cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
    )
    sem = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
        calibration=sem_cal,
        sigmoid_center=sigmoid_center,
        sigmoid_scale=sigmoid_scale,
    ) if config.USE_SEMANTIC_SCORING else None
    tac = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    cau = CausalScorer(technique_io=technique_io) if config.USE_CAUSAL_SCORING else None
    multi = MultiDimTransitionScorer(
        tac_scorer=tac, sem_scorer=sem, cau_scorer=cau,
        w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
        self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0),
    )
    campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)

    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    for i, ds in enumerate(datasets, 1):
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
            sim_gated=sim_gate,
            sim_margin_low=config.VITERBI_SIM_MARGIN_LOW,
            sim_margin_high=config.VITERBI_SIM_MARGIN_HIGH,
            alpha_low_sim=config.VITERBI_ALPHA_LOW_SIM,
            alpha_high_sim=config.VITERBI_ALPHA_HIGH_SIM,
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
    # post-viterbi micro
    import csv
    tp_hits = vit_hits = 0
    n = 0
    with open(config.OUTPUT_BASE_DIR / "eval_post_viterbi_all.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            if str(row.get("faiss_top1_plausible","")).lower() == "true": tp_hits += 1
            if str(row.get("viterbi_pick_plausible","")).lower() == "true": vit_hits += 1
    return {
        "tech_lcs": _avg(chains, "technique_lcs_norm"),
        "tac_lcs":  _avg(chains, "tactic_lcs_norm"),
        "step_cov": _avg(chains, "step_coverage"),
        "order":    _avg(chains, "order_accuracy"),
        "viterbi_mic": vit_hits / max(n, 1),
        "faiss_mic":   tp_hits / max(n, 1),
    }


VARIANTS = [
    ("baseline_orig",      "linear",  False, 0.5, 8.0),
    ("simgate",            "linear",  True,  0.5, 8.0),
    ("sem_c05_b8",         "sigmoid", False, 0.5, 8.0),
    ("sem_c062_b8",        "sigmoid", False, 0.62, 8.0),
    ("sem_c062_b6",        "sigmoid", False, 0.62, 6.0),
    ("simgate+sem_c062_b8","sigmoid", True,  0.62, 8.0),
    ("simgate+sem_c062_b6","sigmoid", True,  0.62, 6.0),
]


def main():
    results = []
    for name, sem_cal, sim_gate, c0, beta in VARIANTS:
        print(f"\n═══ {name}  sem={sem_cal}(c0={c0},β={beta})  sim_gate={sim_gate} ═══")
        run_viterbi_all(sem_cal, sim_gate, c0, beta)
        # evaluate
        eval_post_run(strong_only=False, label="ALL TPs")
        eval_v2()  # emits eval_v2_results.json
        m = read_metrics()
        print(f"  tech_lcs={m['tech_lcs']:.4f}  tac_lcs={m['tac_lcs']:.4f}  "
              f"step_cov={m['step_cov']:.4f}  order={m['order']:.4f}  "
              f"vit_mic={m['viterbi_mic']:.4f}  faiss_mic={m['faiss_mic']:.4f}")
        results.append({"variant": name, **m,
                        "sem_calibration": sem_cal, "sim_gated": sim_gate,
                        "sigmoid_center": c0, "sigmoid_scale": beta})

    out = config.OUTPUT_BASE_DIR / "v16b_ablation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")

    # summary table
    print("\n" + "=" * 110)
    print(f"{'variant':<25} {'tech':>8} {'tac':>8} {'step':>8} {'order':>8} {'vit_mic':>10} {'faiss_mic':>10}")
    for r in results:
        print(f"{r['variant']:<25} {r['tech_lcs']:>8.4f} {r['tac_lcs']:>8.4f} "
              f"{r['step_cov']:>8.4f} {r['order']:>8.4f} {r['viterbi_mic']:>10.4f} {r['faiss_mic']:>10.4f}")


if __name__ == "__main__":
    main()
