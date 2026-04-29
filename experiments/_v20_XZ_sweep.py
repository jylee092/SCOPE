"""
v20 X+Z hybrid sweep.

FAISS top-1 기본 존중, sim(top-1) 이 임계치 이하면 Viterbi 선택 유지.
sim_threshold sweep.
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
    apply_emission_confidence_bypass, _SEM_SCORER_CACHE,
)
from pipeline.technique_io import load_or_build_technique_io
import pandas as pd

from experiments.run_eval_post_viterbi import _run as eval_post_run
from experiments.run_eval_v2 import main as eval_v2


def run_viterbi_all(sim_thr: float | None):
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
    tac = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
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
        if sim_thr is not None:
            vit = apply_emission_confidence_bypass(vit, multi, sim_threshold=sim_thr)
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


# sim distribution: p25=0.654, p50=0.699, p75=0.743, p90=0.809
VARIANTS = [
    ("vanilla_no_bypass",  None),
    ("bypass_sim_0.60",    0.60),  # 10% 그룹이 bypass
    ("bypass_sim_0.65",    0.65),  # ~25% bypass
    ("bypass_sim_0.70",    0.70),  # ~50% bypass
    ("bypass_sim_0.75",    0.75),  # ~75% bypass
    ("bypass_sim_0.80",    0.80),  # ~85% bypass
    ("bypass_sim_0.85",    0.85),  # ~95% bypass
    ("bypass_sim_1.00",    1.00),  # 모두 bypass = FAISS-only
]


def main():
    results = []
    for name, thr in VARIANTS:
        print(f"\n═══ {name}  sim_threshold={thr} ═══")
        run_viterbi_all(thr)
        eval_post_run(strong_only=False, label="ALL TPs")
        eval_v2()
        m = read_metrics()
        print(f"  tech_lcs={m['tech_lcs']:.4f}  tac_lcs={m['tac_lcs']:.4f}  "
              f"step_cov={m['step_cov']:.4f}  order={m['order']:.4f}  "
              f"vit_mic={m['viterbi_mic']:.4f}  faiss_mic={m['faiss_mic']:.4f}")
        results.append({"variant": name, "sim_threshold": thr, **m})

    out = config.OUTPUT_BASE_DIR / "v20_XZ_sweep_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")
    print("\n" + "=" * 115)
    print(f"{'variant':<22} {'thr':>6} {'tech':>8} {'tac':>8} {'step':>8} {'order':>8} {'vit_mic':>10} {'faiss_mic':>10}")
    for r in results:
        t = r["sim_threshold"]
        t_str = "—" if t is None else f"{t:.2f}"
        print(f"{r['variant']:<22} {t_str:>6} {r['tech_lcs']:>8.4f} {r['tac_lcs']:>8.4f} "
              f"{r['step_cov']:>8.4f} {r['order']:>8.4f} "
              f"{r['viterbi_mic']:>10.4f} {r['faiss_mic']:>10.4f}")


if __name__ == "__main__":
    main()
