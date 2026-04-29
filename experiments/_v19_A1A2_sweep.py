"""
v19 A1 (family-consensus boost on emission) + A2 (hard tactic filter) sweep.

"""
from __future__ import annotations
import json, sys, copy
from collections import Counter
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


def apply_family_boost(sorted_results: list, boost: float, width: int) -> list:
    """In-place family-consensus reorder of each group's similar_techniques."""
    if boost <= 0 or width <= 0:
        return sorted_results
    out = []
    for r in sorted_results:
        r = copy.deepcopy(r)
        cands = r.get("similar_techniques", [])
        if len(cands) >= 2:
            wide = min(width, len(cands))
            parents = [c["technique_id"].split(".", 1)[0] for c in cands[:wide]]
            pcount = Counter(parents)
            scored = []
            for i, c in enumerate(cands[:wide]):
                shared = pcount[parents[i]] - 1
                boosted = c.get("p_ttp", c.get("similarity", 0)) * (1.0 + boost * shared)
                scored.append((boosted, i, c))
            scored.sort(key=lambda x: -x[0])
            reranked = [c for _, _, c in scored] + cands[wide:]
            for nr, c in enumerate(reranked, 1):
                c["rank"] = nr
            r["similar_techniques"] = reranked
            if len(reranked) >= 2:
                r["confidence_margin"] = float(
                    reranked[0].get("p_ttp", 0) - reranked[1].get("p_ttp", 0)
                )
        out.append(r)
    return out


def run_viterbi_all(use_a1: bool, use_a2: bool, fam_boost: float = 0.15):
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
        if use_a1:
            sorted_res = apply_family_boost(sorted_res, fam_boost, 10)
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
            hard_tactic_filter=use_a2,
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


# (name, use_a1, use_a2, fam_boost)
VARIANTS = [
    ("baseline_v18β",     False, False, 0.0),
    ("A1_only_b0.10",     True,  False, 0.10),
    ("A1_only_b0.15",     True,  False, 0.15),
    ("A1_only_b0.20",     True,  False, 0.20),
    ("A2_only",           False, True,  0.0),
    ("A1+A2_b0.10",       True,  True,  0.10),
    ("A1+A2_b0.15",       True,  True,  0.15),
    ("A1+A2_b0.20",       True,  True,  0.20),
]


def main():
    results = []
    for name, a1, a2, fb in VARIANTS:
        print(f"\n═══ {name}  A1={a1}(b={fb})  A2={a2} ═══")
        run_viterbi_all(a1, a2, fb)
        eval_post_run(strong_only=False, label="ALL TPs")
        eval_v2()
        m = read_metrics()
        print(f"  tech_lcs={m['tech_lcs']:.4f}  tac_lcs={m['tac_lcs']:.4f}  "
              f"step_cov={m['step_cov']:.4f}  order={m['order']:.4f}  "
              f"vit_mic={m['viterbi_mic']:.4f}  faiss_mic={m['faiss_mic']:.4f}")
        results.append({"variant": name, "use_a1": a1, "use_a2": a2,
                        "family_boost": fb, **m})

    out = config.OUTPUT_BASE_DIR / "v19_A1A2_sweep_results.json"
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
