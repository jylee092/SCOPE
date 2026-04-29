"""
v17 A+B ablation sweep.

A = per-tactic self-loop + per-target wildcard-IN weight (TacticalScorer)
B = fine-grained entity types + graded overlap (CausalScorer + technique_io)

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


def make_tac_scorer(use_a: bool) -> TacticalScorer:
    t = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
    if not use_a:
        t._self_loop_w = {}
        t._wildcard_in_w = {}
    return t


def make_cau_scorer(use_b_io: bool, use_b_graded: bool) -> CausalScorer:
    """Causal scorer with B ...

    use_b_graded=False → exact overlap (legacy). monkey-patch score()
    """
    cache_fp = (config.OUTPUT_BASE_DIR / "technique_io_cache.json"
                if use_b_io else
                config.OUTPUT_BASE_DIR / "technique_io_cache.v16.json.bak")
    io = load_or_build_technique_io(str(config.MITRE_CSV_PATH), cache_path=cache_fp)
    cau = CausalScorer(technique_io=io)
    if not use_b_graded:
        _EPS_LOCAL = 1e-6
        def legacy_score(self, tech_i, tactic_i, tech_j, tactic_j, entities_i, entities_j):
            _, out_i = self._get_io(tech_i, tactic_i)
            in_j, _  = self._get_io(tech_j, tactic_j)
            shared_observed = entities_i & entities_j
            overlap = out_i & in_j & shared_observed
            return len(overlap) / (len(in_j) + _EPS_LOCAL)
        import types as _t
        cau.score = _t.MethodType(legacy_score, cau)
    return cau


def run_viterbi_all(use_a: bool, use_b_io: bool, use_b_graded: bool, use_b_entity: bool):
    """Run Viterbi over all 35 scenarios with given variant flags.

    """
    _SEM_SCORER_CACHE.clear()
    tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
    sem = get_semantic_scorer(
        getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
        backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
        calibration=getattr(config, "SEM_CALIBRATION", "linear"),
        sigmoid_center=getattr(config, "SEM_SIGMOID_CENTER", 0.5),
        sigmoid_scale=getattr(config, "SEM_SIGMOID_SCALE", 8.0),
    ) if config.USE_SEMANTIC_SCORING else None
    tac = make_tac_scorer(use_a)
    cau = make_cau_scorer(use_b_io, use_b_graded) if config.USE_CAUSAL_SCORING else None
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
        # Optionally coarsify entity types to match legacy 6-type taxonomy.
        # fine → legacy coarse mapping (proc→process, reg→registry, net→network,
        # svc/task/driver→service, pipe→process).
        _COARSE = {
            "proc": "process", "file": "file", "reg": "registry",
            "net": "network", "user": "user", "svc": "service",
            "task": "service", "driver": "service", "pipe": "process",
        }
        if not use_b_entity:
            for n in nodes:
                new_set = set()
                for e in n.entity_types:
                    head = e.split(".", 1)[0] if "." in e else e
                    new_set.add(_COARSE.get(head, head))
                n.entity_types = new_set
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


# (A, B_io, B_graded, B_entity)
VARIANTS = [
    ("baseline_v15",  False, False, False, False),

    ("A_only",        True,  False, False, False),

    ("B_io_only",     False, True,  False, False),

    ("B_graded_only", False, False, True,  False),

    ("B_entity_only", False, False, False, True),

    ("B_full",        False, True,  True,  True),

    ("AB_full",       True,  True,  True,  True),   # A+B
]


def main():
    results = []
    for name, a, b_io, b_gr, b_ent in VARIANTS:
        print(f"\n═══ {name}  A={a}  B_io={b_io}  B_graded={b_gr}  B_entity={b_ent} ═══")
        run_viterbi_all(a, b_io, b_gr, b_ent)
        eval_post_run(strong_only=False, label="ALL TPs")
        eval_v2()
        m = read_metrics()
        print(f"  tech_lcs={m['tech_lcs']:.4f}  tac_lcs={m['tac_lcs']:.4f}  "
              f"step_cov={m['step_cov']:.4f}  order={m['order']:.4f}  "
              f"vit_mic={m['viterbi_mic']:.4f}  faiss_mic={m['faiss_mic']:.4f}")
        results.append({"variant": name, "use_a": a, "use_b_io": b_io,
                        "use_b_graded": b_gr, "use_b_entity": b_ent, **m})

    out = config.OUTPUT_BASE_DIR / "v17_ab_sweep_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")
    print("\n" + "=" * 110)
    print(f"{'variant':<20} {'tech':>8} {'tac':>8} {'step':>8} {'order':>8} {'vit_mic':>10} {'faiss_mic':>10}")
    for r in results:
        print(f"{r['variant']:<20} {r['tech_lcs']:>8.4f} {r['tac_lcs']:>8.4f} "
              f"{r['step_cov']:>8.4f} {r['order']:>8.4f} "
              f"{r['viterbi_mic']:>10.4f} {r['faiss_mic']:>10.4f}")


if __name__ == "__main__":
    main()
