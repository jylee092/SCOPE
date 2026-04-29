"""
Q5 — SCOPE end-to-end timing harness.

Times each pipeline stage on a representative sample of scenarios and
reports throughput (events/sec), per-stage breakdown, and peak resident
memory (psutil). The cache is left hot so timings reflect production use.
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

# Pipeline imports
from pipeline.data_loader       import load_and_normalize
from pipeline.rule_matcher      import (
    load_rules, run_grouping, merge_same_anchor, merge_shared_supporting,
)
from pipeline.feature_extractor import extract_all
from pipeline.feature_sanitizer import sanitize
from pipeline.mitre_mapper      import analyze
from pipeline.attack_chain      import (
    sort_results_by_time, load_tactic_map, build_group_nodes,
    TacticalScorer, get_semantic_scorer, CausalScorer,
    MultiDimTransitionScorer, load_campaign_library, topk_viterbi,
    apply_emission_confidence_bypass,
)
from pipeline.technique_io      import load_or_build_technique_io


import psutil
_proc = psutil.Process(os.getpid())


class PeakMemoryMonitor:
    """Sample RSS at fixed intervals on a background thread."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.peak_mb = _proc.memory_info().rss / (1024 * 1024)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False

    def _run(self):
        while not self._stop.is_set():
            rss_mb = _proc.memory_info().rss / (1024 * 1024)
            if rss_mb > self.peak_mb:
                self.peak_mb = rss_mb
            self._stop.wait(self.interval)


def time_scenario(ds_path: Path) -> dict:
    config.configure_dataset(ds_path)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stages: dict[str, float] = {}
    counts: dict[str, int] = {}

    with PeakMemoryMonitor(interval=0.05) as mem:
        # 1. Load + normalize
        t0 = time.perf_counter()
        df = load_and_normalize(str(config.DATASET_FILE))
        stages["load"] = time.perf_counter() - t0
        counts["events"] = int(len(df))

        # 2. Grouping
        t0 = time.perf_counter()
        rule_list = load_rules(config.RULE_FOLDER)
        groups = run_grouping(
            df=df, rule_list=rule_list,
            before_sec=config.GROUPING_BEFORE_SEC,
            after_sec=config.GROUPING_AFTER_SEC,
            hop_up=config.GROUPING_HOP_UP,
            hop_down=config.GROUPING_HOP_DOWN,
            apply_filters=config.GROUPING_APPLY_FILTER,
            use_shared_entity=config.GROUPING_USE_SHARED_ENTITY,
            max_anchors_per_rule=config.GROUPING_MAX_ANCHORS_PER_RULE,
        )
        groups = merge_same_anchor(groups)
        groups = merge_shared_supporting(
            groups, df, overlap_threshold=config.MERGE_OVERLAP_THRESHOLD,
        )
        if getattr(config, "DROP_FILTER_FAILED_GROUPS", False):
            groups = [g for g in groups if g.get("filter_passed", True)]
        # group cap
        if len(groups) > config.MAX_GROUPS_PER_SCENARIO:
            by_tid: dict[str, list] = defaultdict(list)
            for g in groups:
                by_tid[g["technique_id"]].append(g)
            for tid in by_tid:
                by_tid[tid].sort(key=lambda g: g.get("confidence", 0), reverse=True)
            truncated: list = []
            i = 0
            while len(truncated) < config.MAX_GROUPS_PER_SCENARIO:
                added = False
                for tid, glist in by_tid.items():
                    if i < len(glist):
                        truncated.append(glist[i])
                        added = True
                        if len(truncated) >= config.MAX_GROUPS_PER_SCENARIO:
                            break
                if not added:
                    break
                i += 1
            groups = truncated
        stages["grouping"] = time.perf_counter() - t0
        counts["groups"] = len(groups)

        # 3. Feature extraction
        t0 = time.perf_counter()
        feats = extract_all(groups, df)
        feats_san = [sanitize(f) for f in feats]
        stages["feature"] = time.perf_counter() - t0

        # 4. Mapping (LLM description + retrieval)
        t0 = time.perf_counter()
        # Sub-sample by per-TID cap (matches main.py)
        cap = config.SAMPLE_PER_TECHNIQUE
        if cap <= 0:
            sampled = list(feats_san)
        else:
            sampled = []
            count_by_tid: dict[str, int] = defaultdict(int)
            for f in feats_san:
                tid = f["technique_id"]
                if count_by_tid[tid] < cap:
                    sampled.append(f)
                    count_by_tid[tid] += 1
        ce_for_rerank = None
        if getattr(config, "USE_CE_EMISSION_RERANK", True):
            sem_scorer_for_ce = get_semantic_scorer(
                getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
                backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
            )
            ce_for_rerank = getattr(sem_scorer_for_ce, "_model", None)
        results = analyze(
            sampled,
            str(config.MITRE_CSV_PATH),
            config.GEMINI_API_KEY,
            cache_dir=config.CACHE_DIR,
            cross_encoder=ce_for_rerank,
            ce_rerank_width=getattr(config, "CE_RERANK_WIDTH", 20),
            ce_weight=getattr(config, "CE_RERANK_WEIGHT", 0.0),
            bm25_weight=getattr(config, "BM25_WEIGHT", 0.3),
            bm25_rerank_width=getattr(config, "BM25_RERANK_WIDTH", 30),
            tid_prior=getattr(config, "RULE_TID_PRIOR", 1.15),
            tactic_prior=getattr(config, "RULE_TACTIC_PRIOR", 1.05),
            signature_weight=getattr(config, "SIGNATURE_WEIGHT", 0.0),
            signature_rerank_width=getattr(config, "SIGNATURE_RERANK_WIDTH", 10),
            family_boost=getattr(config, "FAMILY_BOOST", 0.0),
            family_boost_width=getattr(config, "FAMILY_BOOST_WIDTH", 10),
        )
        stages["mapping"] = time.perf_counter() - t0
        counts["mapped_groups"] = len(results)

        # 5. Viterbi (transition + Top-K + bypass)
        t0 = time.perf_counter()
        sorted_results = sort_results_by_time(results, df)
        tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
        features_by_gid = {f["group_id"]: f for f in feats}
        group_nodes = build_group_nodes(sorted_results, tactic_map, features_by_gid)
        tac_scorer = TacticalScorer(anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
        sem_scorer = None
        if config.USE_SEMANTIC_SCORING:
            sem_scorer = get_semantic_scorer(
                getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
                backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
            )
        cau_scorer = None
        if config.USE_CAUSAL_SCORING:
            technique_io = load_or_build_technique_io(
                str(config.MITRE_CSV_PATH),
                cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
            )
            cau_scorer = CausalScorer(technique_io=technique_io)
        multi_scorer = MultiDimTransitionScorer(
            tac_scorer=tac_scorer, sem_scorer=sem_scorer, cau_scorer=cau_scorer,
            w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
            self_loop_tid_penalty=getattr(config, "SELF_LOOP_TID_PENALTY", 1.0),
        )
        campaigns = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
        vit = topk_viterbi(
            group_nodes, multi_scorer,
            beam_k=config.VITERBI_BEAM_K,
            max_skip=config.VITERBI_MAX_SKIP,
            skip_penalty=config.VITERBI_SKIP_PENALTY,
            transition_weight=config.VITERBI_TRANSITION_WEIGHT,
            campaigns=campaigns,
            hard_tactic_filter=getattr(config, "VITERBI_HARD_TACTIC_FILTER", False),
        )
        thr = getattr(config, "EMISSION_BYPASS_SIM_THRESHOLD", None)
        if thr is not None:
            vit = apply_emission_confidence_bypass(
                vit, multi_scorer, sim_threshold=float(thr)
            )
        stages["viterbi"] = time.perf_counter() - t0
        counts["chain_len"] = len(vit.score_breakdown) if vit else 0

    total = sum(stages.values())
    return {
        "scenario": config.DATASET_NAME,
        "events": counts["events"],
        "groups": counts.get("groups", 0),
        "chain_len": counts.get("chain_len", 0),
        "stages": stages,
        "total_sec": total,
        "events_per_sec": counts["events"] / total if total > 0 else 0.0,
        "peak_mem_mb": mem.peak_mb,
    }


def main():
    # Full 35-scenario corpus to match the SHIELD timing's denominator
    # so the Q5 comparison (events/sec aggregate) is on the same surface.
    paths = sorted(config.DATASET_FOLDER.rglob("*.json"))

    rows: list[dict] = []
    for p in paths:
        print(f"\n[timing] {p.name}")
        try:
            r = time_scenario(p)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            continue
        rows.append(r)
        s = r["stages"]
        print(f"  events={r['events']}  groups={r['groups']}  chain={r['chain_len']}")
        print(f"  load={s['load']:.2f}s  group={s['grouping']:.2f}s  "
              f"feat={s['feature']:.2f}s  map={s['mapping']:.2f}s  "
              f"vit={s['viterbi']:.2f}s")
        print(f"  total={r['total_sec']:.2f}s  ({r['events_per_sec']:.1f} ev/s)  "
              f"peak_mem={r['peak_mem_mb']:.0f} MB")

    if not rows:
        print("no rows")
        return
    out = ROOT / "output" / "_q5_scope_timings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Aggregate
    total_ev   = sum(r["events"] for r in rows)
    total_t    = sum(r["total_sec"] for r in rows)
    stage_tot  = defaultdict(float)
    for r in rows:
        for k, v in r["stages"].items():
            stage_tot[k] += v
    stage_pct  = {k: 100 * v / total_t for k, v in stage_tot.items()} if total_t else {}

    valid_eps  = [r["events_per_sec"] for r in rows if r["events_per_sec"] > 0]
    median_eps = sorted(valid_eps)[len(valid_eps) // 2] if valid_eps else 0
    peak_mem   = max(r["peak_mem_mb"] for r in rows)

    print("\n" + "=" * 70)
    print("Q5 SCOPE timing summary")
    print("=" * 70)
    print(f"Scenarios timed : {len(rows)}")
    print(f"Total events    : {total_ev}")
    print(f"Total wall-clock: {total_t:.1f}s")
    print(f"Aggregate ev/s  : {total_ev / total_t:.1f}")
    print(f"Median ev/s     : {median_eps:.1f}")
    print(f"Peak memory     : {peak_mem:.0f} MB ({peak_mem/1024:.2f} GB)")
    print()
    print("Stage breakdown (sum across runs):")
    for k, v in sorted(stage_tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v:>7.1f}s  ({stage_pct[k]:.1f}%)")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
