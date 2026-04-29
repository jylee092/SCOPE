"""
Q2 robustness -- random log-drop experiment.

For each scenario × drop_rate ∈ {0.10, 0.25, 0.50} × seed:
  1. Load the raw JSONL scenario file.
  2. Randomly sample (1 - p) fraction of lines (deterministic per seed).
  3. Write to output/_robustness/<drop>/<seed>/<rel>/<stem>.json.
  4. Run each method's adapter on the dropped scenario.
  5. Score chain alignment vs attack_flows reference.

The "0%" column in tab:robust is the already-measured value from the main
comparison; this script only fills the 10/25/50% columns.

Heavy stage is SCOPE + SHIELD (both call the LLM). We sequence them so the
API rate-limit window is respected. Sigma / MAGIC / DeepAG are I/O-bound
and run quickly.

Adaptation note for SCOPE: VITERBI_MAX_SKIP is overridden to 2 during
robustness runs so the hole-bridging operator described in the paper §5.5
is engaged (the default config has it disabled).
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

ROB_DIR = config.OUTPUT_BASE_DIR / "_robustness"


def _drop_path(drop: float, seed: int, rel: Path) -> Path:
    """output/_robustness/dropP/seedS/<rel>/<stem>.json"""
    pct = int(round(drop * 100))
    return ROB_DIR / f"drop{pct}" / f"seed{seed}" / rel.with_suffix(".json")


def make_dropped_scenarios(drop_rates: list[float], seeds: list[int],
                            ) -> list[tuple[float, int, Path, Path]]:
    """Pre-generate dropped JSONL files. Returns list of
    (drop, seed, original_path, dropped_path) tuples for downstream runs."""
    out = []
    originals = sorted(config.DATASET_FOLDER.rglob("*.json"))
    for drop in drop_rates:
        for seed in seeds:
            rng = random.Random((seed << 16) ^ int(drop * 1000))
            for orig in originals:
                rel = orig.relative_to(config.DATASET_FOLDER).with_suffix("")
                dst = _drop_path(drop, seed, rel)
                if dst.exists():
                    out.append((drop, seed, orig, dst))
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(orig, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                kept = [l for l in lines
                        if rng.random() >= drop]
                if not kept:
                    # avoid empty scenarios -- keep at least 1 event
                    kept = [lines[0]] if lines else []
                with open(dst, "w", encoding="utf-8") as f:
                    f.writelines(kept)
                out.append((drop, seed, orig, dst))
    return out


# ---------------------------------------------------------------------------
# Method runners -- each returns dict {(scenario, drop, seed): tech_lcs}
# ---------------------------------------------------------------------------

def _score_alerts_chain_lcs(result_path: Path) -> tuple[str, float] | None:
    """Inline chain-LCS scoring for any baseline result.json that follows the
    BaselinePrediction shape (alerts list with topk_tids/topk_tactics)."""
    from experiments.attack_flows import get_flow
    from experiments.chain_align  import evaluate_chain_alignment

    try:
        with open(result_path, encoding="utf-8") as f:
            pred = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    scenario = pred.get("scenario") or result_path.stem
    flow = get_flow(scenario)
    if not flow:
        return None
    alerts = pred.get("notes", {}).get("alerts", []) or []
    if "tactic_sequence" in pred and "technique_sequence" in pred:
        # Use the adapter-emitted dedup sequences directly.
        breakdown = []
        last = (None, None)
        for tid, tac in zip(pred.get("technique_sequence") or [],
                              pred.get("tactic_sequence") or []):
            if (tid, tac) == last:
                continue
            breakdown.append({"technique_id": tid, "tactic": tac})
            last = (tid, tac)
        if not breakdown:
            for a in alerts:
                tids = a.get("topk_tids") or []
                tacs = a.get("topk_tactics") or []
                for i, tid in enumerate(tids):
                    tac = tacs[i] if i < len(tacs) else ""
                    if (tid, tac) == last:
                        continue
                    breakdown.append({"technique_id": tid, "tactic": tac})
                    last = (tid, tac)
    else:
        breakdown = []
    if not breakdown:
        return scenario, 0.0
    chain = evaluate_chain_alignment(scenario, breakdown, ref_flow=flow)
    return scenario, chain.get("technique_lcs_norm") or 0.0


def _run_baseline_adapter(drop: float, seed: int, *,
                           method_dir: str, adapter_factory):
    """Common driver: predict → save → inline-score."""
    pct = int(round(drop * 100))
    out_root = config.OUTPUT_BASE_DIR / "_robustness" / f"{method_dir}_drop{pct}_seed{seed}"
    adapter = adapter_factory()
    out: dict[str, float] = {}
    for orig in sorted(config.DATASET_FOLDER.rglob("*.json")):
        rel = orig.relative_to(config.DATASET_FOLDER).with_suffix("")
        dropped = _drop_path(drop, seed, rel)
        if not dropped.exists():
            continue
        out_dir = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "result.json"
        if not result_path.exists():
            pred = adapter.predict(dropped)
            adapter.save_result(pred, out_dir)
        scored = _score_alerts_chain_lcs(result_path)
        if scored:
            out[scored[0]] = scored[1]
    return out


def run_sigma(drop: float, seed: int) -> dict[str, float]:
    from experiments.baselines.sigma.adapter import SigmaAdapter
    return _run_baseline_adapter(drop, seed,
                                   method_dir="sigma", adapter_factory=SigmaAdapter)


def run_magic(drop: float, seed: int) -> dict[str, float]:
    """MAGIC needs the Sigma drop-specific stream as its TID classifier."""
    from experiments.baselines.magic.adapter import MagicAdapter
    from experiments.baselines.magic import adapter as magic_adapter_mod

    pct = int(round(drop * 100))
    sigma_root = config.OUTPUT_BASE_DIR / "_robustness" / f"sigma_drop{pct}_seed{seed}"
    drop_root  = config.OUTPUT_BASE_DIR / "_robustness" / f"drop{pct}" / f"seed{seed}"

    orig_load = magic_adapter_mod._load_sigma
    def _drop_load(scenario_stem: str, scenario_path):
        # The dropped scenario lives under drop_root; rebuild rel from there
        try:
            rel = scenario_path.relative_to(drop_root).with_suffix("")
        except ValueError:
            return None
        cand = sigma_root / rel / "result.json"
        if cand.exists():
            try:
                with open(cand, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    magic_adapter_mod._load_sigma = _drop_load
    try:
        return _run_baseline_adapter(drop, seed,
                                       method_dir="magic",
                                       adapter_factory=MagicAdapter)
    finally:
        magic_adapter_mod._load_sigma = orig_load


def run_deepag(drop: float, seed: int) -> dict[str, float]:
    """DeepAG depends on Sigma's drop-specific output stream."""
    from experiments.baselines.ttp_sequence.adapter import DeepAGAdapter
    from experiments.baselines.ttp_sequence import adapter as deepag_adapter_mod

    pct = int(round(drop * 100))
    sigma_root = config.OUTPUT_BASE_DIR / "_robustness" / f"sigma_drop{pct}_seed{seed}"
    drop_root  = config.OUTPUT_BASE_DIR / "_robustness" / f"drop{pct}" / f"seed{seed}"

    orig_find = deepag_adapter_mod._find_sigma_result
    def _find_drop(scenario_path):
        try:
            rel = scenario_path.relative_to(drop_root).with_suffix("")
        except ValueError:
            return None
        cand = sigma_root / rel / "result.json"
        return cand if cand.exists() else None
    deepag_adapter_mod._find_sigma_result = _find_drop
    try:
        return _run_baseline_adapter(drop, seed,
                                       method_dir="deepag",
                                       adapter_factory=DeepAGAdapter)
    finally:
        deepag_adapter_mod._find_sigma_result = orig_find


def run_shield(drop: float, seed: int) -> dict[str, float]:
    from experiments.baselines.shield.adapter import ShieldAdapter
    return _run_baseline_adapter(drop, seed,
                                   method_dir="shield",
                                   adapter_factory=ShieldAdapter)


def run_scope(drop: float, seed: int) -> dict[str, float]:
    """SCOPE -- full pipeline with VITERBI_MAX_SKIP=2 (hole-bridging).
    Full LLM mode: cache hits use cached descriptions, misses call Gemini."""
    # We use main.py-equivalent steps inline so we can write to a custom dir.
    from pipeline.data_loader       import load_and_normalize
    from pipeline.rule_matcher      import (
        load_rules, run_grouping, merge_shared_supporting,
    )
    from pipeline.feature_extractor import extract_all
    from pipeline.feature_sanitizer import sanitize
    from pipeline.mitre_mapper      import analyze
    from pipeline.attack_chain      import (
        sort_results_by_time, load_tactic_map, build_group_nodes,
        TacticalScorer, get_semantic_scorer, CausalScorer,
        MultiDimTransitionScorer, load_campaign_library,
        topk_viterbi, apply_emission_confidence_bypass,
    )
    from pipeline.technique_io      import load_or_build_technique_io
    from experiments.attack_flows   import get_flow
    from experiments.chain_align    import evaluate_chain_alignment

    pct = int(round(drop * 100))
    out_root = config.OUTPUT_BASE_DIR / "_robustness" / f"scope_drop{pct}_seed{seed}"

    # Override VITERBI_MAX_SKIP for this run; keep other config as-is.
    prev_max_skip = config.VITERBI_MAX_SKIP
    config.VITERBI_MAX_SKIP = 2

    try:
        out: dict[str, float] = {}
        tactic_map = load_tactic_map(str(config.MITRE_CSV_PATH))
        rule_list  = load_rules(config.RULE_FOLDER)
        campaigns  = load_campaign_library(str(config.CAMPAIGN_FOLDER), tactic_map)
        technique_io = load_or_build_technique_io(
            str(config.MITRE_CSV_PATH),
            cache_path=config.OUTPUT_BASE_DIR / "technique_io_cache.json",
        )
        sem = get_semantic_scorer(
            getattr(config, "SEMANTIC_MODEL", config.CROSS_ENCODER_MODEL),
            backend=getattr(config, "SEMANTIC_BACKEND", "cross-encoder"),
            calibration=getattr(config, "SEM_CALIBRATION", "linear"),
        ) if config.USE_SEMANTIC_SCORING else None
        cau = CausalScorer(technique_io=technique_io) if config.USE_CAUSAL_SCORING else None

        for orig in sorted(config.DATASET_FOLDER.rglob("*.json")):
            rel = orig.relative_to(config.DATASET_FOLDER).with_suffix("")
            stem = orig.stem
            dropped = _drop_path(drop, seed, rel)
            if not dropped.exists():
                continue
            out_dir = out_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            vit_path = out_dir / f"{stem}_viterbi.json"
            if not vit_path.exists():
                df = load_and_normalize(str(dropped))
                groups = run_grouping(
                    df=df, rule_list=rule_list,
                    before_sec=config.GROUPING_BEFORE_SEC,
                    after_sec=config.GROUPING_AFTER_SEC,
                    hop_up=config.GROUPING_HOP_UP,
                    hop_down=config.GROUPING_HOP_DOWN,
                    apply_filters=config.GROUPING_APPLY_FILTER,
                    use_shared_entity=config.GROUPING_USE_SHARED_ENTITY,
                )
                groups = merge_shared_supporting(
                    groups, df,
                    overlap_threshold=config.MERGE_OVERLAP_THRESHOLD,
                )
                if not groups:
                    with open(vit_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
                    out[stem] = 0.0
                    continue
                feats = extract_all(groups, df)
                feats_sanitized = [sanitize(f) for f in feats]
                ttp = analyze(
                    feats_sanitized,
                    str(config.MITRE_CSV_PATH),
                    config.GEMINI_API_KEY,
                    cache_dir=config.CACHE_DIR,
                    use_llm=True,
                )
                sorted_results = sort_results_by_time(ttp, df)
                features_by_gid = {f["group_id"]: f for f in feats}
                group_nodes = build_group_nodes(sorted_results, tactic_map,
                                                  features_by_gid)
                tac_scorer = TacticalScorer(
                    anomaly_threshold=config.TACTIC_ANOMALY_THRESHOLD)
                multi = MultiDimTransitionScorer(
                    tac_scorer=tac_scorer, sem_scorer=sem, cau_scorer=cau,
                    w_tac=config.W_TAC, w_sem=config.W_SEM, w_cau=config.W_CAU,
                    self_loop_tid_penalty=getattr(
                        config, "SELF_LOOP_TID_PENALTY", 1.0),
                )
                vit = topk_viterbi(
                    group_nodes, multi,
                    beam_k=config.VITERBI_BEAM_K,
                    max_skip=config.VITERBI_MAX_SKIP,
                    skip_penalty=config.VITERBI_SKIP_PENALTY,
                    transition_weight=config.VITERBI_TRANSITION_WEIGHT,
                    campaigns=campaigns,
                )
                bypass_thr = getattr(
                    config, "EMISSION_BYPASS_SIM_THRESHOLD", None)
                if bypass_thr is not None:
                    vit = apply_emission_confidence_bypass(
                        vit, multi, sim_threshold=float(bypass_thr))
                with open(vit_path, "w", encoding="utf-8") as f:
                    json.dump(vit.score_breakdown, f, ensure_ascii=False, indent=2)

            # score
            with open(vit_path, encoding="utf-8") as f:
                breakdown = json.load(f)
            flow = get_flow(stem)
            if flow and breakdown:
                chain = evaluate_chain_alignment(stem, breakdown, ref_flow=flow)
                out[stem] = chain.get("technique_lcs_norm", 0.0)
            else:
                out[stem] = 0.0
        return out
    finally:
        config.VITERBI_MAX_SKIP = prev_max_skip


# ---------------------------------------------------------------------------

METHODS = [
    ("Sigma",  run_sigma,  "fast"),
    ("MAGIC",  run_magic,  "fast"),
    ("DeepAG", run_deepag, "fast"),
    ("SHIELD", run_shield, "slow"),    # LLM
    ("SCOPE",  run_scope,  "slow"),    # LLM
]

DROP_RATES = [0.10, 0.25, 0.50]
SEEDS      = [0]                       # start with one seed; expand if time permits


def main(methods: list[str] | None = None) -> None:
    print(f"[robust] generating dropped scenarios for "
          f"drops={DROP_RATES}, seeds={SEEDS} ...")
    pairs = make_dropped_scenarios(DROP_RATES, SEEDS)
    print(f"[robust] {len(pairs)} (drop, seed, scenario) triples ready")

    summary: dict[str, dict[float, list[float]]] = {}
    methods = methods or [m[0] for m in METHODS]
    for name, runner, kind in METHODS:
        if name not in methods:
            continue
        summary[name] = {}
        for drop in DROP_RATES:
            per_drop = []
            for seed in SEEDS:
                print(f"\n=== {name} drop={drop:.2f} seed={seed} ({kind}) ===")
                t0 = time.time()
                lcs_by_scen = runner(drop, seed)
                if lcs_by_scen:
                    per_drop.append(mean(lcs_by_scen.values()))
                print(f"  → tech-LCS mean = {per_drop[-1]:.4f} "
                      f"({time.time()-t0:.1f}s, {len(lcs_by_scen)} scen)")
            summary[name][drop] = per_drop

    out = config.OUTPUT_BASE_DIR / "_robustness_scores.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {str(d): v for d, v in dv.items()}
                   for k, dv in summary.items()}, f, indent=2)
    print("\n--- Summary ---")
    for name, drops in summary.items():
        line = [f"{name:<8}"]
        for d in DROP_RATES:
            vals = drops.get(d) or []
            line.append(f"  {d*100:.0f}%={mean(vals):.3f}" if vals else f"  {d*100:.0f}%=NA")
        print("".join(line))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=None,
                    choices=[m[0] for m in METHODS])
    args = ap.parse_args()
    main(methods=args.methods)
