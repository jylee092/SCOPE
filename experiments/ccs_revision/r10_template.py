"""
R10 (CCS reviewer ⑩): LLM-swap ablation -- TEMPLATE (no-LLM) variant.

Regenerates the mapping + Viterbi chain for every scenario with the LLM
description replaced by the deterministic template text (feature_to_text),
holding EVERYTHING else fixed (same groups, same ATTACK-BERT embedding, same
search/rerank, same Viterbi config incl. v20 emission bypass).

Isolation: all outputs go under output/_ccs_revision/R10_llm_swap/template/
via a temporary OUTPUT_BASE_DIR swap. The canonical output/ and the Gemini
LLM cache are never written. Upstream features are READ from canonical
<stem>_feature_result.json (grouping is LLM-independent, so it is reused).

Run:  python -m experiments.ccs_revision.r10_template
"""
from __future__ import annotations
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from pipeline.feature_sanitizer import sanitize
from pipeline.mitre_mapper import analyze
from experiments._rerun_viterbi_only import run_one

CANON = config.OUTPUT_BASE_DIR
R10 = CANON / "_ccs_revision" / "R10_llm_swap" / "template"


def sample_features(all_features):
    """Reproduce main.py: sanitize, then cap per technique_id."""
    sani = [sanitize(f) for f in all_features]
    cap = config.SAMPLE_PER_TECHNIQUE
    if cap <= 0:
        return sani
    out, cnt = [], defaultdict(int)
    for f in sani:
        t = f["technique_id"]
        if cnt[t] < cap:
            out.append(f)
            cnt[t] += 1
    return out


def map_one_template(all_features):
    """analyze() with use_llm=False, same params as main.py run_pipeline."""
    sampled = sample_features(all_features)
    return analyze(
        sampled,
        str(config.MITRE_CSV_PATH),
        "",                       # no api key needed for template
        cache_dir=config.CACHE_DIR,   # canonical FAISS index (read-only); no LLM cache write
        use_llm=False,
        cross_encoder=None,           # config.USE_CE_EMISSION_RERANK is False
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


def main():
    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    done, skipped = 0, 0
    try:
        for i, ds in enumerate(datasets, 1):
            # --- read canonical inputs (features + finale csv) ---
            config.OUTPUT_BASE_DIR = CANON
            config.configure_dataset(ds)
            feat_fp = config.FEATURE_RESULT_JSON_PATH
            csv_fp = config.FINALE_CSV_PATH
            name = config.DATASET_NAME
            if not (feat_fp.exists() and csv_fp.exists()):
                print(f"[{i}/{len(datasets)}] skip {name}: missing canonical inputs")
                skipped += 1
                continue
            all_features = json.load(open(feat_fp, encoding="utf-8"))

            # resume: skip scenarios whose R10 viterbi already exists
            config.OUTPUT_BASE_DIR = R10
            config.configure_dataset(ds)
            if config.VITERBI_JSON_PATH.exists():
                print(f"[{i}/{len(datasets)}] skip {name}: already done")
                config.OUTPUT_BASE_DIR = CANON
                done += 1
                continue
            config.OUTPUT_BASE_DIR = CANON
            config.configure_dataset(ds)

            print(f"\n[{i}/{len(datasets)}] {name}: template mapping ({len(all_features)} groups)")
            results = map_one_template(all_features)

            # --- stage inputs + write template mapping under R10 ---
            config.OUTPUT_BASE_DIR = R10
            config.configure_dataset(ds)
            config.TTP_MAPPING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            json.dump(all_features, open(config.FEATURE_RESULT_JSON_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            shutil.copy(csv_fp, config.FINALE_CSV_PATH)
            json.dump(results, open(config.TTP_MAPPING_JSON_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

            # --- Viterbi (reads R10 paths since OUTPUT_BASE_DIR=R10) ---
            ok = run_one(ds)
            done += 1 if ok else 0
    finally:
        config.OUTPUT_BASE_DIR = CANON

    print(f"\n[R10/template] done={done} skipped={skipped} -> {R10}")


if __name__ == "__main__":
    main()
