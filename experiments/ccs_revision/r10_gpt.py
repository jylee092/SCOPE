"""
R10 (CCS reviewer ⑩): LLM-swap ablation -- GPT (OpenAI) variant.

Same as r10_template but the description generator is OpenAI GPT instead of
Gemini. EVERYTHING else is held fixed (same groups, same build_prompt, same
ATTACK-BERT embedding, same search/rerank, same Viterbi config). This answers
"is the result Gemini-specific, or does any capable LLM recover it?".

Mechanism: monkeypatch mitre_mapper.generate_description_cached with a GPT
version, then reuse analyze(use_llm=True) unchanged. GPT descriptions are
cached in an ISOLATED file (keyed by model+prompt) so reruns cost nothing and
the canonical Gemini cache is never touched.

Key (in priority order):  env OPENAI_API_KEY  ->  Final_Code/.secrets/openai_key.txt
Model:                     env R10_GPT_MODEL   ->  default below

Run:   python -m experiments.ccs_revision.r10_gpt
List:  python -m experiments.ccs_revision.r10_gpt --list-models
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
import pipeline.mitre_mapper as mm
from pipeline.feature_sanitizer import sanitize
from pipeline.mitre_mapper import analyze, build_prompt
from experiments._rerun_viterbi_only import run_one

CANON = config.OUTPUT_BASE_DIR
R10 = CANON / "_ccs_revision" / "R10_llm_swap" / "gpt"
GPT_CACHE = R10 / "_cache" / "gpt_descriptions.json"
DEFAULT_MODEL = "gpt-4o-mini"   # flash/mini tier ~ gemini-2.5-flash; override via R10_GPT_MODEL
MODEL = os.environ.get("R10_GPT_MODEL", DEFAULT_MODEL)

_TIMEOUT = 60
_MAX_RETRIES = 6
_client = None
_gpt_cache: dict | None = None


def load_key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return k
    f = ROOT / ".secrets" / "openai_key.txt"
    if f.exists():
        k = f.read_text(encoding="utf-8").strip()
        if k:
            return k
    raise SystemExit(
        "No OpenAI key found.\n"
        "  Provide it one of these ways:\n"
        "  (a) create Final_Code/.secrets/openai_key.txt containing only the key, or\n"
        "  (b) set the OPENAI_API_KEY environment variable."
    )


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=load_key(), timeout=_TIMEOUT)
    return _client


def _load_gpt_cache() -> dict:
    global _gpt_cache
    if _gpt_cache is None:
        if GPT_CACHE.exists():
            _gpt_cache = json.load(open(GPT_CACHE, encoding="utf-8"))
        else:
            _gpt_cache = {}
    return _gpt_cache


def _save_gpt_cache():
    GPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = GPT_CACHE.with_suffix(".tmp")
    json.dump(_gpt_cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    tmp.replace(GPT_CACHE)


def _call_gpt(prompt: str) -> str:
    client = get_client()
    last = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.35,
                max_tokens=2500,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            s = str(e).lower()
            if any(k in s for k in ("rate", "429", "quota", "timeout", "503", "500", "overload")):
                wait = min(10 * attempt, 120)
                print(f"  [gpt-retry {attempt}/{_MAX_RETRIES}] {type(e).__name__}: wait {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"GPT failed after {_MAX_RETRIES} retries: {last}")


def gpt_generate_description_cached(feat, api_key, cache_dir=None):
    """Drop-in replacement for mm.generate_description_cached (GPT, isolated cache).
    Signature matches the original; api_key/cache_dir args are ignored (we use our own)."""
    prompt = build_prompt(feat)
    key = f"{MODEL}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]}"
    cache = _load_gpt_cache()
    if key in cache:
        return cache[key]["description"], True
    desc = _call_gpt(prompt)
    cache[key] = {"description": desc, "model": MODEL, "prompt_preview": prompt[:200]}
    _save_gpt_cache()
    return desc, False


def sample_features(all_features):
    sani = [sanitize(f) for f in all_features]
    cap = config.SAMPLE_PER_TECHNIQUE
    if cap <= 0:
        return sani
    out, cnt = [], defaultdict(int)
    for f in sani:
        t = f["technique_id"]
        if cnt[t] < cap:
            out.append(f); cnt[t] += 1
    return out


def map_one_gpt(all_features):
    sampled = sample_features(all_features)
    return analyze(
        sampled, str(config.MITRE_CSV_PATH), "",
        cache_dir=config.CACHE_DIR,   # canonical FAISS index (read-only)
        use_llm=True,                 # uses our monkeypatched GPT generator
        cross_encoder=None,
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


def list_models():
    client = get_client()
    ids = sorted(m.id for m in client.models.list().data)
    gpt_ids = [i for i in ids if i.startswith(("gpt", "o1", "o3", "o4", "chatgpt"))]
    print("Available chat-capable models:")
    for i in gpt_ids:
        print(" ", i)


def main():
    if "--list-models" in sys.argv:
        list_models(); return

    # install GPT description generator in place of Gemini's
    mm.generate_description_cached = gpt_generate_description_cached
    print(f"[R10/gpt] model = {MODEL}  (override with R10_GPT_MODEL)")
    _ = get_client()  # fail fast if key missing

    datasets = sorted(config.DATASET_FOLDER.rglob("*.json"))
    done = 0
    try:
        for i, ds in enumerate(datasets, 1):
            config.OUTPUT_BASE_DIR = CANON
            config.configure_dataset(ds)
            feat_fp, csv_fp = config.FEATURE_RESULT_JSON_PATH, config.FINALE_CSV_PATH
            name = config.DATASET_NAME
            if not (feat_fp.exists() and csv_fp.exists()):
                print(f"[{i}/{len(datasets)}] skip {name}: missing canonical inputs")
                continue
            all_features = json.load(open(feat_fp, encoding="utf-8"))

            config.OUTPUT_BASE_DIR = R10
            config.configure_dataset(ds)
            if config.VITERBI_JSON_PATH.exists():
                print(f"[{i}/{len(datasets)}] skip {name}: already done")
                config.OUTPUT_BASE_DIR = CANON
                done += 1
                continue
            config.OUTPUT_BASE_DIR = CANON
            config.configure_dataset(ds)

            print(f"\n[{i}/{len(datasets)}] {name}: GPT mapping ({len(all_features)} groups)")
            results = map_one_gpt(all_features)

            config.OUTPUT_BASE_DIR = R10
            config.configure_dataset(ds)
            config.TTP_MAPPING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            json.dump(all_features, open(config.FEATURE_RESULT_JSON_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            shutil.copy(csv_fp, config.FINALE_CSV_PATH)
            json.dump(results, open(config.TTP_MAPPING_JSON_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            ok = run_one(ds)
            done += 1 if ok else 0
    finally:
        config.OUTPUT_BASE_DIR = CANON

    print(f"\n[R10/gpt] done={done} -> {R10}")


if __name__ == "__main__":
    main()
