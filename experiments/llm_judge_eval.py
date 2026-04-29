"""
LLM-as-judge per-group TTP evaluation.

For each TP group, give the LLM the raw log evidence (anchor + sample_logs)
plus (a) Viterbi's final technique pick, and (b) top-5 FAISS candidates.
The LLM judges whether the prediction matches the evidence — this is the
ground-truth we actually care about, replacing the hardcoded attack_flows.py
reference with a human-like semantic verdict.

Output per group:
  - viterbi_verdict     : correct | partial | incorrect
  - viterbi_confidence  : 0.0..1.0
  - best_of_top5        : tid (LLM's preferred candidate from top-5)
  - reasoning           : one-line

Caching: results keyed by SHA256(scenario, group_id, pred_tid, top5).
"""
from __future__ import annotations
import csv, hashlib, json, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_CACHE = CACHE_DIR / "llm_judge_cache.json"
RESULTS_JSON = OUTPUT_DIR / "llm_judge_results.json"
RESULTS_CSV = OUTPUT_DIR / "llm_judge_per_scenario.csv"

sys.path.insert(0, str(ROOT))
from pipeline.evaluator import load_ground_truth
from experiments.run_eval_plausible import _is_strong_tp
import config


# ──────────────────────────────────────────────────────────────────────────────
# Gemini 호출 (mitre_mapper.py의 retry 로직 재사용)
# ──────────────────────────────────────────────────────────────────────────────
import google.generativeai as genai

# Judge 전용 모델 — 파이프라인 LLM(gemini-2.0-flash)과 별도 quota 활용.
# gemini-2.5-flash-lite는 빠르고 저렴해 대량 판정에 적합.
_JUDGE_MODEL_NAME = "models/gemini-2.5-flash-lite"
genai.configure(api_key=config.GEMINI_API_KEY)
_JUDGE_MODEL = genai.GenerativeModel(_JUDGE_MODEL_NAME)


def _call_llm(prompt: str, max_retries: int = 4) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _JUDGE_MODEL.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, max_output_tokens=400,
                ),
                request_options={"timeout": 60},
            )
            return resp.text.strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                delay = min(60 * attempt, 240)
                print(f"    [rate] attempt {attempt} wait {delay}s")
                time.sleep(delay)
                continue
            if "timeout" in msg:
                time.sleep(5 * attempt)
                continue
            raise
    raise RuntimeError(f"LLM failed after {max_retries}: {last_err}")


# ──────────────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────────────
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        if JUDGE_CACHE.exists():
            try:
                _cache = json.load(open(JUDGE_CACHE, encoding="utf-8"))
            except Exception:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    tmp = JUDGE_CACHE.with_suffix(".tmp")
    json.dump(_cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    tmp.replace(JUDGE_CACHE)


def _cache_key(scenario: str, gid: str, pred_tid: str, top5: list[str]) -> str:
    payload = f"{scenario}|{gid}|{pred_tid}|{','.join(top5)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_log(entry: dict) -> str:
    fields = []
    for k in ("EventID", "Image", "ParentImage", "CommandLine", "TargetObject",
              "TargetFilename", "TargetImage", "GrantedAccess", "QueryName"):
        v = entry.get(k)
        if v and str(v).lower() != "nan":
            vs = str(v)[:180]
            fields.append(f"{k}={vs}")
    return "    - " + " | ".join(fields) if fields else ""


def _fmt_anchor(anchor: dict) -> str:
    if not anchor:
        return "  (no anchor detail)"
    lines = []
    for k in ("EventID", "Image", "ParentImage", "CommandLine", "TargetObject",
              "TargetFilename", "TargetImage", "GrantedAccess"):
        v = anchor.get(k)
        if v and str(v).lower() != "nan":
            lines.append(f"  {k}: {str(v)[:200]}")
    return "\n".join(lines) if lines else "  (empty)"


def _fmt_candidate(c: dict) -> str:
    tid = c.get("technique_id", "")
    name = c.get("technique_name", "")
    desc = (c.get("description") or "")[:250]
    sim = c.get("similarity", 0)
    return f"    {tid} ({name}): sim={sim:.3f} | {desc}"


def build_judge_prompt(
    scenario: str,
    group: dict,
    pred_tid: str,
    top5_cands: list[dict],
) -> str:
    anchor = group.get("anchor") or {}
    sample_logs = group.get("sample_logs") or []

    cands_block = "\n".join(_fmt_candidate(c) for c in top5_cands[:5])
    logs_block = "\n".join(_fmt_log(s) for s in sample_logs[:10]) or "  (none)"

    # pred_tid name/desc 찾기
    pred_cand = next((c for c in top5_cands if c.get("technique_id") == pred_tid), None)
    if pred_cand:
        pred_name = pred_cand.get("technique_name", "")
        pred_desc = (pred_cand.get("description") or "")[:400]
    else:
        pred_name = "(not in top-5)"
        pred_desc = ""

    return f"""You are a senior threat-analysis reviewer.
Given the forensic evidence below from a single behavior group in a Windows log
trace, judge whether the predicted MITRE ATT&CK technique is a semantically
plausible mapping, and pick which of the top-5 candidates best fits.

## Forensic Evidence

Scenario context: {scenario}
Group ID: {group.get("group_id", "")}

Anchor event:
{_fmt_anchor(anchor)}

Supporting log events (up to 10):
{logs_block}

## Predicted Technique (from Viterbi)
  {pred_tid}: {pred_name}
  Description: {pred_desc}

## Top-5 Candidates (from FAISS/embedding retrieval, after transition rerank)
{cands_block}

## Your Task

1. Does the evidence support the PREDICTED technique ({pred_tid})?
   - "correct" = evidence clearly matches predicted technique.
   - "partial" = evidence is related but a different sibling/parent technique fits better,
                 or the technique is plausible but not the best among candidates.
   - "incorrect" = evidence clearly does not support this technique.
2. Which candidate from the Top-5 list BEST matches the evidence?
   (Use exact ID from the list above.)
3. One-sentence reasoning.

Respond ONLY in JSON (no code fence, no extra text):
{{
  "verdict": "correct" | "partial" | "incorrect",
  "confidence": 0.0..1.0,
  "best_of_top5": "T10xx[.yyy]",
  "reasoning": "one short sentence"
}}
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> dict:
    m = _JSON_RE.search(raw)
    if not m:
        return {"verdict": "incorrect", "confidence": 0.0,
                "best_of_top5": "", "reasoning": f"parse_failed: {raw[:100]}"}
    try:
        obj = json.loads(m.group(0))
        v = obj.get("verdict", "").lower()
        if v not in ("correct", "partial", "incorrect"):
            v = "incorrect"
        return {
            "verdict": v,
            "confidence": float(obj.get("confidence", 0.0)),
            "best_of_top5": str(obj.get("best_of_top5", "")).strip(),
            "reasoning": str(obj.get("reasoning", ""))[:300],
        }
    except Exception as e:
        return {"verdict": "incorrect", "confidence": 0.0,
                "best_of_top5": "", "reasoning": f"json_err: {e}"}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run(strong_only: bool = True, limit_per_scenario: int | None = None) -> dict:
    cache = _load_cache()
    per_scenario = []
    all_details = []

    ann_files = sorted(OUTPUT_DIR.rglob("*_annotation.json"))

    for ann in ann_files:
        gt = load_ground_truth(ann)
        if not gt:
            continue
        with open(ann, encoding="utf-8") as f:
            ad = json.load(f)
        scenario = ad.get("scenario", ann.parent.name)
        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")
        if not (ttp_fp.exists() and vit_fp.exists()):
            continue
        with open(ttp_fp, encoding="utf-8") as f:
            ttp = json.load(f)
        with open(vit_fp, encoding="utf-8") as f:
            vit = json.load(f)

        group_by_gid = {g["group_id"]: g for g in ad["groups"]}
        ttp_by_gid = {r["group_id"]: r for r in ttp}
        vit_tid_by_gid = {b["group_id"]: b["technique_id"] for b in vit}

        verdicts = {"correct": 0, "partial": 0, "incorrect": 0}
        n_top5_hit = 0
        n_judge_top1_eq_viterbi = 0
        n = 0
        scen_details = []

        tp_gids = [gid for gid, t in gt.items() if t["is_tp"]]
        if strong_only:
            tp_gids = [gid for gid in tp_gids if _is_strong_tp(gt[gid].get("notes", ""))]
        if limit_per_scenario:
            tp_gids = tp_gids[:limit_per_scenario]

        for gid in tp_gids:
            r = ttp_by_gid.get(gid)
            grp = group_by_gid.get(gid)
            if not r or not grp:
                continue
            top5 = r.get("similar_techniques", [])[:5]
            if not top5:
                continue
            top5_tids = [c["technique_id"] for c in top5]
            pred_tid = vit_tid_by_gid.get(gid, top5_tids[0])

            key = _cache_key(scenario, gid, pred_tid, top5_tids)
            if key in cache:
                verdict = cache[key]
            else:
                prompt = build_judge_prompt(scenario, grp, pred_tid, top5)
                try:
                    raw = _call_llm(prompt)
                except Exception as e:
                    print(f"    [err] {gid}: {e}")
                    continue
                verdict = _parse_verdict(raw)
                cache[key] = verdict
                _save_cache()

            v = verdict["verdict"]
            verdicts[v] += 1
            n += 1
            top5_pick = verdict.get("best_of_top5", "")
            if top5_pick in top5_tids:
                n_top5_hit += 1
            if top5_pick == pred_tid:
                n_judge_top1_eq_viterbi += 1

            scen_details.append({
                "group_id": gid,
                "pred_tid": pred_tid,
                "top5": top5_tids,
                **verdict,
            })

        if n == 0:
            continue

        correct_rate = verdicts["correct"] / n
        any_match_rate = (verdicts["correct"] + verdicts["partial"]) / n
        per_scenario.append({
            "scenario": scenario[:60],
            "n": n,
            "correct": verdicts["correct"],
            "partial": verdicts["partial"],
            "incorrect": verdicts["incorrect"],
            "correct_rate": round(correct_rate, 3),
            "any_match_rate": round(any_match_rate, 3),
            "judge_top5_hit": n_top5_hit,
            "judge_eq_viterbi": n_judge_top1_eq_viterbi,
        })
        all_details.extend(scen_details)
        print(f"  {scenario[:50]:<50s} n={n:>3d}  correct={verdicts['correct']:>3d}  "
              f"partial={verdicts['partial']:>3d}  incorrect={verdicts['incorrect']:>3d}  "
              f"rate={correct_rate:.2f}")

    # Aggregate
    total = sum(p["n"] for p in per_scenario)
    total_correct = sum(p["correct"] for p in per_scenario)
    total_partial = sum(p["partial"] for p in per_scenario)
    total_incorrect = sum(p["incorrect"] for p in per_scenario)
    total_top5_hit = sum(p["judge_top5_hit"] for p in per_scenario)
    total_agree = sum(p["judge_eq_viterbi"] for p in per_scenario)

    print()
    print("=" * 95)
    print(f"  LLM-as-JUDGE RESULTS  (strong_only={strong_only})")
    print("=" * 95)
    print(f"  scenarios evaluated: {len(per_scenario)}")
    print(f"  total TP groups judged: {total}")
    print()
    print(f"  Correct        : {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    print(f"  Partial        : {total_partial}/{total} ({100*total_partial/total:.1f}%)")
    print(f"  Incorrect      : {total_incorrect}/{total} ({100*total_incorrect/total:.1f}%)")
    print()
    print(f"  Correct+Partial: {total_correct+total_partial}/{total} "
          f"({100*(total_correct+total_partial)/total:.1f}%)")
    print()
    print(f"  Judge's pick IN top-5     : {total_top5_hit}/{total} "
          f"({100*total_top5_hit/total:.1f}%)")
    print(f"  Judge agrees with Viterbi : {total_agree}/{total} "
          f"({100*total_agree/total:.1f}%)")

    # Save
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate": {
                "total": total, "correct": total_correct, "partial": total_partial,
                "incorrect": total_incorrect,
                "correct_rate": total_correct/total if total else 0,
                "any_match_rate": (total_correct+total_partial)/total if total else 0,
            },
            "per_scenario": per_scenario,
            "details": all_details,
        }, f, ensure_ascii=False, indent=2)
    if per_scenario:
        with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_scenario[0].keys()))
            w.writeheader(); w.writerows(per_scenario)
    print(f"\n  saved: {RESULTS_JSON}, {RESULTS_CSV}")

    return {
        "total": total, "correct": total_correct, "partial": total_partial,
        "incorrect": total_incorrect,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tp", action="store_true", help="Judge all TP (default: strong only)")
    parser.add_argument("--limit", type=int, default=None, help="limit groups per scenario (debug)")
    args = parser.parse_args()
    run(strong_only=not args.all_tp, limit_per_scenario=args.limit)
