"""

"""
from __future__ import annotations
import csv, hashlib, json, re, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
CACHE = OUT / "_cache" / "llm_free_judge_cache.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
import config
from pipeline.evaluator import load_ground_truth
from experiments.run_eval_plausible import _is_strong_tp, tid_family_match
from experiments.attack_flows import get_flow, all_acceptable_tids
import google.generativeai as genai

genai.configure(api_key=config.GEMINI_API_KEY)
_MODEL = genai.GenerativeModel("models/gemini-2.5-flash-lite")


def _call(prompt: str, retries: int = 4) -> str:
    last = None
    for i in range(1, retries + 1):
        try:
            r = _MODEL.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, max_output_tokens=300),
                request_options={"timeout": 45},
            )
            return r.text.strip()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg:
                time.sleep(min(60 * i, 240))
                continue
            if "timeout" in msg:
                time.sleep(5 * i)
                continue
            raise
    raise RuntimeError(f"LLM failed: {last}")


_cache = None
def _load_cache() -> dict:
    global _cache
    if _cache is None:
        if CACHE.exists():
            _cache = json.load(open(CACHE, encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def _save_cache():
    tmp = CACHE.with_suffix(".tmp")
    json.dump(_cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    tmp.replace(CACHE)


def _fmt_log(e: dict) -> str:
    bits = []
    for k in ("EventID", "Image", "ParentImage", "CommandLine",
              "TargetObject", "TargetFilename", "TargetImage",
              "GrantedAccess", "QueryName"):
        v = e.get(k)
        if v and str(v).lower() != "nan":
            bits.append(f"{k}={str(v)[:180]}")
    return "    - " + " | ".join(bits) if bits else ""


def _fmt_anchor(a: dict) -> str:
    if not a: return "  (no anchor)"
    lines = []
    for k in ("EventID", "Image", "ParentImage", "CommandLine",
              "TargetObject", "TargetFilename", "TargetImage", "GrantedAccess"):
        v = a.get(k)
        if v and str(v).lower() != "nan":
            lines.append(f"  {k}: {str(v)[:200]}")
    return "\n".join(lines) or "  (empty)"


def build_prompt(scenario: str, group: dict) -> str:
    anchor = group.get("anchor") or {}
    samples = group.get("sample_logs") or []
    logs = "\n".join(_fmt_log(s) for s in samples[:10]) or "  (none)"
    return f"""You are a senior threat-analysis reviewer.
Given the forensic evidence below from a single behavior group in a Windows log
trace, identify the MITRE ATT&CK technique that best characterizes what the
adversary is doing in this group.

IMPORTANT: Use your own MITRE ATT&CK knowledge. Do NOT restrict yourself to
any candidate list -- return whichever technique ID fits best.

## Forensic Evidence

Scenario context: {scenario}
Group ID: {group.get('group_id', '')}

Anchor event:
{_fmt_anchor(anchor)}

Supporting log events (up to 10):
{logs}

## Your Task

Return JSON only (no code fence):
{{
  "primary_tid": "T10xx[.yyy]",
  "alt_tid": "T10xx[.yyy]",
  "confidence": 0.0..1.0,
  "reasoning": "one short sentence"
}}
"""


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> dict:
    m = _JSON.search(raw)
    if not m:
        return {"primary_tid": "", "alt_tid": "", "confidence": 0.0, "reasoning": "parse_fail"}
    try:
        obj = json.loads(m.group(0))
        return {
            "primary_tid": str(obj.get("primary_tid", "")).strip().upper(),
            "alt_tid": str(obj.get("alt_tid", "")).strip().upper(),
            "confidence": float(obj.get("confidence", 0.0)),
            "reasoning": str(obj.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return {"primary_tid": "", "alt_tid": "", "confidence": 0.0,
                "reasoning": f"json_err:{e}"}


def _key(scenario: str, gid: str) -> str:
    return hashlib.sha256(f"{scenario}|{gid}".encode()).hexdigest()[:20]


def main(strong_only: bool = True):
    cache = _load_cache()

    # stats
    n = 0
    n_primary_in_top5 = 0
    n_primary_or_alt_in_top5 = 0
    n_primary_in_acceptable = 0
    n_primary_matches_viterbi = 0

    for ann in sorted(OUT.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt: continue
        ad = json.load(open(ann, encoding="utf-8"))
        scenario = ad.get("scenario", ann.parent.name)
        flow = get_flow(scenario)
        acc = set(all_acceptable_tids(flow)) if flow else set()
        for a in list(acc): acc.add(a.split(".")[0])

        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")
        if not (ttp_fp.exists() and vit_fp.exists()): continue
        ttp = json.load(open(ttp_fp, encoding="utf-8"))
        vit = json.load(open(vit_fp, encoding="utf-8"))
        gbg = {g["group_id"]: g for g in ad["groups"]}
        tbg = {r["group_id"]: r for r in ttp}
        vbg = {b["group_id"]: b["technique_id"] for b in vit}

        for gid, truth in gt.items():
            if not truth["is_tp"]: continue
            if strong_only and not _is_strong_tp(truth.get("notes", "")): continue
            grp = gbg.get(gid); r = tbg.get(gid)
            if not grp or not r: continue
            top5 = [c["technique_id"] for c in r.get("similar_techniques", [])[:5]]
            if not top5: continue
            vit_tid = vbg.get(gid, "")

            k = _key(scenario, gid)
            if k in cache:
                res = cache[k]
            else:
                try:
                    raw = _call(build_prompt(scenario, grp))
                except Exception as e:
                    print(f"  err {gid}: {e}")
                    continue
                res = _parse(raw)
                cache[k] = res
                _save_cache()

            primary = res["primary_tid"]
            alt = res["alt_tid"]

            def in_top5(t):
                if not t: return False
                if t in top5: return True
                root = t.split(".")[0]
                return any(c.split(".")[0] == root for c in top5)

            def in_acc(t):
                return any(tid_family_match(t, a) for a in acc) if t else False

            p_t5 = in_top5(primary)
            a_t5 = in_top5(alt) if alt else False

            n += 1
            if p_t5: n_primary_in_top5 += 1
            if p_t5 or a_t5: n_primary_or_alt_in_top5 += 1
            if in_acc(primary): n_primary_in_acceptable += 1
            if primary and (primary == vit_tid or primary.split(".")[0] == vit_tid.split(".")[0]):
                n_primary_matches_viterbi += 1

    print(f"Unconstrained LLM judgement (strong_only={strong_only})")
    print(f"  N groups: {n}")
    print()
    print(f"  LLM primary TID ∈ our Top-5 (family match) : {n_primary_in_top5}/{n} "
          f"({100*n_primary_in_top5/n:.1f}%)")
    print(f"  LLM primary or alt ∈ Top-5                 : {n_primary_or_alt_in_top5}/{n} "
          f"({100*n_primary_or_alt_in_top5/n:.1f}%)")
    print(f"  LLM primary ∈ scenario acceptable set      : {n_primary_in_acceptable}/{n} "
          f"({100*n_primary_in_acceptable/n:.1f}%)")
    print(f"  LLM primary matches Viterbi pick           : {n_primary_matches_viterbi}/{n} "
          f"({100*n_primary_matches_viterbi/n:.1f}%)")


if __name__ == "__main__":
    main(strong_only=True)
