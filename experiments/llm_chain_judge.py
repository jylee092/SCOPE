"""
Scenario-level chain comparison: LLM vs Viterbi.

For each scenario, feed the full raw log sequence (or compact group summaries)
to the LLM and ask it to reconstruct the attack chain as an ordered list of
(technique_id, tactic) tuples. Then compare against Viterbi's output chain
with the same step_match / LCS metrics as experiments/chain_align.py.

Outputs per-scenario:
  - llm_chain (list of TIDs)
  - viterbi_chain (list of TIDs)
  - step_coverage   (LLM-derived GT coverage by Viterbi)
  - tactic_lcs_norm (tactic order preservation)
  - technique_lcs_norm (technique order preservation, family match)
  - order_accuracy
"""
from __future__ import annotations
import csv, hashlib, json, re, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
CACHE = OUT / "_cache" / "llm_chain_judge_cache.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "llm_chain_judge_results.json"
CSV_OUT = OUT / "llm_chain_judge.csv"

sys.path.insert(0, str(ROOT))
import config
from experiments.chain_align import step_match, lcs_length
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
                    temperature=0.1, max_output_tokens=1200),
                request_options={"timeout": 90},
            )
            return r.text.strip()
        except Exception as e:
            last = e
            m = str(e).lower()
            if "429" in m or "quota" in m:
                time.sleep(min(60 * i, 240)); continue
            if "timeout" in m:
                time.sleep(5 * i); continue
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


def _short_log(e: dict) -> str:
    """Compact log line for prompt."""
    bits = []
    eid = e.get("EventID")
    if eid: bits.append(f"EID{eid}")
    for k in ("Image", "ParentImage", "CommandLine", "TargetObject",
              "TargetFilename", "TargetImage", "GrantedAccess"):
        v = e.get(k)
        if v and str(v).lower() != "nan":
            s = str(v)
            if len(s) > 120:
                s = s[:120] + "..."
            bits.append(f"{k}={s}")
    return " | ".join(bits)


def build_scenario_prompt(scenario: str, groups_summary: list[dict]) -> str:
    """
    groups_summary: list of {group_id, anchor, sample_logs}
    We present groups in time order with a compact rendering.
    """
    blocks = []
    for i, g in enumerate(groups_summary, 1):
        gid = g.get("group_id", "")
        anc = g.get("anchor") or {}
        anc_line = _short_log(anc)
        logs = g.get("sample_logs") or []
        log_lines = [f"      {_short_log(L)}" for L in logs[:6]]
        blocks.append(
            f"  [Group {i}] {gid}\n"
            f"    anchor: {anc_line}\n"
            + ("    supporting:\n" + "\n".join(log_lines) if log_lines else "")
        )
    body = "\n\n".join(blocks)

    return f"""You are a senior threat analyst examining a Windows event log trace.
The trace has been segmented into time-ordered behavior groups. For each group,
you see the anchor event and a few supporting events. Using ONLY this evidence,
reconstruct the most likely MITRE ATT&CK attack chain as an ordered list of
(technique_id, tactic) pairs for the underlying attack steps.

Rules:
- You are NOT restricted to the order of groups -- consolidate consecutive
  groups that show the same attack step. Drop groups that appear to be benign
  system noise (don't include them in the chain).
- Be specific with technique IDs (use sub-techniques when possible, e.g., T1003.001
  instead of T1003 when the evidence points to LSASS memory).
- Typical chain length for atomic scenarios: 2–4 steps; compound: 3–6 steps.

## Scenario name hint (for context only, do NOT copy-paste its TIDs; rely on evidence)
{scenario}

## Behavior groups (time order)

{body}

## Respond in JSON only (no code fence):
{{
  "chain": [
    {{"tid": "T10xx[.yyy]", "tactic": "Execution|Persistence|...", "note": "brief why"}},
    ...
  ],
  "confidence": 0.0..1.0,
  "reasoning": "one-sentence overall"
}}
"""


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse_chain(raw: str) -> dict:
    m = _JSON.search(raw)
    if not m:
        return {"chain": [], "confidence": 0.0, "reasoning": f"parse_fail: {raw[:150]}"}
    try:
        obj = json.loads(m.group(0))
        chain = []
        for s in obj.get("chain", []):
            tid = str(s.get("tid", "")).strip().upper()
            tac = str(s.get("tactic", "")).strip()
            if tid:
                chain.append({"tid": tid, "tactic": tac,
                              "note": str(s.get("note", ""))[:200]})
        return {
            "chain": chain,
            "confidence": float(obj.get("confidence", 0.0)),
            "reasoning": str(obj.get("reasoning", ""))[:300],
        }
    except Exception as e:
        return {"chain": [], "confidence": 0.0, "reasoning": f"json_err:{e}"}


def compare_chains(llm_chain: list[dict], viterbi_chain: list[dict]) -> dict:
    """llm_chain = LLM-derived GT. viterbi_chain = our pipeline prediction.
    Metrics mirror experiments/chain_align.py."""
    if not llm_chain:
        return {"error": "empty llm chain"}
    llm_steps = [{"tid": s["tid"], "alts": []} for s in llm_chain]
    llm_tids = [s["tid"] for s in llm_chain]
    llm_tacs = [s.get("tactic", "") for s in llm_chain]
    vit_tids = [b["technique_id"] for b in viterbi_chain]
    vit_tacs = [b["tactic"] for b in viterbi_chain]

    matched = sum(1 for s in llm_steps
                  if any(step_match(s, p) for p in vit_tids))
    step_cov = matched / len(llm_steps)

    tac_lcs = lcs_length(llm_tacs, vit_tacs)
    tac_lcs_norm = tac_lcs / len(llm_tacs) if llm_tacs else 0

    def tech_eq(l_step, p_tid): return step_match(l_step, p_tid)
    tech_lcs = lcs_length(llm_steps, vit_tids, eq=tech_eq)
    tech_lcs_norm = tech_lcs / len(llm_steps) if llm_steps else 0

    matched_idx = []
    for s in llm_steps:
        for i, p in enumerate(vit_tids):
            if step_match(s, p):
                matched_idx.append(i); break
        else:
            matched_idx.append(None)
    valid = [i for i in matched_idx if i is not None]
    if len(valid) <= 1:
        order_acc = 1.0 if valid else 0.0
    else:
        ordered = sum(1 for a, b in zip(valid, valid[1:]) if a < b)
        order_acc = ordered / (len(valid) - 1)

    return {
        "llm_steps": len(llm_chain),
        "viterbi_steps": len(viterbi_chain),
        "step_coverage": round(step_cov, 4),
        "tactic_lcs_norm": round(tac_lcs_norm, 4),
        "technique_lcs_norm": round(tech_lcs_norm, 4),
        "order_accuracy": round(order_acc, 4),
    }


def _key(scenario: str) -> str:
    return hashlib.sha256(scenario.encode()).hexdigest()[:16]


def run():
    cache = _load_cache()
    per_scen_rows = []
    details = []

    scen_coverage = scen_tac_lcs = scen_tech_lcs = scen_order = 0.0
    n = 0

    for ann in sorted(OUT.rglob("*_annotation.json")):
        ad = json.load(open(ann, encoding="utf-8"))
        scenario = ad.get("scenario", ann.parent.name)
        stem = ann.name.replace("_annotation.json", "")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")
        if not vit_fp.exists(): continue
        viterbi_chain = json.load(open(vit_fp, encoding="utf-8"))
        groups = ad.get("groups", [])
        if not groups: continue

        # Build LLM chain (cached)
        k = _key(scenario)
        if k in cache:
            llm_res = cache[k]
        else:
            prompt = build_scenario_prompt(scenario, groups)
            # Cap prompt size: if too many groups, sample first 30
            try:
                raw = _call(prompt)
            except Exception as e:
                print(f"  [err] {scenario}: {e}")
                continue
            llm_res = _parse_chain(raw)
            cache[k] = llm_res
            _save_cache()

        llm_chain = llm_res.get("chain", [])
        if not llm_chain:
            print(f"  [empty] {scenario}")
            continue

        metrics = compare_chains(llm_chain, viterbi_chain)
        n += 1
        scen_coverage += metrics["step_coverage"]
        scen_tac_lcs += metrics["tactic_lcs_norm"]
        scen_tech_lcs += metrics["technique_lcs_norm"]
        scen_order += metrics["order_accuracy"]

        row = {
            "scenario": scenario[:55],
            "llm_steps": metrics["llm_steps"],
            "viterbi_steps": metrics["viterbi_steps"],
            "step_cov": metrics["step_coverage"],
            "tac_lcs": metrics["tactic_lcs_norm"],
            "tech_lcs": metrics["technique_lcs_norm"],
            "order": metrics["order_accuracy"],
            "llm_tids": ",".join(s["tid"] for s in llm_chain),
            "viterbi_tids": ",".join(b["technique_id"] for b in viterbi_chain[:10]),
        }
        per_scen_rows.append(row)
        details.append({
            "scenario": scenario,
            "llm_chain": llm_chain,
            "viterbi_chain": [{"tid": b["technique_id"], "tactic": b["tactic"], "gid": b["group_id"]}
                              for b in viterbi_chain],
            **metrics,
            "reasoning": llm_res.get("reasoning", ""),
        })
        print(f"  {scenario[:48]:<48s} LLM={metrics['llm_steps']:>2d} Vit={metrics['viterbi_steps']:>3d} "
              f"cov={metrics['step_coverage']:.2f} tac={metrics['tactic_lcs_norm']:.2f} "
              f"tech={metrics['technique_lcs_norm']:.2f} ord={metrics['order_accuracy']:.2f}")

    if n == 0:
        print("No scenarios")
        return

    print()
    print("=" * 95)
    print("  LLM vs Viterbi CHAIN COMPARISON (LLM as chain-level judge)")
    print("=" * 95)
    print(f"  scenarios: {n}")
    print(f"  step_coverage     (LLM chain covered by Viterbi): {scen_coverage/n:.4f}")
    print(f"  tactic_lcs_norm   (tactic sequence preservation): {scen_tac_lcs/n:.4f}")
    print(f"  technique_lcs_norm(tech sequence, family match) : {scen_tech_lcs/n:.4f}")
    print(f"  order_accuracy    (matched pairs ordered)       : {scen_order/n:.4f}")

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({"aggregate": {
            "scenarios": n,
            "step_coverage": scen_coverage/n,
            "tactic_lcs_norm": scen_tac_lcs/n,
            "technique_lcs_norm": scen_tech_lcs/n,
            "order_accuracy": scen_order/n,
        }, "per_scenario": details}, f, ensure_ascii=False, indent=2)
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_scen_rows[0].keys()))
        w.writeheader(); w.writerows(per_scen_rows)
    print(f"\n  saved: {RESULTS}, {CSV_OUT}")


if __name__ == "__main__":
    run()
