"""
Structured / adversarial log-drop experiment (CCS->C&S revision, weakness #2).

The main paper's Q2 robustness study uses *uniform random* event drop. This
script adds three *structured* drop policies, each exercising a specific design
feature claimed (qualitatively) in the Discussion:

  - anchor : delete 50% of ANCHOR (rule-flagged) events -> tests the
             "anchor-targeted deletion localizes to a missed group, never a
             spurious group" claim.
  - burst  : delete a contiguous TIME WINDOW totalling 25% of events -> tests
             the hole-bridging operator (the regime it targets).
  - type   : delete 25% of events concentrated in registry/network/file types
             -> tests the shared-entity criterion (cross-artifact survival).

Drops are applied at the RAW-LINE level by ORIGINAL LINE INDEX (robust: some
scenarios lack RecordNumber, so a content key is unreliable). SCOPE and every
baseline then consume exactly the same dropped scenario (fair comparison). We
reuse the method runners from experiments/_robustness_run.py by monkeypatching
its _drop_path resolver to point at our structured-dropped files.

Anchor line indices are obtained by writing a temp JSONL with a passthrough
'_lineidx' column, running load_and_normalize + _find_anchors, and reading the
'_lineidx' of anchor rows -- this avoids any df<->raw key matching.

Output: experiments/ccs_revision/output/structured_drop_scores.json + a table.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]   # Final_Code/
sys.path.insert(0, str(ROOT))

import config
from pipeline.data_loader import load_and_normalize
from pipeline.rule_matcher import load_rules, _find_anchors
import experiments._robustness_run as rob

MODE_SEED = {"anchor": 910, "burst": 911, "type": 912}
NOMINAL_DROP = 0.25
ANCHOR_FRAC = 0.50
BURST_FRAC = 0.25
TYPE_FRAC = 0.25
TYPE_PRIORITY = [13, 12, 14, 3, 11, 7]   # reg-set/create/rename, net, file, imageload

STRUCT_ROOT = config.OUTPUT_BASE_DIR / "_structured"

_RULES_CACHE = None


def _rules():
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = load_rules(config.RULE_FOLDER)
    return _RULES_CACHE


def _struct_path(mode: str, rel: Path) -> Path:
    return STRUCT_ROOT / mode / rel.with_suffix(".json")


def _ts(rec: dict) -> str:
    # consistent within a scenario; lexicographic sort is correct for the
    # zero-padded "YYYY-MM-DD HH:MM:SS.fff" / ISO formats used here.
    return str(rec.get("@timestamp") or rec.get("TimeCreated") or
               rec.get("EventTime") or "")


def _anchor_lineidxs(recs: list[dict]) -> list[int]:
    """Return original line indices of anchor events via the _lineidx trick."""
    aug = []
    for i, r in enumerate(recs):
        rr = dict(r)
        rr["_lineidx"] = i
        aug.append(rr)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as tf:
        for rr in aug:
            tf.write(json.dumps(rr, ensure_ascii=False) + "\n")
        tmp = tf.name
    try:
        df = load_and_normalize(tmp)
        anchors = set()
        for rule in _rules():
            anchors.update(_find_anchors(df, rule).tolist())
        out = []
        for idx in anchors:
            v = df.loc[idx, "_lineidx"]
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                pass
        return out
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass


def _compute_drop_lineidxs(mode: str, recs: list[dict], seed: int) -> set[int]:
    rng = random.Random((seed << 8) ^ (sum(ord(c) for c in mode)))
    n = len(recs)
    if n == 0:
        return set()
    if mode == "anchor":
        anchors = sorted(_anchor_lineidxs(recs))
        k = int(round(len(anchors) * ANCHOR_FRAC))
        if k <= 0:
            return set()
        return set(rng.sample(anchors, k)) if len(anchors) >= k else set(anchors)
    if mode == "burst":
        w = int(round(n * BURST_FRAC))
        if w <= 0:
            return set()
        order = sorted(range(n), key=lambda i: _ts(recs[i]))   # time order
        start = rng.randint(0, max(0, n - w))
        return set(order[start:start + w])
    if mode == "type":
        budget = int(round(n * TYPE_FRAC))
        chosen: list[int] = []
        def eid(r):
            v = r.get("EventID", r.get("event_id"))
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        for t in TYPE_PRIORITY:
            for i in range(n):
                if len(chosen) >= budget:
                    break
                if eid(recs[i]) == t:
                    chosen.append(i)
            if len(chosen) >= budget:
                break
        return set(chosen)
    raise ValueError(mode)


def make_structured_scenarios(modes: list[str]) -> None:
    originals = sorted(config.DATASET_FOLDER.rglob("*.json"))
    for mode in modes:
        seed = MODE_SEED[mode]
        tot_drop = tot_kept = 0
        for orig in originals:
            rel = orig.relative_to(config.DATASET_FOLDER).with_suffix("")
            dst = _struct_path(mode, rel)
            if dst.exists():
                continue
            raw_lines = []
            recs = []
            with open(orig, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    raw_lines.append(line)
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        recs.append({})
            drop_idx = _compute_drop_lineidxs(mode, recs, seed)
            kept = [raw_lines[i] for i in range(len(raw_lines)) if i not in drop_idx]
            if not kept:
                kept = raw_lines[:1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                f.writelines(kept)
            tot_drop += len(drop_idx)
            tot_kept += len(kept)
        print(f"[{mode}] generated; dropped~{tot_drop} kept~{tot_kept}")


def run_all(modes: list[str], methods: list[str]) -> dict:
    make_structured_scenarios(modes)

    orig_drop_path = rob._drop_path
    current = {"mode": None}
    rob._drop_path = lambda drop, seed, rel: _struct_path(current["mode"], rel)

    runners = {
        "Sigma": rob.run_sigma, "MAGIC": rob.run_magic, "DeepAG": rob.run_deepag,
        "SHIELD": rob.run_shield, "SCOPE": rob.run_scope,
    }
    results: dict = {}
    try:
        for mode in modes:
            current["mode"] = mode
            seed = MODE_SEED[mode]
            results[mode] = {}
            for m in methods:
                print(f"\n=== {m}  mode={mode} ===")
                lcs = runners[m](NOMINAL_DROP, seed)
                results[mode][m] = mean(lcs.values()) if lcs else None
                v = results[mode][m]
                print(f"  -> {m} {mode}: tech-LCS = "
                      f"{'NA' if v is None else round(v,4)} ({len(lcs)} scen)")
    finally:
        rob._drop_path = orig_drop_path

    outp = Path(__file__).resolve().parent / "output" / "structured_drop_scores.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n--- Structured-drop tech-LCS ---")
    for mode, mm in results.items():
        print(f"  {mode:8s} " + "  ".join(
            f"{m}={'NA' if v is None else round(v,3)}" for m, v in mm.items()))
    print(f"\nSaved: {outp}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["anchor", "burst", "type"],
                    choices=["anchor", "burst", "type"])
    ap.add_argument("--methods", nargs="+",
                    default=["Sigma", "MAGIC", "DeepAG", "SCOPE"],
                    choices=["Sigma", "MAGIC", "DeepAG", "SHIELD", "SCOPE"])
    args = ap.parse_args()
    run_all(args.modes, args.methods)
