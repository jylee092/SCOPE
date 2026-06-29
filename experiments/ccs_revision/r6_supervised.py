"""
R6 (CCS reviewer ⑥): Supervised-classifier emission baseline vs SCOPE.

Contrast (paradigm-level, NOT a head swap):
  SCOPE     : telemetry -> LLM description -> ATTACK-BERT embedding ->
              nearest-neighbour match to the FULL MITRE catalogue (learning-free,
              covers all 600+ techniques incl. unseen).
  Supervised: telemetry features -> trained classifier -> technique
              (learning-based, limited to techniques seen in training).

Two representations / models (no LLM, no ATTACK-BERT on the baseline side):
  (A) TF-IDF(feature_to_text)            -> Linear SVM     [text route]
  (B) dense feature vector (event-ID     -> CatBoost       [tabular SOTA, main]
      histogram, signal flags, counts, rule_tid categorical)

Protocol: Leave-One-Scenario-Out CV (the realistic test of generalising to a
new scenario). Evaluated with the SAME family-match / acceptable-set machinery
as the paper (tid_family_match, all_acceptable_tids), so numbers are directly
comparable to SCOPE's emission (R7 mapping: strict 0.43 / H@1 0.62 / H@5 0.84).

Reported regimes:
  realistic : all TP test groups (gt unseen in train -> automatic miss).
  seen-only : test groups whose gt technique family appears in the train set
              (optimistic upper bound for the classifier).

Key data fact (why supervised is capped): 251 TP groups, 30 techniques, 24/30
appear in only ONE scenario -> under LOSO they are never in the train fold.

Writes solely under output/_ccs_revision/R6_supervised/.
Run:  python -m experiments.ccs_revision.r6_supervised
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from experiments.ablation.helpers import feature_to_text
from experiments.attack_flows import get_flow, all_acceptable_tids
from experiments.run_eval_plausible import tid_family_match

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier

OUT_DIR = ROOT / "output" / "_ccs_revision" / "R6_supervised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC = ["num_idxs", "n_proc_chains", "n_cmdlines", "has_obf", "n_registry",
           "n_dropped", "n_connections", "log_cleared", "n_deleted",
           "integrity_ord", "confidence"]
CAT = ["rule_tid"]
_INTEG = {"system": 3, "high": 2, "medium": 1, "low": 0, "": 0, None: 0}


def _root(t):
    return t.split(".")[0] if t else ""


def dense_features(feat):
    f = feat.get("features", {}) or {}
    ctx = f.get("execution_context") or {}
    cmd = f.get("command_script") or {}
    per = f.get("persistence") or {}
    net = f.get("network") or {}
    eva = f.get("evasion") or {}
    idn = f.get("identity") or {}
    return {
        "num_idxs": len(feat.get("all_idxs", []) or []),
        "n_proc_chains": len(ctx.get("process_chains", []) or []),
        "n_cmdlines": len(cmd.get("entries", []) or []),
        "has_obf": int(bool(cmd.get("has_obfuscation"))),
        "n_registry": len(per.get("registry_signals", []) or []),
        "n_dropped": len(per.get("dropped_files", []) or []),
        "n_connections": len(net.get("connections", []) or []),
        "log_cleared": int(bool(eva.get("log_cleared"))),
        "n_deleted": len(eva.get("deleted_files", []) or []),
        "integrity_ord": _INTEG.get((idn.get("integrity_level") or "").lower(), 0),
        "confidence": float(feat.get("confidence", 0.0) or 0.0),
        "rule_tid": str(feat.get("technique_id") or "UNK"),
    }


def collect():
    """Return list of dicts: scenario, gt, text, dense(dict), acceptable(set)."""
    rows = []
    for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
        config.configure_dataset(ds)
        ann_fp, feat_fp = config.ANNOTATION_JSON_PATH, config.FEATURE_RESULT_JSON_PATH
        if not (ann_fp.exists() and feat_fp.exists()):
            continue
        flow = get_flow(config.DATASET_NAME)
        if not flow:
            continue
        acceptable = all_acceptable_tids(flow)
        ann = json.load(open(ann_fp, encoding="utf-8"))
        feats = {f["group_id"]: f for f in json.load(open(feat_fp, encoding="utf-8"))}
        for g in ann.get("groups", []):
            if str(g.get("gt_is_true_positive")).lower() != "true":
                continue
            feat = feats.get(g["group_id"])
            if feat is None:
                continue
            rows.append({
                "scenario": config.DATASET_NAME,
                "gt": g.get("gt_technique_id") or "",
                "text": feature_to_text(feat) or "",
                "dense": dense_features(feat),
                "acceptable": acceptable,
            })
    return rows


def rank_eval(ranked, gt, acceptable, train_class_roots):
    """ranked: predicted technique ids best-first (only train classes).
    Returns (strict_h1, plaus_h1, plaus_h5, seen)."""
    seen = _root(gt) in train_class_roots
    strict = int(bool(ranked) and tid_family_match(ranked[0], gt))
    p1 = int(bool(ranked) and any(tid_family_match(ranked[0], a) for a in acceptable))
    p5 = int(any(tid_family_match(r, a) for r in ranked[:5] for a in acceptable))
    return strict, p1, p5, seen


def svm_rank(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
    Xtr = vec.fit_transform([r["text"] for r in train])
    ytr = [r["gt"] for r in train]
    clf = LinearSVC()
    clf.fit(Xtr, ytr)
    classes = clf.classes_
    Xte = vec.transform([r["text"] for r in test])
    scores = clf.decision_function(Xte)
    if scores.ndim == 1:  # binary edge case
        scores = np.vstack([-scores, scores]).T
    order = np.argsort(-scores, axis=1)
    return [[classes[j] for j in row] for row in order], set(_root(c) for c in classes)


def catboost_rank(train, test):
    dtr = pd.DataFrame([r["dense"] for r in train])
    dte = pd.DataFrame([r["dense"] for r in test])
    ytr = [r["gt"] for r in train]
    clf = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.15,
                             loss_function="MultiClass", verbose=False,
                             cat_features=CAT, random_seed=0, thread_count=-1)
    clf.fit(dtr, ytr)
    classes = np.array(clf.classes_).ravel()
    proba = clf.predict_proba(dte)
    order = np.argsort(-proba, axis=1)
    return [[classes[j] for j in row] for row in order], set(_root(c) for c in classes)


def run_model(name, rank_fn, rows, scenarios):
    """LOSO; returns dict of micro metrics for realistic & seen-only."""
    agg = {"realistic": [], "seen": []}
    for s in scenarios:
        train = [r for r in rows if r["scenario"] != s]
        test = [r for r in rows if r["scenario"] == s]
        if not test or len(set(r["gt"] for r in train)) < 2:
            continue
        ranked_lists, train_roots = rank_fn(train, test)
        for r, ranked in zip(test, ranked_lists):
            strict, p1, p5, seen = rank_eval(ranked, r["gt"], r["acceptable"], train_roots)
            agg["realistic"].append((strict, p1, p5))
            if seen:
                agg["seen"].append((strict, p1, p5))

    def macro(pool):
        if not pool:
            return None
        a = np.array(pool)
        return {"n": len(pool), "strict_h1": round(a[:, 0].mean(), 4),
                "plausible_h1": round(a[:, 1].mean(), 4),
                "plausible_h5": round(a[:, 2].mean(), 4)}
    return {"model": name, "realistic": macro(agg["realistic"]), "seen_only": macro(agg["seen"])}


def main():
    rows = collect()
    scenarios = sorted(set(r["scenario"] for r in rows))
    print(f"[R6] {len(rows)} TP groups, {len(scenarios)} scenarios, "
          f"{len(set(r['gt'] for r in rows))} distinct techniques")

    results = []
    print("  training LinearSVM (TF-IDF) ...")
    results.append(run_model("LinearSVM_tfidf", svm_rank, rows, scenarios))
    print("  training CatBoost (dense features) ...")
    results.append(run_model("CatBoost_features", catboost_rank, rows, scenarios))

    # SCOPE emission reference (R7 mapping, same metric machinery)
    scope = None
    r7p = ROOT / "output" / "_ccs_revision" / "R7_self_metrics" / "self_metrics.json"
    if r7p.exists():
        m = json.load(open(r7p, encoding="utf-8")).get("mapping_micro", {})
        scope = {"model": "SCOPE_emission(R7)", "strict_h1": m.get("strict_h1"),
                 "plausible_h1": m.get("plausible_h1"), "plausible_h5": m.get("plausible_h5")}

    out = {"n_tp_groups": len(rows), "n_scenarios": len(scenarios),
           "n_techniques": len(set(r["gt"] for r in rows)),
           "scope_emission": scope, "supervised": results}
    json.dump(out, open(OUT_DIR / "r6_compare.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n[R6] comparison -> {OUT_DIR/'r6_compare.json'}")
    print(f"  {'model':<22}{'regime':<11}{'n':>5}{'strict':>9}{'plaus_h1':>10}{'plaus_h5':>10}")
    if scope:
        print(f"  {'SCOPE emission':<22}{'(TP)':<11}{'-':>5}{scope['strict_h1']:>9}"
              f"{scope['plausible_h1']:>10}{scope['plausible_h5']:>10}")
    for r in results:
        for reg in ("realistic", "seen_only"):
            m = r[reg]
            if m:
                print(f"  {r['model']:<22}{reg:<11}{m['n']:>5}{m['strict_h1']:>9}"
                      f"{m['plausible_h1']:>10}{m['plausible_h5']:>10}")


if __name__ == "__main__":
    main()
