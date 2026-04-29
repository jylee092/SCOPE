"""CE weight ablation — test ce_weight = 0.0, 0.2, 0.4, 0.6 for emission rerank."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from pipeline.mitre_mapper import (
    build_or_load_faiss_index, get_embed_model, search_similar,
)
from pipeline.attack_chain import get_semantic_scorer
from pipeline.evaluator import load_ground_truth
from experiments.run_eval import load_tactic_map as _tm, patch_candidate_tactics
from experiments.attack_flows import get_flow, all_acceptable_tids
from experiments.run_eval_plausible import _is_strong_tp, tid_family_match

MITRE_CSV = ROOT / "TTP_Data" / "Final_merged_mitre_attack_data.csv"


def match(p, acc):
    return any(tid_family_match(p, a) for a in acc) if p else False


def evaluate_with_ce(ce_weight: float):
    embed_model = get_embed_model()
    index, meta_df = build_or_load_faiss_index(
        str(config.MITRE_CSV_PATH), embed_model, cache_dir=config.CACHE_DIR,
    )
    ce_model = None
    if ce_weight > 0:
        sem = get_semantic_scorer(config.CROSS_ENCODER_MODEL)
        ce_model = getattr(sem, "_model", None)

    tm = _tm(MITRE_CSV)
    n_total = n_hit_top1 = n_hit_top5 = 0
    n_strong = n_strong_top1 = n_strong_top5 = 0

    # Re-score each TP group using cached LLM description (from ttp_mapping.json)
    # but re-run search_similar with new ce_weight.
    for ann in sorted(config.OUTPUT_BASE_DIR.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt: continue
        with open(ann, encoding="utf-8") as f:
            ad = json.load(f)
        scenario = ad.get("scenario", ann.parent.name)
        flow = get_flow(scenario)
        if not flow: continue
        acc = set(all_acceptable_tids(flow))
        for a in list(acc): acc.add(a.split(".")[0])

        stem = ann.name.replace("_annotation.json", "")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        if not ttp_fp.exists(): continue
        with open(ttp_fp, encoding="utf-8") as f:
            ttp = json.load(f)
        patch_candidate_tactics(ttp, tm)

        rule_by_gid = {g["group_id"]: g.get("rule_technique_id","") for g in ad["groups"]}

        for r in ttp:
            gid = r["group_id"]
            if gid not in gt or not gt[gid]["is_tp"]: continue
            desc = r.get("generated_description", "")
            if not desc: continue
            rule_tid = rule_by_gid.get(gid, "")
            # Re-run FAISS + CE rerank with current ce_weight
            new_cands, _ = search_similar(
                desc, index, meta_df, embed_model,
                rule_tid=rule_tid,
                cross_encoder=ce_model,
                ce_rerank_width=20,
                ce_weight=ce_weight,
            )
            ranked = [c["technique_id"] for c in new_cands[:5]]
            if not ranked: continue
            top1_hit = int(match(ranked[0], acc))
            top5_hit = int(any(match(t, acc) for t in ranked))
            n_total += 1
            n_hit_top1 += top1_hit
            n_hit_top5 += top5_hit
            if _is_strong_tp(gt[gid].get("notes", "")):
                n_strong += 1
                n_strong_top1 += top1_hit
                n_strong_top5 += top5_hit

    return {
        "ce_weight": ce_weight,
        "n_all": n_total,
        "all_h1": n_hit_top1 / n_total if n_total else 0,
        "all_h5": n_hit_top5 / n_total if n_total else 0,
        "n_strong": n_strong,
        "strong_h1": n_strong_top1 / n_strong if n_strong else 0,
        "strong_h5": n_strong_top5 / n_strong if n_strong else 0,
    }


def main():
    print(f"{'ce_w':>5s} {'n_all':>6s} {'allH1':>7s} {'allH5':>7s}  {'n_str':>6s} {'strH1':>7s} {'strH5':>7s}")
    print("-" * 65)
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
        r = evaluate_with_ce(w)
        print(f"{r['ce_weight']:>5.2f} {r['n_all']:>6d} {r['all_h1']:>7.3f} {r['all_h5']:>7.3f}  "
              f"{r['n_strong']:>6d} {r['strong_h1']:>7.3f} {r['strong_h5']:>7.3f}")


if __name__ == "__main__":
    main()
