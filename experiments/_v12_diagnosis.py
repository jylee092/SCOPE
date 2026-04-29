"""
v12 심층 진단:
(1) Description 품질 샘플 — 시나리오별 다양성
(2) Top-5 miss 20% 실패 패턴 분석
(3) Viterbi chain 엉뚱하게 고르는 원인 분석
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT))
from pipeline.evaluator import load_ground_truth
from experiments.run_eval_plausible import _is_strong_tp, tid_family_match
from experiments.attack_flows import get_flow, all_acceptable_tids


def fam(p, t):
    if not p or not t: return False
    return p == t or p.split(".")[0] == t.split(".")[0]


def match_acc(p, acc):
    return any(fam(p, a) for a in acc) if p else False


def main():
    # Collect all data
    top5_miss_cases = []       # cases where no top-5 matches acceptable
    good_desc_samples = []     # cases where top-1 correct
    bad_desc_samples = []      # cases where top-1 wrong (but acceptable is still in top-5)
    chain_issues = []          # scenarios where Viterbi chain differs from ideal

    for ann in sorted(OUT.rglob("*_annotation.json")):
        gt = load_ground_truth(ann)
        if not gt: continue
        ad = json.load(open(ann, encoding="utf-8"))
        scenario = ad.get("scenario", ann.parent.name)
        flow = get_flow(scenario)
        if not flow: continue
        acc = set(all_acceptable_tids(flow))
        for a in list(acc): acc.add(a.split(".")[0])

        stem = ann.name.replace("_annotation.json","")
        ttp_fp = ann.with_name(f"{stem}_ttp_mapping.json")
        vit_fp = ann.with_name(f"{stem}_viterbi.json")
        if not (ttp_fp.exists() and vit_fp.exists()): continue
        ttp = json.load(open(ttp_fp, encoding="utf-8"))
        vit = json.load(open(vit_fp, encoding="utf-8"))

        ttp_by_gid = {r["group_id"]: r for r in ttp}
        vit_by_gid = {b["group_id"]: b["technique_id"] for b in vit}
        grp_by_gid = {g["group_id"]: g for g in ad["groups"]}

        # Viterbi chain flow summary
        chain_tids = [b["technique_id"] for b in vit]
        ref_tids = [s["tid"] for s in flow]
        matched_ref = sum(1 for r in flow if any(fam(p, r["tid"]) or any(fam(p,a) for a in r.get("alts",[])) for p in chain_tids))
        chain_issues.append({
            "scenario": scenario,
            "ref": ref_tids,
            "chain_len": len(chain_tids),
            "chain_unique": list(Counter(chain_tids).keys())[:15],
            "matched_ref": matched_ref,
            "ref_steps": len(ref_tids),
        })

        for gid, truth in gt.items():
            if not truth["is_tp"]: continue
            if not _is_strong_tp(truth.get("notes", "")): continue
            r = ttp_by_gid.get(gid)
            grp = grp_by_gid.get(gid)
            if not r or not grp: continue
            top5 = [c["technique_id"] for c in r.get("similar_techniques",[])[:5]]
            if not top5: continue
            desc = r.get("generated_description", "")
            anchor_img = (grp.get("anchor",{}) or {}).get("Image","")
            anchor_cl = (grp.get("anchor",{}) or {}).get("CommandLine","")

            top5_hit = any(match_acc(t, acc) for t in top5)
            top1_hit = match_acc(top5[0], acc)

            if not top5_hit:
                top5_miss_cases.append({
                    "scenario": scenario[:40],
                    "gid": gid,
                    "rule_tid": grp.get("rule_technique_id",""),
                    "desc": desc[:250],
                    "top5": top5,
                    "acc": sorted(acc)[:10],
                    "anchor_img": str(anchor_img)[:60],
                    "anchor_cl": str(anchor_cl)[:120],
                })
            elif top1_hit:
                if len(good_desc_samples) < 5:
                    good_desc_samples.append({
                        "scenario": scenario[:40],
                        "gid": gid,
                        "desc": desc[:350],
                        "top1": top5[0],
                    })
            else:
                if len(bad_desc_samples) < 8:
                    bad_desc_samples.append({
                        "scenario": scenario[:40],
                        "gid": gid,
                        "rule_tid": grp.get("rule_technique_id",""),
                        "desc": desc[:350],
                        "top5": top5,
                        "acc": sorted(acc)[:8],
                    })

    print("=" * 100)
    print(f"TOP-5 MISS CASES  (n={len(top5_miss_cases)})")
    print("=" * 100)
    # Group by scenario for patterns
    by_scen = defaultdict(list)
    for c in top5_miss_cases:
        by_scen[c["scenario"]].append(c)

    for scen, cases in list(by_scen.items())[:8]:
        print(f"\n[{scen}] ({len(cases)} misses)")
        for c in cases[:2]:
            print(f"  rule={c['rule_tid']}")
            print(f"  anchor img={c['anchor_img']}")
            print(f"  anchor cl={c['anchor_cl']}")
            print(f"  DESC: {c['desc']}")
            print(f"  top5: {c['top5']}")
            print(f"  acc:  {c['acc']}")
            print()

    print()
    print("=" * 100)
    print("TOP-1 WRONG but HIT in TOP-2..5 (desc is understandable but embedding mis-ranks)")
    print("=" * 100)
    for c in bad_desc_samples:
        print(f"\n[{c['scenario']}]  rule={c['rule_tid']}")
        print(f"  DESC: {c['desc']}")
        print(f"  top5: {c['top5']}")
        print(f"  acc:  {c['acc']}")

    print()
    print("=" * 100)
    print("GOOD DESC EXAMPLES (top-1 correct)")
    print("=" * 100)
    for c in good_desc_samples:
        print(f"\n[{c['scenario']}] top-1={c['top1']}")
        print(f"  DESC: {c['desc']}")

    print()
    print("=" * 100)
    print("CHAIN STRUCTURE (reference vs predicted length & uniqueness)")
    print("=" * 100)
    for ci in chain_issues:
        if ci["matched_ref"] < ci["ref_steps"]:  # missed at least one reference step
            uniq = Counter([c["chain_unique"][0] if ci["chain_unique"] else ""] for c in []).keys()
            print(f"\n  {ci['scenario'][:45]:<45s} ref={ci['ref_steps']} matched={ci['matched_ref']} "
                  f"pred_len={ci['chain_len']}")
            print(f"    REF  tids: {ci['ref']}")
            print(f"    PRED uniq: {ci['chain_unique']}")


if __name__ == "__main__":
    main()
