"""Reproduces the '21% of steps resolved by transition rather than emission alone'
figure/caption claim: across canonical OTRF outputs, count chain steps whose
Viterbi-chosen technique differs from the emission top-1 (similar_techniques[0])."""
import json, glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

BASE = str(config.OUTPUT_BASE_DIR)
vit_files = [p for p in glob.glob(BASE + "/**/*_viterbi.json", recursive=True)
             if "ablation" not in p and "_ccs_revision" not in p
             and ("atomic" in p or "compound" in p)]

total = deviations = 0
examples = []
for vf in vit_files:
    mf = vf.replace("_viterbi.json", "_ttp_mapping.json")
    if not os.path.exists(mf):
        continue
    vit = json.load(open(vf, encoding="utf-8"))
    bd = vit.get("score_breakdown") if isinstance(vit, dict) else vit
    ttp = json.load(open(mf, encoding="utf-8"))
    rows = ttp if isinstance(ttp, list) else ttp.get("results", [])
    top1 = {r.get("group_id"): (r["similar_techniques"][0].get("technique_id"),
                                float(r["similar_techniques"][0].get("similarity", 0)))
            for r in rows if r.get("group_id") and r.get("similar_techniques")}
    for step in (bd or []):
        gid, chosen = step.get("group_id"), step.get("technique_id")
        if gid not in top1:
            continue
        total += 1
        if chosen != top1[gid][0]:
            deviations += 1
            if len(examples) < 12:
                examples.append((os.path.basename(vf).replace("_viterbi.json", ""),
                                 step.get("step"), chosen, top1[gid][0],
                                 round(top1[gid][1], 3), step.get("transition_rule")))

print(f"total chain steps : {total}")
print(f"transition overrides (chosen != emission top-1): {deviations} "
      f"({100*deviations/max(total,1):.1f}%)")
for e in examples:
    print(f"  {e[0][:34]:34} s{e[1]}  {e[2]} over {e[3]} (sim {e[4]}) [{e[5]}]")
