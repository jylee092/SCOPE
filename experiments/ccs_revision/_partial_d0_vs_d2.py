"""Incremental fair comparison: for whichever scope_d0_drop25 viterbi files exist
so far, score D=0 AND the matching D=2 (scope_drop25) on the SAME subset."""
import json, sys
from pathlib import Path
from statistics import mean
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from experiments.attack_flows import get_flow
from experiments.chain_align import evaluate_chain_alignment

base = config.OUTPUT_BASE_DIR / "_robustness"
d0root = base / "scope_d0_drop25_seed0"
d2root = base / "scope_drop25_seed0"

def score(root, stem, rel):
    # d0 file: <rel>/<stem>_viterbi.json ; d2 file: <rel>/result? actually scope_drop25 stores *_viterbi.json too
    f = root / rel / f"{stem}_viterbi.json"
    if not f.exists():
        return None
    bd = json.load(open(f, encoding="utf-8"))
    flow = get_flow(stem)
    if not flow or not bd:
        return 0.0 if flow else None
    return evaluate_chain_alignment(stem, bd, ref_flow=flow).get("technique_lcs_norm", 0.0)

d0s, d2s, done = [], [], []
for ds in sorted(config.DATASET_FOLDER.rglob("*.json")):
    rel = ds.relative_to(config.DATASET_FOLDER).with_suffix("")
    stem = ds.stem
    s0 = score(d0root, stem, rel)
    if s0 is None:
        continue
    s2 = score(d2root, stem, rel)
    if s2 is None:
        continue
    d0s.append(s0); d2s.append(s2); done.append((stem, s0, s2))

print(f"matched subset n={len(done)}")
if done:
    print(f"  D=0 @25% (partial): {mean(d0s):.4f}")
    print(f"  D=2 @25% (partial): {mean(d2s):.4f}   [full-set D2 = 0.49]")
    print(f"  delta (D0-D2) = {mean(d0s)-mean(d2s):+.4f}")
    for stem, a, b in done:
        print(f"    {stem[:45]:45s} D0={a:.2f} D2={b:.2f}")
