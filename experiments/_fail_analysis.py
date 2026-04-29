"""Failure analysis — categorize TP labels by confidence and measure plausibility."""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.attack_flows import get_flow, all_acceptable_tids

OUT = ROOT / 'output'

stats = {
    'anchor-kw':       {'tp': 0, 'hit': 0},
    'sample-kw+ctx':   {'tp': 0, 'hit': 0},
    'rule-exact':      {'tp': 0, 'hit': 0},
    'rule-fam-strong': {'tp': 0, 'hit': 0},
    'other':           {'tp': 0, 'hit': 0},
}

noise_anchors = {
    'svchost.exe', 'lsass.exe', 'wmiprvse.exe', 'backgroundtaskhost.exe',
    'msmpeng.exe', 'services.exe', 'csrss.exe', 'smss.exe', 'dllhost.exe',
    'taskhostw.exe', 'system', 'searchindexer.exe', 'searchprotocolhost.exe',
    'msiexec.exe', 'wininit.exe', 'winlogon.exe', 'explorer.exe',
    'sppsvc.exe', 'fontdrvhost.exe', 'wsqmcons.exe',
}
noise_groups = 0
noise_tp = 0

# specific MISS cases by rule
miss_by_rule = Counter()
miss_by_pred = Counter()


def fam(a, b):
    if not a or not b:
        return False
    return a == b or a.split('.')[0] == b.split('.')[0]


def basename(path):
    if not path:
        return ''
    p = path.lower()
    for sep in ('\\', '/'):
        if sep in p:
            p = p.rsplit(sep, 1)[-1]
    return p


for ann in sorted(OUT.rglob('*_annotation.json')):
    scenario = ann.parent.name
    flow = get_flow(scenario)
    if not flow:
        continue
    acceptable = set(all_acceptable_tids(flow))
    for a in list(acceptable):
        acceptable.add(a.split('.')[0])

    with open(ann, encoding='utf-8') as f:
        ad = json.load(f)

    anchor_by_gid = {g['group_id']: (g.get('anchor') or {}) for g in ad['groups']}
    reason_by_gid = {g['group_id']: (g.get('gt_notes', '').replace('auto: ', '')) for g in ad['groups']}
    rule_by_gid = {g['group_id']: g.get('rule_technique_id', '') for g in ad['groups']}
    tp_gids = {g['group_id'] for g in ad['groups'] if g.get('gt_is_true_positive')}

    for g in ad['groups']:
        anchor = (g.get('anchor') or {}).get('Image', '') or ''
        img = basename(anchor)
        if img in noise_anchors:
            noise_groups += 1
            if g.get('gt_is_true_positive'):
                noise_tp += 1

    stem = ann.name.replace('_annotation.json', '')
    ttp_fp = ann.with_name(f'{stem}_ttp_mapping.json')
    if not ttp_fp.exists():
        continue
    with open(ttp_fp, encoding='utf-8') as f:
        ttp = json.load(f)

    for r in ttp:
        gid = r['group_id']
        if gid not in tp_gids:
            continue
        reason = reason_by_gid.get(gid, 'other')
        if reason.startswith('anchor-kw'):
            cat = 'anchor-kw'
        elif reason.startswith('sample-kw'):
            cat = 'sample-kw+ctx'
        elif reason.startswith('rule-exact'):
            cat = 'rule-exact'
        elif reason.startswith('rule-fam'):
            cat = 'rule-fam-strong'
        else:
            cat = 'other'

        cands = r.get('similar_techniques', [])[:5]
        ranked = [c['technique_id'] for c in cands]
        plausible = any(fam(p, a) for p in ranked for a in acceptable)

        stats[cat]['tp'] += 1
        if plausible:
            stats[cat]['hit'] += 1
        else:
            miss_by_rule[rule_by_gid[gid]] += 1
            miss_by_pred[ranked[0] if ranked else 'NONE'] += 1


print("=== TP 라벨 품질별 plausibility ===")
print(f"{'category':<22s} {'n_tp':>6s} {'hit':>6s} {'rate':>7s}")
for cat, s in stats.items():
    rate = s['hit'] / s['tp'] if s['tp'] else 0.0
    print(f"{cat:<22s} {s['tp']:>6d} {s['hit']:>6d} {rate:>7.3f}")

print()
print(f"=== 노이즈 프로세스 anchor 그룹 ===")
print(f"  노이즈 앵커 총 그룹: {noise_groups}")
print(f"  중 TP 라벨된 것:    {noise_tp}")

print()
print("=== MISS rule TID TOP ===")
for tid, c in miss_by_rule.most_common(10):
    print(f"  {tid}: {c}")

print()
print("=== MISS predicted top-1 TOP ===")
for tid, c in miss_by_pred.most_common(10):
    print(f"  pred={tid}: {c}")
