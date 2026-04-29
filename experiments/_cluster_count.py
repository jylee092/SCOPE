import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

total_pending = 0
total_clusters = 0
for ann in sorted((ROOT/'output').rglob('*_annotation.json')):
    ftr = ann.with_name(ann.name.replace('_annotation.json','_feature_result.json'))
    if not ftr.exists():
        continue
    with open(ann,'r',encoding='utf-8') as f: a = json.load(f)
    with open(ftr,'r',encoding='utf-8') as f: fd = json.load(f)
    fmap = {g['group_id']: g for g in fd}

    clusters = Counter()
    pending = 0
    for g in a['groups']:
        if g.get('gt_is_true_positive') is not None:
            continue
        pending += 1
        img = (g.get('anchor',{}).get('Image','') or '')
        a_img = img.replace('\\','/').split('/')[-1].lower() or 'nan'
        rule = g.get('rule_technique_id','?')
        f = fmap.get(g['group_id'],{}).get('features',{})
        has_cmd = 'cmd' if f.get('command_script',{}).get('entries') else 'no-cmd'
        chains = f.get('execution_context',{}).get('process_chains') or []
        lsass = any('lsass' in str(c.get('child_image','')).lower() for c in chains)
        sig = (a_img, rule, has_cmd, 'lsass' if lsass else 'no-lsass')
        clusters[sig] += 1

    if pending > 0:
        rel = '/'.join(ann.relative_to(ROOT/'output').as_posix().split('/')[:3])
        top3 = ', '.join([f'{k}:{v}' for k,v in clusters.most_common(3)])
        print(f'{rel[-58:]:<58} pend={pending:>4} clust={len(clusters):>3}  top={top3[:60]}')
        total_pending += pending
        total_clusters += len(clusters)

print(f'\nTOTAL pending={total_pending}  unique clusters={total_clusters}')
