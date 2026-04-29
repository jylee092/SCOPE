import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
anchors = Counter()
sigs = Counter()
for ann in (ROOT/'output').rglob('*_annotation.json'):
    with open(ann,'r',encoding='utf-8') as f: a = json.load(f)
    for g in a['groups']:
        if g.get('gt_is_true_positive') is not None: continue
        img = (g.get('anchor',{}).get('Image','') or '')
        b = img.replace('\\','/').split('/')[-1].lower() or 'nan'
        anchors[b] += 1
        sigs[(b, g.get('rule_technique_id','?'))] += 1

print('Top remaining pending anchors:')
for k,v in anchors.most_common(25):
    print(f'  {v:>4} {k}')
print('\nTop pending signatures (anchor, rule):')
for k,v in sigs.most_common(30):
    print(f'  {v:>4} {k}')
