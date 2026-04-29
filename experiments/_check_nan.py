import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

# nan-anchor pending: how many have feature signal?
total = 0
buckets = Counter()
samples_by_sig = {}
for ann in (ROOT/'output').rglob('*_annotation.json'):
    ftr = ann.with_name(ann.name.replace('_annotation.json','_feature_result.json'))
    if not ftr.exists(): continue
    with open(ann,'r',encoding='utf-8') as f: a = json.load(f)
    with open(ftr,'r',encoding='utf-8') as f: fd = json.load(f)
    fmap = {g['group_id']: g.get('features',{}) for g in fd}
    for g in a['groups']:
        if g.get('gt_is_true_positive') is not None: continue
        img = (g.get('anchor',{}).get('Image','') or '').strip().lower()
        if img and img != 'nan': continue
        total += 1
        f = fmap.get(g['group_id'], {})
        has_cmd = bool(f.get('command_script',{}).get('entries'))
        has_pc = bool(f.get('execution_context',{}).get('process_chains'))
        has_reg = bool(f.get('persistence',{}).get('registry_signals') or f.get('persistence',{}).get('registry_noise'))
        has_net = bool(f.get('network',{}).get('connections'))
        scenario = a.get('scenario','?')
        rule = g.get('rule_technique_id','?')
        sig = (scenario.split('_2')[0][:40], rule, 'cmd' if has_cmd else 'no-cmd',
               'pc' if has_pc else '-', 'reg' if has_reg else '-', 'net' if has_net else '-')
        buckets[sig] += 1
        if sig not in samples_by_sig and has_cmd:
            entries = f.get('command_script',{}).get('entries') or []
            samples_by_sig[sig] = (g['group_id'], [
                f"{e.get('image','?').split(chr(92))[-1]}: {(e.get('cmdline','') or '')[:80]}"
                for e in entries[:2]
            ])

print(f'total nan pending: {total}')
print(f'unique buckets: {len(buckets)}')
print('\nTop 25 nan buckets (scenario, rule, cmd, pc, reg, net):')
for sig, n in buckets.most_common(25):
    print(f'  [{n:>3}] {sig}')
    if sig in samples_by_sig:
        gid, cmds = samples_by_sig[sig]
        for c in cmds:
            print(f'         sample({gid}): {c[:100]}')
