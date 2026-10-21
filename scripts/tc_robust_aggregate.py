"""Aggregate the TrashCan corruption evaluation (robust_trashcan.json of the slave) into results/trashcan_robust.json.
Named corruptions = blur, contrast and haze (nine conditions); held-out = motion, defocus, jpeg, redloss (twelve).
environment: RUNS_REMOTE=user@host:/path/to/revision_runs for --fetch.   usage: python scripts/tc_robust_aggregate.py [--fetch]"""
import json
import os
import statistics as st
import subprocess
import sys

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(H, 'scripts'))
from statutil import welch_p  # noqa: E402

# robust_trashcan.json holds round 1 (tcdet_*), robust_tclight_{a,b,c}.json the light-recipe pair (seeds 10-14 in a and b, seeds 0-2 in c)
FILES = ('robust_trashcan.json', 'robust_tclight_a.json', 'robust_tclight_b.json', 'robust_tclight_c.json')
res = {}
for f in FILES:
    raw = os.path.join(H, 'results', 'raw', f)
    if '--fetch' in sys.argv:
        subprocess.run(['scp', os.environ['RUNS_REMOTE'] + '/' + f, raw], check=True)
    if os.path.exists(raw):
        res.update({k: v for k, v in json.load(open(raw, encoding='utf8')).items() if len(v) == 22})
NAMED = ('blur', 'contrast', 'haze')
ARMS = {'base': ('tcdet_base', (0, 1, 2)), 'pwst': ('tcdet_pwst', (0, 1, 2)), 'gin': ('tcdet_gin', (0, 1, 2)),
        'lbase': ('tclight_base', (0, 1, 2)), 'lgin': ('tclight_gin5', (0, 1, 2)),
        'lbase_f': ('tclight_base', (10, 11, 12, 13, 14)), 'lgin_f': ('tclight_gin5', (10, 11, 12, 13, 14))}
out = {}
for key, (name, seeds) in ARMS.items():
    labs = [name + '_s%d' % sd for sd in seeds if name + '_s%d' % sd in res]
    if not labs:
        continue
    named = [st.mean([v for c, v in res[k].items() if c.split('_')[0] in NAMED]) for k in labs]
    held = [st.mean([v for c, v in res[k].items() if c != 'clean' and c.split('_')[0] not in NAMED]) for k in labs]
    clean = [res[k]['clean'] for k in labs]
    out[key] = {'n': len(labs), 'labels': labs, 'per_seed': {c: [res[k][c] for k in labs] for c in res[labs[0]]}}
    for nm, v in (('clean', clean), ('named', named), ('heldout', held)):
        out[key][nm] = {'mean': round(st.mean(v), 2), 'sd': round(st.stdev(v), 2) if len(v) > 1 else None, 'values': [round(x, 2) for x in v]}
    for c in res[labs[0]]:
        if c != 'clean':
            out[key][c] = round(st.mean([res[k][c] for k in labs]), 2)
for new, ref in (('pwst', 'base'), ('gin', 'base'), ('lgin', 'lbase'), ('lgin_f', 'lbase_f')):
    if new in out and ref in out and out[new]['n'] > 1 and out[ref]['n'] > 1:
        for nm in ('clean', 'named', 'heldout'):
            a, b = out[ref][nm]['values'], out[new][nm]['values']
            out['diff_%s_%s' % (new, nm)] = {'delta': round(st.mean(b) - st.mean(a), 2), 'p': round(welch_p(b, a), 4)}
        out['per_condition_%s' % new] = {c: {'delta': round(st.mean([res[k][c] for k in out[new]['labels']]) - st.mean([res[k][c] for k in out[ref]['labels']]), 2),
                                             'p': round(welch_p([res[k][c] for k in out[new]['labels']], [res[k][c] for k in out[ref]['labels']]), 4)}
                                         for c in res[out[new]['labels'][0]]}
json.dump(out, open(os.path.join(H, 'results', 'trashcan_robust.json'), 'w'), indent=1)
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk in ('n', 'clean', 'named', 'heldout', 'delta', 'p')} if isinstance(v, dict) else v) for k, v in out.items()}, indent=1)[:2500])
