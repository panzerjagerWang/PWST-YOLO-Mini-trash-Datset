"""Aggregate the controlled TrashCan re-run (summary_trashcan.jsonl of the slave) into results/trashcan_rerun.json.
environment: RUNS_REMOTE=user@host:/path/to/revision_runs for --fetch.   usage: python scripts/tc_aggregate.py [--fetch]"""
import json
import os
import statistics as st
import subprocess
import sys

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(H, 'scripts'))
from statutil import welch_p  # noqa: E402

RAW = os.path.join(H, 'results', 'raw', 'summary_trashcan.jsonl')
if '--fetch' in sys.argv:
    subprocess.run(['scp', os.environ['RUNS_REMOTE'] + '/summary_trashcan.jsonl', RAW], check=True)
rows = [json.loads(ln) for ln in open(RAW, encoding='utf8') if ln.strip()]
out = {}
# round 1 (paper recipe, seeds 0-2), round 2 (gin0 with the paper recipe; light-recipe pair, seeds 0-2) and the final test (light-recipe pair, seeds 10-14)
ARMS = {'base': ('tcdet_base', (0, 1, 2)), 'pwst': ('tcdet_pwst', (0, 1, 2)), 'gin': ('tcdet_gin', (0, 1, 2)), 'gin0': ('tcdet_gin0', (0, 1, 2)),
        'lbase': ('tclight_base', (0, 1, 2)), 'lgin': ('tclight_gin5', (0, 1, 2)),
        'lbase_f': ('tclight_base', (10, 11, 12, 13, 14)), 'lgin_f': ('tclight_gin5', (10, 11, 12, 13, 14))}
for key, (name, seeds) in ARMS.items():
    rs = [r for r in rows if r['name'] == name and r['seed'] in seeds]
    out[key] = {'n': len(rs), 'seeds': sorted(r['seed'] for r in rs), 'params_M': rs[0]['params_M'] if rs else None, 'GFLOPs': rs[0]['GFLOPs'] if rs else None}
    for sp in ('val', 'test'):
        for m in ('P', 'R', 'mAP50', 'mAP50-95'):
            v = [r[sp][m] for r in rs]
            out[key][sp + '_' + m] = {'mean': round(st.mean(v), 2), 'sd': round(st.stdev(v), 2) if len(v) > 1 else None, 'values': v}
for new, ref, tag in (('pwst', 'base', ''), ('gin', 'base', 'gin_'), ('gin0', 'base', 'gin0_'), ('lgin', 'lbase', 'lgin_'), ('lgin_f', 'lbase_f', 'lgin_f_')):
    if out[new]['n'] < 2 or out[ref]['n'] < 2:
        continue
    for sp in ('val', 'test'):
        for m in ('P', 'R', 'mAP50', 'mAP50-95'):
            a = out[ref][sp + '_' + m]['values']
            b = out[new][sp + '_' + m]['values']
            out['diff_%s%s_%s' % (tag, sp, m)] = {'delta': round(st.mean(b) - st.mean(a), 2), 'p': round(welch_p(b, a), 4)}
json.dump(out, open(os.path.join(H, 'results', 'trashcan_rerun.json'), 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('n', 'delta', 'p', 'test_mAP50', 'test_mAP50-95')}) for k, v in out.items()}, indent=1)[:2500])
