"""Analysis of the TrashCan-initialisation experiment (the design was written to the queue file before the first run of it).
Arms (seeds 20-27 each, all fine-tuned on mini-Trash with the protocol of Section IV-B):
  B_TC  YOLOv11n, full augmentation           L_TC  YOLOv11n, light augmentation          S_TC  SRAW-TM + AmpMix, light augmentation
Primary endpoint: clean test mAP50 (Welch, two-sided). Confirmatory comparisons C1 B_TC vs full augmentation, C2 S_TC vs full augmentation, C3 S_TC vs L_TC,
Bonferroni level 0.05/3. Everything else is descriptive.   usage: python scripts/analyze_tc.py [--json]"""
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leanlib as L  # noqa: E402
from statutil import _ibeta, welch_p  # noqa: E402

S, R = L.S, L.R
GROUPS = [
    ('paper', 'YOLOv11n, full augmentation, COCO init', ['yolo11n_base'], ['base']),
    ('la', 'YOLOv11n, light aug., COCO init', ['base_lightaug', 'base_lightaug_rep'], ['base_la', 'base_la_rep']),
    ('amp', 'YOLOv11n + AmpMix, light aug., COCO init', ['x_amp_lightaug', 'x_amp_lightaug_rep', 'x_amp_lightaug_conf'], ['x_amp_la', 'x_amp_la_rep', 'x_amp_la_conf']),
    ('S', 'SRAW-TM + AmpMix, light aug., COCO init', ['x_A_amp_lightaug', 'x_A_amp_lightaug_rep', 'x_A_amp_lightaug_conf'], ['x_A_amp_la', 'x_A_amp_la_rep', 'x_A_amp_la_conf']),
    ('B_TC', 'YOLOv11n, full augmentation, TrashCan init', ['tc_yolo11n_base'], ['tc_base']),
    ('L_TC', 'YOLOv11n, light aug., TrashCan init', ['tc_base_lightaug'], ['tc_base_la']),
    ('S_TC', 'SRAW-TM + AmpMix, light aug., TrashCan init', ['tc_x_A_amp_lightaug'], ['tc_x_A_amp_la']),
]
ALPHA = 0.05 / 3


def vals(g, sp, k):
    names = [n for gg in GROUPS if gg[0] == g for n in gg[2]]
    return [S[n][s][sp][k] for n in names if n in S for s in sorted(S[n])]


def rvals(g, conds):
    names = [n for gg in GROUPS if gg[0] == g for n in gg[3]]
    return [st.mean(R[n][s][c] for c in conds) for n in names if n in R for s in sorted(R[n])]


def tq(df, level=0.95):
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        p = _ibeta(df / 2, 0.5, df / (df + mid * mid))          # two-sided p of |t| = mid
        lo, hi = (mid, hi) if p > 1 - level else (lo, mid)
    return (lo + hi) / 2


def cmp(a, b):
    """difference of means, Welch p, 95% CI"""
    if len(a) < 2 or len(b) < 2:
        return None
    d = st.mean(a) - st.mean(b)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = tq(df) * se
    return d, welch_p(a, b), d - h, d + h


def fmt(x):
    return '%.1f' % x


def main():
    out = {}
    print('%-46s %2s  %-12s %-12s %-12s %-12s' % ('group', 'n', 'mAP50', 'mAP50-95', 'nine named', 'twelve add.'))
    for g, lab, _, _ in GROUPS:
        a = vals(g, 'test', 'mAP50')
        if not a:
            print('%-46s  0  (not available)' % lab)
            continue
        b = vals(g, 'test', 'mAP50-95')
        ri, ro = rvals(g, L.IN), rvals(g, L.OUT)
        pm = lambda x: ('%.1f+-%.1f' % (st.mean(x), st.stdev(x))) if len(x) > 1 else ('%.1f' % x[0] if x else '--')
        print('%-46s %2d  %-12s %-12s %-12s %-12s' % (lab, len(a), pm(a), pm(b), pm(ri), pm(ro)))
        out[g] = {'n': len(a), 'mAP50': (st.mean(a), st.stdev(a) if len(a) > 1 else None), 'mAP50-95': (st.mean(b), st.stdev(b) if len(b) > 1 else None),
                  'nine': st.mean(ri) if ri else None, 'twelve': st.mean(ro) if ro else None, 'n_rob': len(ro)}
    print()
    comps = [('C1', 'B_TC', 'paper'), ('C2', 'S_TC', 'paper'), ('C3', 'S_TC', 'L_TC'),
             ('d1', 'L_TC', 'la'), ('d2', 'B_TC', 'L_TC'), ('d3', 'S_TC', 'S'), ('d4', 'S_TC', 'B_TC'), ('d5', 'S_TC', 'la')]
    print('%-4s %-6s vs %-6s | %-28s | %-28s | %-28s' % ('id', 'a', 'b', 'mAP50  diff [95% CI] p', 'mAP50-95 diff [95% CI] p', 'twelve add. diff [95% CI] p'))
    out['cmp'] = {}
    for cid, ga, gb in comps:
        row = []
        res = {}
        for lab, ya, yb in (('mAP50', vals(ga, 'test', 'mAP50'), vals(gb, 'test', 'mAP50')), ('mAP50-95', vals(ga, 'test', 'mAP50-95'), vals(gb, 'test', 'mAP50-95')),
                            ('corr', rvals(ga, L.OUT), rvals(gb, L.OUT))):
            c = cmp(ya, yb) if ya and yb else None
            res[lab] = c
            row.append('%+.2f [%+.2f,%+.2f] p=%.3f' % (c[0], c[2], c[3], c[1]) if c else 'n/a')
        out['cmp'][cid] = res
        print('%-4s %-6s vs %-6s | %-28s | %-28s | %-28s' % (cid, ga, gb, row[0], row[1], row[2]))
    print('\nconfirmatory level (Bonferroni, 3 comparisons): %.4f' % ALPHA)
    for cid in ('C1', 'C2', 'C3'):
        c = out['cmp'][cid]['mAP50']
        if c:
            print('  %s: %s' % (cid, 'MET' if c[1] < ALPHA and c[0] > 0 else 'not met'))
    if '--json' in sys.argv:
        json.dump(out, open(os.path.join(L.H, 'results', 'tc_init_analysis.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
