"""Data access and formatting helpers shared by the generators of the lean manuscript (build_lean_tex.py).

Everything the paper states numerically is computed from the raw result files:
  results/raw/summary.jsonl      one line per trained model (val/test P, R, mAP50, mAP50-95)         -> S[name][seed]
  results/raw/robust2.json       mAP50 of every trained model under 21 synthetic corruptions + clean   -> R[group][seed]
  results/raw/cv_results.json    clip-grouped cross-validation of the candidate modules                -> CV
  results/raw/flops.json         parameters / GMACs of the networks (nc = 3, 640 x 640)               -> FL
  results/raw/trashcan_overlap.json, results/overlap_analysis_minitrash.json                          -> overlap analyses
"""
import json
import os
import re
import statistics as st
import sys
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statutil import welch_p  # noqa: E402

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(H, 'results', 'raw')
PARTIAL = bool(os.environ.get('ALLOW_PARTIAL'))

FAM = {'blur': ['blur_1', 'blur_2', 'blur_3'], 'contrast': ['contrast_0.7', 'contrast_0.5', 'contrast_0.3'], 'haze': ['haze_0.8', 'haze_0.6', 'haze_0.4'],
       'motion': ['motion_9', 'motion_17', 'motion_25'], 'defocus': ['defocus_2', 'defocus_4', 'defocus_6'],
       'jpeg': ['jpeg_30', 'jpeg_15', 'jpeg_8'], 'redloss': ['redloss_0.8', 'redloss_0.6', 'redloss_0.4']}
IN = [c for f in ('blur', 'contrast', 'haze') for c in FAM[f]]          # the nine named corruptions (blur, contrast reduction, turbidity)
OUT = [c for f in ('motion', 'defocus', 'jpeg', 'redloss') for c in FAM[f]]   # twelve additional corruptions (fixed before the AmpMix runs)


def q(x, k):
    return str(Decimal(repr(round(x, 5))).quantize(Decimal(k), rounding=ROUND_HALF_UP))


def r1(x): return q(x, '0.1')
def r2(x): return q(x, '0.01')
def r3(x): return q(x, '0.001')


def pf(p, eq=True):
    """p value as '<0.001' / '=0.003' / '=0.47' (for use after the letter p)"""
    if p != p:
        return '=n/a'
    return '<0.001' if p < 0.001 else (('=' if eq else '') + (r3(p) if p < 0.01 else r2(p)))


def sg(x, nd=1):
    """signed number with a LaTeX minus; zero is printed without sign"""
    s = '%.*f' % (nd, abs(x))
    if float(s) == 0:
        return s
    return ('+' if x > 0 else '$-$') + s


# ---------------------------------------------------------------- data
# the two pre-registered confirmation comparisons use seeds 0-4; a replication cohort (seeds >= 5) of the second comparison is kept apart
# under the suffix '_rep' so that it is never pooled into the pre-registered analyses by accident
REP_GROUPS = ('x_amp_lightaug', 'x_A_amp_lightaug', 'x_amp_la', 'x_A_amp_la', 'pw_A_sraw57', 'pw_D_sra57', 'A', 'D')
REP_FROM = 5
REP_SIM = bool(os.environ.get('REP_SIM'))      # dry-run switch for testing the generator only: the _rep groups become copies of the main groups
# further seeds of the two baselines are kept apart from the pre-registered analyses as well (name -> (first extra seed, suffix))
# YOLOv11n with full augmentation is used with all its seeds (0-7), so it is not split.
# Replication cohorts (new seeds, decided after the first cohort) are kept apart so that they are never pooled into the first analyses by accident.
SPLIT = {'x_amp_lightaug': (5, '_rep'), 'x_A_amp_lightaug': (5, '_rep'), 'x_amp_la': (5, '_rep'), 'x_A_amp_la': (5, '_rep'),
         'base_lightaug': (5, '_rep'), 'base_la': (5, '_rep'),
         'pw_A_sraw57': (3, '_rep'), 'pw_D_sra57': (3, '_rep'), 'A': (3, '_rep'), 'D': (3, '_rep')}
# pre-registered confirmation of the final system (seeds 10-19 of AmpMix and of SRAW-TM + AmpMix, light augmentation; written to the queue before the runs):
# kept apart from the pre-registered cohort (seeds 0-4) and from the replication cohort (seeds 5-9)
CONF_FROM = int(os.environ.get('CONF_FROM', '10'))      # CONF_FROM=5 CONF_N=5 is a dry-run switch for testing the generator only
CONF_N = int(os.environ.get('CONF_N', '10'))
CONF_NAMES = ('x_amp_lightaug', 'x_A_amp_lightaug', 'x_amp_la', 'x_A_amp_la')


def split_name(name, seed):
    if name.startswith('x_') and 70 <= seed <= 74:      # round-7 screening seeds (validation-split selection): never pooled with a confirmation
        return name + '_scr'
    if name in CONF_NAMES and seed >= CONF_FROM:
        return name + '_conf'
    f = SPLIT.get(name)
    return name + f[1] if f and seed >= f[0] else name


def load_summary():
    S = {}
    for fn in ('summary.jsonl',):
        p = os.path.join(RAW, fn)
        for l in open(p, encoding='utf8'):
            if l.strip():
                d = json.loads(l)
                if not d['name'].startswith('INVALID_'):
                    n_ = split_name(d['name'], d['seed'])
                    S.setdefault(n_, {})[d['seed']] = d
                    if REP_SIM and d['name'] in REP_GROUPS and d['seed'] < REP_FROM:
                        S.setdefault(d['name'] + '_rep', {})[d['seed'] + REP_FROM] = d
    return S


def load_robust():
    rob = json.load(open(os.path.join(RAW, 'robust2.json')))
    R = {}
    for lab, d in rob.items():
        m = re.match(r'(.+)_s(\d+)$', lab)
        if m and len(d) == 22:
            g_ = split_name(m.group(1), int(m.group(2)))
            R.setdefault(g_, {})[int(m.group(2))] = d
            if REP_SIM and m.group(1) in REP_GROUPS and int(m.group(2)) < REP_FROM:
                R.setdefault(m.group(1) + '_rep', {})[int(m.group(2)) + REP_FROM] = d
    return R


S = load_summary()
R = load_robust()
FL = json.load(open(os.path.join(RAW, 'flops.json')))
CVRAW = json.load(open(os.path.join(RAW, 'cv_results.json'))) if os.path.exists(os.path.join(RAW, 'cv_results.json')) else {}
TCO = json.load(open(os.path.join(RAW, 'trashcan_overlap.json'))) if os.path.exists(os.path.join(RAW, 'trashcan_overlap.json')) else {}
OVM = json.load(open(os.path.join(H, 'results', 'overlap_analysis_minitrash.json'))) if os.path.exists(os.path.join(H, 'results', 'overlap_analysis_minitrash.json')) else {}


def has(n, k=2):
    return n in S and len(S[n]) >= k


def hasr(g, k=2):
    return g in R and len(R[g]) >= k


def v(n, sp, k='mAP50'):
    return [S[n][s][sp][k] for s in sorted(S[n])]


def mean(n, sp, k='mAP50'):
    return st.mean(v(n, sp, k))


def sd(n, sp, k='mAP50'):
    x = v(n, sp, k)
    return st.stdev(x) if len(x) > 1 else float('nan')


def ms(n, sp, k='mAP50', math=True):
    """mean and standard deviation over seeds"""
    x = v(n, sp, k)
    if len(x) == 1:
        return r1(x[0])
    t = '%s\\pm%s' % (r1(st.mean(x)), r1(st.stdev(x)))
    return '$%s$' % t if math else t


def pv(a, b, sp, k='mAP50'):
    x, y = v(a, sp, k), v(b, sp, k)
    return welch_p(x, y) if len(x) > 1 and len(y) > 1 else float('nan')


def rs(g, conds):
    """per-seed mean over the corruption conditions"""
    return [st.mean(R[g][s][c] for c in conds) for s in sorted(R[g])]


def rms(g, conds, math=True):
    x = rs(g, conds)
    if len(x) == 1:
        return r1(x[0])
    t = '%s\\pm%s' % (r1(st.mean(x)), r1(st.stdev(x)))
    return '$%s$' % t if math else t


def rp(a, b, conds):
    x, y = rs(a, conds), rs(b, conds)
    return welch_p(x, y) if len(x) > 1 and len(y) > 1 else float('nan')


def rcmp(a, b, conds):
    """(difference of means, p, 'better'|'worse'|'same')"""
    xa, xb = rs(a, conds), rs(b, conds)
    d = st.mean(xa) - st.mean(xb)
    p = welch_p(xa, xb) if len(xa) > 1 and len(xb) > 1 else float('nan')
    kind = 'better' if p < 0.05 and d > 0 else 'worse' if p < 0.05 and d < 0 else 'same'
    return d, p, kind


def T(text, **kw):
    """template substitution: <<name>> -> value (LaTeX braces and percent signs stay untouched)"""
    for k, val in kw.items():
        text = text.replace('<<%s>>' % k, str(val))
    left = re.findall(r'<<(\w+)>>', text)
    assert not left, 'unfilled placeholders: %s' % left
    return text


def splice(path, marker, block):
    """replace the text between '% >>> NAME BEGIN' and '% <<< NAME END' of a .tex file"""
    t = open(path, encoding='utf8').read()
    i, j = t.index('%% >>> %s BEGIN' % marker), t.index('%% <<< %s END' % marker)
    t = t[:i] + '%% >>> %s BEGIN (generated by scripts/build_lean_tex.py; do not edit by hand)\n' % marker + block.strip('\n') + '\n' + t[j:]
    open(path, 'w', encoding='utf8', newline='\n').write(t)
