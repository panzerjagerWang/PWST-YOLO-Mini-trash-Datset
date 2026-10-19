"""Reproduce the pre-registered test of the gated input standardization (GIN) from the raw result files.

Hypothesis (written to the experiment queue before the runs, see ../preregistrations.txt): GIN alone (x_gin5) raises the mean mAP50 on the nine named
corruptions (blur, contrast reduction, haze) of YOLOv11n with light augmentation.  New seeds 90-99, comparison with YOLOv11n light augmentation
(seeds 0-9), Welch two-sided test, level 0.05 (one comparison).   usage: python scripts/analyze_gin.py [results_dir]"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from statutil import welch_p  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'results', 'raw')
rows = [json.loads(l) for l in open(os.path.join(R, 'summary.jsonl'), encoding='utf8') if l.strip()]
rob = json.load(open(os.path.join(R, 'robust2.json')))
NAMED = ['blur_1', 'blur_2', 'blur_3', 'contrast_0.7', 'contrast_0.5', 'contrast_0.3', 'haze_0.8', 'haze_0.6', 'haze_0.4']
EXTRA = ['motion_9', 'motion_17', 'motion_25', 'defocus_2', 'defocus_4', 'defocus_6', 'jpeg_30', 'jpeg_15', 'jpeg_8', 'redloss_0.8', 'redloss_0.6', 'redloss_0.4']


def clean(name, lo, hi, key='mAP50'):
    return [d['test'][key] for d in rows if d['name'] == name and lo <= d['seed'] <= hi]


def robust(prefix, lo, hi, conds):
    return [st.mean(rob['%s_s%d' % (prefix, s)][c] for c in conds) for s in range(lo, hi + 1) if '%s_s%d' % (prefix, s) in rob]


def line(label, a, b):
    print('%-34s n=%2d vs %2d   %5.2f +- %4.2f  vs  %5.2f +- %4.2f   difference %+.2f   Welch p = %.4f' % (label, len(a), len(b), st.mean(a), st.stdev(a), st.mean(b), st.stdev(b), st.mean(a) - st.mean(b), welch_p(a, b)))


print('YOLOv11n + GIN (x_gin5, light augmentation, seeds 90-99)  vs  YOLOv11n (light augmentation, seeds 0-9)')
line('PRIMARY: nine named corruptions', robust('x_gin5_la', 90, 99, NAMED), robust('base_la', 0, 9, NAMED))
line('twelve additional corruptions', robust('x_gin5_la', 90, 99, EXTRA), robust('base_la', 0, 9, EXTRA))
line('clean test mAP50', clean('x_gin5_lightaug', 90, 99), clean('base_lightaug', 0, 9))
line('clean test mAP50-95', clean('x_gin5_lightaug', 90, 99, 'mAP50-95'), clean('base_lightaug', 0, 9, 'mAP50-95'))
