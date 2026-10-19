"""How close are the mini-Trash test images to the TrashCan images used for pre-training (after the >= 0.95 exclusion), compared with their closeness
to the mini-Trash training images?  Measure: maximum correlation of 32 x 32 z-scored grey thumbnails (the measure of scripts/trashcan_overlap.py).
Writes revision_runs/tc_leak_check.json.   usage: [TCDIR=trashcan_yolo_v] python tc_leak_check.py"""
import json
import os

import numpy as np
from PIL import Image

R = '/media/w/DATA1/Paper/JOE_paper'
MT = R + '/Code/dataset'
TCDIR = os.environ.get('TCDIR', 'trashcan_yolo')
TCY = R + '/revision_runs/' + TCDIR


def feat(p):
    a = np.asarray(Image.open(p).convert('L').resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def stack(d):
    fs = sorted(os.listdir(d))
    return np.stack([feat(os.path.join(d, f)) for f in fs])


Ft = stack(MT + '/test/images')
Fv = stack(MT + '/val/images')
Fm = stack(MT + '/train/images')
Fc = np.concatenate([stack(TCY + '/train/images'), stack(TCY + '/val/images')])
n = Ft.shape[1]
out = {'n_test': len(Ft), 'n_val': len(Fv), 'n_minitrash_train': len(Fm), 'n_trashcan_kept': len(Fc)}
for name, A in (('test', Ft), ('val', Fv)):
    cm = (A @ Fm.T / n).max(1)
    cc = (A @ Fc.T / n).max(1)
    for thr in (0.85, 0.90, 0.95):
        out['%s_to_minitrash_train_ge_%.2f' % (name, thr)] = float((cm >= thr).mean())
        out['%s_to_trashcan_kept_ge_%.2f' % (name, thr)] = float((cc >= thr).mean())
    out['%s_to_minitrash_train_median_max' % name] = float(np.median(cm))
    out['%s_to_trashcan_kept_median_max' % name] = float(np.median(cc))
json.dump(out, open(R + '/revision_runs/tc_leak_check%s.json' % ('' if TCDIR == 'trashcan_yolo' else '_' + TCDIR), 'w'), indent=1)
print(json.dumps(out, indent=1))
