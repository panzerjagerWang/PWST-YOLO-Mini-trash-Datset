"""TrashCan 1.0 (instance version, 22 classes) -> YOLO folders for pre-training, with leakage control.

Images whose 32x32 grey-thumbnail correlation (z-scored, the measure of scripts/trashcan_overlap.py) with ANY mini-Trash validation or test image is >= THR
are excluded, so that no pre-training image is a (near-)duplicate of an image that mini-Trash evaluates on.
Writes <OUT>/{train,val}/{images,labels} (images are symlinks), <OUT>.yaml and <OUT>/excluded.json.
usage: python make_trashcan_yolo.py"""
import json
import os
import zipfile

import numpy as np
from PIL import Image

R = '/media/w/DATA1/Paper/JOE_paper'
TC = R + '/trashcan'
IMG = TC + '/imgs'
MT = R + '/Code/dataset'
OUT = R + '/revision_runs/trashcan_yolo'
THR = float(os.environ.get("THR", "0.95"))

z = zipfile.ZipFile(TC + '/dataset.zip')
cats = None
recs = {}
for sp in ('train', 'val'):
    d = json.loads(z.read('dataset/instance_version/instances_%s_trashcan.json' % sp))
    if cats is None:
        cats = sorted(d['categories'], key=lambda c: c['id'])
    cid = {c['id']: i for i, c in enumerate(cats)}
    imgs = {im['id']: im for im in d['images']}
    ann = {}
    for a in d['annotations']:
        im = imgs[a['image_id']]
        x, y, w, h = a['bbox']
        W, H = im['width'], im['height']
        xc, yc = (x + w / 2.0) / W, (y + h / 2.0) / H
        ann.setdefault(a['image_id'], []).append('%d %.6f %.6f %.6f %.6f' % (cid[a['category_id']], xc, yc, w / W, h / H))
    recs[sp] = [(im['file_name'], ann.get(i, [])) for i, im in sorted(imgs.items())]
print({k: len(v) for k, v in recs.items()}, 'classes', len(cats))


def feat(p):
    a = np.asarray(Image.open(p).convert('L').resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


mt = []
for s in ('val', 'test'):
    d = os.path.join(MT, s, 'images')
    mt += [os.path.join(d, f) for f in sorted(os.listdir(d))]
Fm = np.stack([feat(p) for p in mt])
print('mini-Trash val+test images:', len(mt))

excl = {}
for sp in ('train', 'val'):
    kept = 0
    os.makedirs(OUT + '/%s/images' % sp, exist_ok=True)
    os.makedirs(OUT + '/%s/labels' % sp, exist_ok=True)
    for fn, lines in recs[sp]:
        src = IMG + '/%s__%s' % (sp, fn)
        c = float((Fm @ feat(src) / Fm.shape[1]).max())
        if c >= THR:
            excl['%s/%s' % (sp, fn)] = round(c, 3)
            continue
        dst = OUT + '/%s/images/%s' % (sp, fn)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        open(OUT + '/%s/labels/%s.txt' % (sp, os.path.splitext(fn)[0]), 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        kept += 1
    print(sp, 'kept', kept, 'excluded', sum(1 for k in excl if k.startswith(sp + '/')))
json.dump({'threshold': THR, 'excluded': excl}, open(OUT + '/excluded.json', 'w'), indent=1)
with open(OUT + '.yaml', 'w') as f:
    f.write('path: %s\ntrain: train/images\nval: val/images\nnames:\n' % OUT)
    for i, c in enumerate(cats):
        f.write('  %d: %s\n' % (i, c['name']))
print('written', OUT)
