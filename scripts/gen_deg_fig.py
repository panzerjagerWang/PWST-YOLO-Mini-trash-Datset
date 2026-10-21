"""Qualitative figure under a synthetic corruption, YOLOv11n against GIN (same recipe), with the per-image counts over all seeds.
usage: python gen_deg_fig.py OUT_DIR CORRUPTION BASE=w1,w2,... GIN=w1,w2,...      (environment: TEST_DIR, TILE=450x300, N=6)
A prediction (confidence >= 0.25) detects an annotated object if it has the same class and IoU >= 0.5 with it (each object matched once).
For every test image the number of detected objects is averaged over the seeds of each model; the figure uses the first weights of each list and
shows N images equally spaced among the images on which GIN detects more objects than YOLOv11n with these weights.
Rows: corrupted image with the ground truth, YOLOv11n, GIN. Green: detected, dashed red: missed, dashed yellow: false positive."""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

R = os.environ.get('JOE_ROOT', '/media/w/DATA1/Paper/JOE_paper')      # directory that contains Code/ and revision_runs/
C = os.environ.get('CODE_DIR', R + '/Code')
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402


def haze(im, t):
    A = np.array([0.8, 0.75, 0.55]) * 255
    return np.clip(im.astype(np.float32) * t + A * (1 - t), 0, 255).astype(np.uint8)


def contrast(im, a):
    m = im.mean()
    return np.clip(a * (im.astype(np.float32) - m) + m, 0, 255).astype(np.uint8)


CORR = {'haze': haze, 'contrast': contrast}
out, corr = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
kind, level = corr.split('_')
fn = lambda im: CORR[kind](im, float(level))      # same functions as eval_robustness2.py (BGR images)
W = dict(a.split('=', 1) for a in sys.argv[3:])
W = {k: v.split(',') for k, v in W.items()}
D = os.environ.get('TEST_DIR', R + '/revision_runs/trashcan_split/test')
N = int(os.environ.get('N', '6'))
SZ = tuple(int(v) for v in os.environ.get('TILE', '450x300').split('x'))
names = sorted(os.listdir(D + '/images'))


def gt_of(n, w0, h0):
    p = D + '/labels/' + os.path.splitext(n)[0] + '.txt'
    rows = []
    if os.path.exists(p):
        for ln in open(p):
            v = ln.split()
            if len(v) >= 5:
                c = int(v[0])
                x, y, w, h = [float(t) for t in v[1:5]]
                rows.append((c, (x - w / 2) * w0, (y - h / 2) * h0, (x + w / 2) * w0, (y + h / 2) * h0))
    return rows


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def match(preds, gts):
    used, fps = set(), []
    hit = []
    for c, x0, y0, x1, y1 in preds:
        j = [k for k, g in enumerate(gts) if g[0] == c and k not in used and iou((x0, y0, x1, y1), g[1:]) >= 0.5]
        if j:
            used.add(j[0])
            hit.append((x0, y0, x1, y1))
        else:
            fps.append((x0, y0, x1, y1))
    return hit, fps, [g[1:] for k, g in enumerate(gts) if k not in used]


imgs = {}
for n in names:
    im = cv2.imread(D + '/images/' + n)
    h0, w0 = im.shape[:2]
    g = gt_of(n, w0, h0)
    if g:
        imgs[n] = (fn(im), g)
counts = {k: {n: [] for n in imgs} for k in W}
fpc = {k: {n: [] for n in imgs} for k in W}
first = {}
for k, ws in W.items():
    for wi, wpath in enumerate(ws):
        model = YOLO(wpath)
        for n, (im, g) in imgs.items():
            r = model.predict(im, conf=0.25, imgsz=640, verbose=False, device='cpu')[0]
            preds = [(int(b.cls), *[float(t) for t in b.xyxy[0]]) for b in r.boxes]
            hit, fps, miss = match(preds, g)
            counts[k][n].append(len(hit))
            fpc[k][n].append(len(fps))
            if wi == 0:
                first.setdefault(k, {})[n] = (hit, fps, miss)
        print(k, wi, 'done', flush=True)
mb = {n: float(np.mean(counts['BASE'][n])) for n in imgs}
mg = {n: float(np.mean(counts['GIN'][n])) for n in imgs}
ngt = sum(len(g) for _, g in imgs.values())
stats = {'corruption': corr, 'test_images_with_objects': len(imgs), 'objects': ngt, 'seeds': {k: len(v) for k, v in W.items()},
         'images_gin_more': sum(1 for n in imgs if mg[n] > mb[n] + 1e-9), 'images_gin_fewer': sum(1 for n in imgs if mg[n] < mb[n] - 1e-9),
         'detected_share_base': 100.0 * sum(mb.values()) / ngt, 'detected_share_gin': 100.0 * sum(mg.values()) / ngt,
         'fp_per_image_base': float(np.mean([np.mean(fpc['BASE'][n]) for n in imgs])), 'fp_per_image_gin': float(np.mean([np.mean(fpc['GIN'][n]) for n in imgs]))}
adv = [n for n in imgs if len(first['GIN'][n][0]) > len(first['BASE'][n][0])]
stats['images_gin_more_first_seed'] = len(adv)
stats['images_gin_fewer_first_seed'] = sum(1 for n in imgs if len(first['GIN'][n][0]) < len(first['BASE'][n][0]))
step = max(1, len(adv) // N)
pick = [adv[i * step + step // 2] for i in range(min(N, len(adv)))]
stats['images'] = pick
stats['per_image'] = {n: {'gt': len(imgs[n][1]), 'base': [len(x) for x in first['BASE'][n]], 'gin': [len(x) for x in first['GIN'][n]]} for n in pick}


def dashed(d, box, color, width=4, dash=8):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = max(abs(bx - ax), abs(by - ay), 1)
        for k in range(0, int(L // dash), 2):
            t0, t1 = k * dash / L, min((k + 1) * dash / L, 1)
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=color, width=width)


tiles = {'GT': [], 'BASE': [], 'GIN': []}
for n in pick:
    im, g = imgs[n]
    base = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    t = base.copy()
    d = ImageDraw.Draw(t)
    for _, x0, y0, x1, y1 in g:
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=3)
    tiles['GT'].append(t.resize(SZ))
    for k in ('BASE', 'GIN'):
        hit, fps, miss = first[k][n]
        t = base.copy()
        d = ImageDraw.Draw(t)
        for b in hit:
            d.rectangle(list(b), outline=(0, 220, 0), width=3)
        for b in fps:
            dashed(d, b, (255, 215, 0))
        for b in miss:
            dashed(d, b, (255, 0, 0))
        tiles[k].append(t.resize(SZ))
canvas = Image.new('RGB', (SZ[0] * len(pick) + 6 * (len(pick) - 1), SZ[1] * 3 + 12), 'white')
for ri, rk in enumerate(('GT', 'BASE', 'GIN')):
    for ci in range(len(pick)):
        canvas.paste(tiles[rk][ci], (ci * (SZ[0] + 6), ri * (SZ[1] + 6)))
canvas.save(out + '/degraded_qualitative.png')
json.dump(stats, open(out + '/degraded_qualitative.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != 'per_image'}))
