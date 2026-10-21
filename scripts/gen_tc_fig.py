"""Qualitative TrashCan figure from the re-trained models (runs on the slave).
Rows: ground truth and one row per model (default YOLOv11n and PWST-YOLOv11, seed 0 each).  Columns: six test images at equal spacing among the test images with at least two ground-truth objects.
Predicted boxes (conf >= 0.25) are drawn in green; missed ground-truth objects (no prediction of the same class with IoU >= 0.5) are dashed red and false positives dashed yellow.
usage: python gen_tc_fig.py OUT_DIR [LABEL=weights.pt ...]"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

R = os.environ.get('JOE_ROOT', '/media/w/DATA1/Paper/JOE_paper')      # directory that contains Code/ and revision_runs/
C = os.environ.get('CODE_DIR', R + '/Code')
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
D = os.environ.get('TEST_DIR', R + '/revision_runs/trashcan_split/test')      # mini-Trash: TEST_DIR=Code/dataset/test TILE=450x300
W = dict(a.split('=', 1) for a in sys.argv[2:]) or {'YOLOv11n': R + '/revision_runs/tcdet_base_s0/weights/best.pt', 'PWST-YOLOv11': R + '/revision_runs/tcdet_pwst_s0/weights/best.pt'}
names = sorted(os.listdir(D + '/images'))


def gt_of(n):
    p = D + '/labels/' + os.path.splitext(n)[0] + '.txt'
    rows = []
    if os.path.exists(p):
        for ln in open(p):
            v = ln.split()
            if len(v) >= 5:
                c = int(v[0])
                x, y, w, h = [float(t) for t in v[1:5]]
                rows.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return rows


cand = [n for n in names if len(gt_of(n)) >= 2]
step = len(cand) // 6
pick = [cand[i * step + step // 2] for i in range(6)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def dashed(d, box, color, width=3, dash=8):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = max(abs(bx - ax), abs(by - ay))
        n = int(L // dash)
        for k in range(0, n, 2):
            t0, t1 = k * dash / L, min((k + 1) * dash / L, 1)
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=color, width=width)


SZ = tuple(int(v) for v in os.environ.get('TILE', '400x300').split('x'))
tiles = {'GT': [], **{k: [] for k in W}}
info = {'images': pick, 'weights': W, 'stats': {}}
models = {k: YOLO(v) for k, v in W.items()}
for n in pick:
    im = Image.open(D + '/images/' + n).convert('RGB')
    w0, h0 = im.size
    gts = [(c, x0 * w0, y0 * h0, x1 * w0, y1 * h0) for c, x0, y0, x1, y1 in gt_of(n)]
    g = im.copy()
    d = ImageDraw.Draw(g)
    for c, x0, y0, x1, y1 in gts:
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=3)
    tiles['GT'].append(g.resize(SZ))
    for k, m in models.items():
        r = m.predict(D + '/images/' + n, conf=0.25, imgsz=640, verbose=False)[0]
        preds = [(int(b.cls), *[float(t) for t in b.xyxy[0]]) for b in r.boxes]
        t = im.copy()
        d = ImageDraw.Draw(t)
        used, miss, fp = set(), 0, 0
        for c, x0, y0, x1, y1 in preds:
            hit = [j for j, gg in enumerate(gts) if gg[0] == c and j not in used and iou((x0, y0, x1, y1), gg[1:]) >= 0.5]
            if hit:
                used.add(hit[0])
                d.rectangle([x0, y0, x1, y1], outline=(0, 220, 0), width=3)
            else:
                fp += 1
                dashed(d, (x0, y0, x1, y1), (255, 215, 0), 4)
        for j, gg in enumerate(gts):
            if j not in used:
                miss += 1
                dashed(d, gg[1:], (255, 0, 0), 4)
        info['stats'].setdefault(k, []).append({'image': n, 'gt': len(gts), 'pred': len(preds), 'missed': miss, 'false_positive': fp})
        tiles[k].append(t.resize(SZ))
rows = ['GT', *W]
canvas = Image.new('RGB', (SZ[0] * 6 + 5 * 6, SZ[1] * len(rows) + (len(rows) - 1) * 6), 'white')
for ri, rk in enumerate(rows):
    for ci in range(6):
        canvas.paste(tiles[rk][ci], (ci * (SZ[0] + 6), ri * (SZ[1] + 6)))
canvas.save(out + '/trashcan_qualitative.png')
json.dump(info, open(out + '/trashcan_qualitative.json', 'w'), indent=1)
print(json.dumps(info['stats']))
