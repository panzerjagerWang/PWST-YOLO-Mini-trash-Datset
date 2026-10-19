"""Robustness evaluation, extended: the 9 original corruptions + 12 held-out ones (never used for training augmentation).

usage: python eval_robustness2.py data.yaml out.json label=weights.pt [label=weights.pt ...]
in-family : blur_{1,2,3}, contrast_{.7,.5,.3}, haze_{.8,.6,.4}
held-out  : motion_{9,17,25} (line kernel, 20 deg), defocus_{2,4,6} (disk radius, px), jpeg_{30,15,8} (quality),
            redloss_{.8,.6,.4} (red channel attenuated: underwater absorption)
Test split only, no re-training; mAP50 in %.
"""
import os, sys, json, shutil, tempfile, yaml
import numpy as np
import cv2

CODE = os.environ.get('CODE_DIR')      # optional: directory of a patched Ultralytics checkout; by default the installed (patched) package is used
if CODE:
    sys.path[:0] = [CODE]
from ultralytics import YOLO


def blur(im, s): return cv2.GaussianBlur(im, (0, 0), s)


def contrast(im, a):
    m = im.mean()
    return np.clip(a * (im.astype(np.float32) - m) + m, 0, 255).astype(np.uint8)


def haze(im, t):
    A = np.array([0.8, 0.75, 0.55]) * 255
    return np.clip(im.astype(np.float32) * t + A * (1 - t), 0, 255).astype(np.uint8)


def motion(im, L, ang=20):
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2
    dx, dy = np.cos(np.deg2rad(ang)) * c, np.sin(np.deg2rad(ang)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return cv2.filter2D(im, -1, k / k.sum())


def defocus(im, r):
    n = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((xx ** 2 + yy ** 2) <= r * r + 0.5).astype(np.float32)
    return cv2.filter2D(im, -1, k / k.sum())


def jpeg(im, q):
    ok, enc = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, 1)


def redloss(im, g):
    o = im.astype(np.float32).copy()
    o[..., 2] *= g   # BGR: red
    return np.clip(o, 0, 255).astype(np.uint8)


CORR = {'clean': lambda im: im}
CORR.update({f'blur_{s}': (lambda im, s=s: blur(im, s)) for s in (1, 2, 3)})
CORR.update({f'contrast_{a}': (lambda im, a=a: contrast(im, a)) for a in (.7, .5, .3)})
CORR.update({f'haze_{t}': (lambda im, t=t: haze(im, t)) for t in (.8, .6, .4)})
CORR.update({f'motion_{L}': (lambda im, L=L: motion(im, L)) for L in (9, 17, 25)})
CORR.update({f'defocus_{r}': (lambda im, r=r: defocus(im, r)) for r in (2, 4, 6)})
CORR.update({f'jpeg_{q}': (lambda im, q=q: jpeg(im, q)) for q in (30, 15, 8)})
CORR.update({f'redloss_{g}': (lambda im, g=g: redloss(im, g)) for g in (.8, .6, .4)})

data_yaml, out_json = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(data_yaml))
root = data.get('path', os.path.dirname(data_yaml))
src_img = os.path.join(root, data['test'])
src_lab = src_img.replace('images', 'labels')
res = json.load(open(out_json)) if os.path.exists(out_json) else {}
imgs = {f: cv2.imread(os.path.join(src_img, f)) for f in sorted(os.listdir(src_img))}
for spec in sys.argv[3:]:
    label, w = spec.split('=', 1)
    if label in res and len(res[label]) == len(CORR):
        continue
    model = YOLO(w)
    res[label] = {}
    for name, fn in CORR.items():
        tmp = tempfile.mkdtemp()
        os.makedirs(f'{tmp}/test/images')
        shutil.copytree(src_lab, f'{tmp}/test/labels')
        for f, im in imgs.items():
            cv2.imwrite(f'{tmp}/test/images/{f}', fn(im))
        yaml.safe_dump({'path': tmp, 'train': 'test/images', 'val': 'test/images', 'names': data['names']},
                       open(f'{tmp}/d.yaml', 'w'))
        r = model.val(data=f'{tmp}/d.yaml', split='val', imgsz=640, batch=16, verbose=False, plots=False, workers=2)
        res[label][name] = round(float(r.box.map50) * 100, 2)
        shutil.rmtree(tmp)
    print(label, {k: v for k, v in res[label].items() if k in ('clean', 'blur_3', 'haze_0.4', 'motion_25', 'defocus_6')}, flush=True)
    json.dump(res, open(out_json, 'w'), indent=1)
