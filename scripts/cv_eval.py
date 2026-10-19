"""Evaluate the CV runs of the module screening (revision_runs/cv_<variant>_<key>_s<fold>).
For every finished run: (a) clean held-out accuracy = mean of the last 5 epochs of results.csv (unfused EMA model, no checkpoint selection),
(b) mAP50 of last.pt on the held-out fold after seven synthetic corruptions (blur 3, defocus 4, motion 17, contrast .5, haze .4, JPEG 15, red loss .6).
Writes revision_runs/cv_results.json  {run: {clean: {...}, corrupt: {...}, fold: k}}.   usage: python cv_eval.py [--force]"""
import os, sys, re, csv, json, glob, shutil, tempfile
import numpy as np
import cv2, yaml
C = '/media/w/DATA1/Paper/JOE_paper/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
os.environ.setdefault('PWAM_MODE', 'full')
from ultralytics import YOLO

R = '/media/w/DATA1/Paper/JOE_paper/revision_runs'
OUT = R + '/cv_results.json'
res = json.load(open(OUT)) if os.path.exists(OUT) and '--force' not in sys.argv else {}


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
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((xx ** 2 + yy ** 2) <= r * r + 0.5).astype(np.float32)
    return cv2.filter2D(im, -1, k / k.sum())


def jpeg(im, q):
    ok, enc = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, 1)


def redloss(im, g):
    o = im.astype(np.float32).copy()
    o[..., 2] *= g
    return np.clip(o, 0, 255).astype(np.uint8)


CORR = {'blur_3': lambda im: blur(im, 3), 'defocus_4': lambda im: defocus(im, 4), 'motion_17': lambda im: motion(im, 17),
        'contrast_0.5': lambda im: contrast(im, 0.5), 'haze_0.4': lambda im: haze(im, 0.4), 'jpeg_15': lambda im: jpeg(im, 15),
        'redloss_0.6': lambda im: redloss(im, 0.6)}

runs = sorted(glob.glob(R + '/cv_*_s[0-9]*'))
cache = {}
for rd in runs:
    name = os.path.basename(rd)
    m = re.match(r'cv_([a-z]+)_(.+)_s(\d+)$', name)
    if not m or name in res:
        continue
    var, key, seed = m.group(1), m.group(2), int(m.group(3))
    fold = seed % 4          # seeds 0..3 = folds, 4..7 / 8..11 = replicate trainings of the same folds
    rcsv = rd + '/results.csv'
    last = rd + '/weights/last.pt'
    if not os.path.exists(rcsv) or not os.path.exists(last):
        continue
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(rcsv))]
    if len(rows) < 100:
        continue
    tail = rows[-5:]
    clean = {k: float(np.mean([float(r[c]) for r in tail])) * 100 for k, c in
             (('mAP50', 'metrics/mAP50(B)'), ('mAP50-95', 'metrics/mAP50-95(B)'), ('P', 'metrics/precision(B)'), ('R', 'metrics/recall(B)'))}
    # held-out fold images (same for every variant, the fold directory of the raw pool)
    fold_val = R + '/cv/f%d/val' % fold
    if fold not in cache:
        d = fold_val + '/images'
        cache[fold] = {f: cv2.imread(os.path.join(d, f)) for f in sorted(os.listdir(d))}
    imgs = cache[fold]
    model = YOLO(last)
    corrupt = {}
    for cname, fn in CORR.items():
        tmp = tempfile.mkdtemp()
        os.makedirs(tmp + '/val/images')
        shutil.copytree(os.path.realpath(fold_val + '/labels'), tmp + '/val/labels')
        for f, im in imgs.items():
            cv2.imwrite(tmp + '/val/images/' + f, fn(im))
        yaml.safe_dump({'path': tmp, 'train': 'val/images', 'val': 'val/images', 'names': {0: 'debris', 1: 'bio', 2: 'machine'}}, open(tmp + '/d.yaml', 'w'))
        r = model.val(data=tmp + '/d.yaml', split='val', imgsz=640, batch=16, verbose=False, plots=False, workers=2)
        corrupt[cname] = round(float(r.box.map50) * 100, 2)
        shutil.rmtree(tmp)
    res[name] = {'variant': var, 'key': key, 'fold': fold, 'seed': seed, 'clean': clean, 'corrupt': corrupt}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(name, '%.1f/%.1f' % (clean['mAP50'], clean['mAP50-95']), corrupt, flush=True)
print('CV_EVAL_DONE', len(res), flush=True)
