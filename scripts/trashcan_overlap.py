"""Cross-dataset overlap between mini-Trash (1006 images, YOLO folders) and TrashCan 1.0 (instance version, train + val = 7212 images).
Content-based (names are not comparable: vid_XXXXXX_frameNNNNNNN vs bio/obj_frameNNNNNNN):
  - correlation of 32x32 grey thumbnails (z-scored), best match per mini-Trash image
  - 64-bit difference hash, minimum Hamming distance
  - the same statistics for the mini-Trash test split alone, and, for calibration, the within-mini-Trash statistics (test -> train).
Writes trashcan_overlap.json and a montage of the best-matching pairs.   usage: python trashcan_overlap.py"""
import json
import os
import zipfile
import numpy as np
from PIL import Image

R = '/media/w/DATA1/Paper/JOE_paper'
TC = R + '/trashcan'
MT = R + '/Code/dataset'
IMG = TC + '/imgs'
os.makedirs(IMG, exist_ok=True)
if len(os.listdir(IMG)) < 7000:
    z = zipfile.ZipFile(TC + '/dataset.zip')
    for m in z.namelist():
        if m.startswith('dataset/instance_version/') and m.endswith('.jpg'):
            sp = m.split('/')[2]
            out = os.path.join(IMG, sp + '__' + os.path.basename(m))
            if not os.path.exists(out):
                open(out, 'wb').write(z.read(m))
tc = sorted(os.path.join(IMG, f) for f in os.listdir(IMG))
print('TrashCan images:', len(tc))


def listing(split):
    d = os.path.join(MT, split, 'images')
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


mt = {s: listing(s) for s in ('train', 'val', 'test')}
print({s: len(v) for s, v in mt.items()})


def feat(p):
    im = Image.open(p).convert('L')
    a = np.asarray(im.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    h = np.asarray(im.resize((9, 8), Image.BILINEAR), np.int16)
    return a, (h[:, 1:] > h[:, :-1]).ravel()


def feats(paths):
    f = [feat(p) for p in paths]
    return np.stack([a for a, _ in f]), np.stack([b for _, b in f])


Ft, Ht = feats(tc)
res = {'trashcan_images': len(tc)}
allp, alls = [], []
for s in ('train', 'val', 'test'):
    for p in mt[s]:
        allp.append(p)
        alls.append(s)
Fm, Hm = feats(allp)
C = Fm @ Ft.T / Fm.shape[1]
D = np.stack([(Hm[i][None, :] != Ht).sum(1) for i in range(len(Hm))])
cbest, ci = C.max(1), C.argmax(1)
dbest = D.min(1)
alls = np.array(alls)
for s in ('train', 'val', 'test', 'all'):
    m = np.ones(len(alls), bool) if s == 'all' else alls == s
    res['minitrash_%s' % s] = {'n': int(m.sum()), 'corr>=0.99': int((cbest[m] >= .99).sum()), 'corr>=0.95': int((cbest[m] >= .95).sum()),
                               'corr>=0.90': int((cbest[m] >= .90).sum()), 'corr>=0.80': int((cbest[m] >= .80).sum()),
                               'dhash<=5': int((dbest[m] <= 5).sum()), 'dhash<=10': int((dbest[m] <= 10).sum()),
                               'median_best_corr': float(np.median(cbest[m]))}
# calibration: mini-Trash test -> mini-Trash train (same statistics) and mini-Trash test -> random TrashCan subsample of the size of mini-Trash train
ti, tri = np.where(alls == 'test')[0], np.where(alls == 'train')[0]
Cin = Fm[ti] @ Fm[tri].T / Fm.shape[1]
res['calibration_test_to_train_minitrash'] = {'n': len(ti), 'corr>=0.95': int((Cin.max(1) >= .95).sum()), 'corr>=0.90': int((Cin.max(1) >= .90).sum()),
                                              'median_best_corr': float(np.median(Cin.max(1)))}
# the same for TrashCan val -> TrashCan train frames of the same clip would be an upper bound; not needed
vid = lambda p: os.path.basename(p).split('__')[1].rsplit('_frame', 1)[0]
res['trashcan_videos'] = len({vid(p) for p in tc})
order = np.argsort(-cbest)
pairs = [(allp[i], tc[ci[i]], float(cbest[i]), int(D[i, ci[i]]), alls[i]) for i in order[:40]]
res['top_pairs'] = [{'minitrash': os.path.basename(a), 'split': s, 'trashcan': os.path.basename(b), 'corr': round(c, 3), 'dhash': d} for a, b, c, d, s in pairs[:25]]
# histogram of best correlations
res['best_corr_hist'] = np.histogram(cbest, bins=[-1, .5, .6, .7, .8, .9, .95, .99, 1.01])[0].tolist()
json.dump(res, open(TC + '/trashcan_overlap.json', 'w'), indent=1)
print(json.dumps(res, indent=1)[:3000])

# montage of the 8 best pairs and 8 pairs around corr ~ 0.9 (visual check of what a given correlation means)
sel = list(order[:8]) + [order[int((cbest >= .9).sum()) - 1 - k * 3] for k in range(4)] + [order[int((cbest >= .8).sum()) - 1 - k * 5] for k in range(4)]
W, Hh = 240, 160
canvas = Image.new('RGB', (W * 4, Hh * 2 * 4 // 1), 'white')
for k, i in enumerate(sel[:16]):
    a = Image.open(allp[i]).convert('RGB').resize((W, Hh))
    b = Image.open(tc[ci[i]]).convert('RGB').resize((W, Hh))
    r, c = divmod(k, 4)
    canvas.paste(a, (c * W, r * 2 * Hh))
    canvas.paste(b, (c * W, r * 2 * Hh + Hh))
canvas.save(TC + '/trashcan_pairs.png')
print('montage written; corr of the 16 pairs:', [round(float(cbest[i]), 3) for i in sel[:16]])
