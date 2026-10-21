"""Heat maps of the stride-16 feature map that enters the detection head, and their alignment with the ground truth, on the mini-Trash test split.
usage: python heatmap_align.py OUT_DIR LABEL=weights.pt [LABEL=weights.pt ...]      (environment: TEST_DIR, OVERLAY=label1,label2,... for the figure)
For every model and test image: channel mean of the stride-16 input of the detection head, letterbox padding removed, resized to the image,
min-max normalized to H in [0, 1]. M is the union of the ground-truth boxes.
IoU-HM = IoU of {H >= 0.5} and M; AOR = share of sum(H) inside M in %; Pointing = share of images whose maximum of H lies inside M in %.
The figure uses four test images equally spaced among the test images with at least one object; the first column shows the image with the ground truth."""
import json
import os
import sys

import cv2
import numpy as np
import torch

C = os.environ.get('CODE_DIR', '/media/w/DATA1/Paper/JOE_paper/Code')      # directory that contains ultralytics_repo and dataset/
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402

torch.set_num_threads(int(os.environ.get('THREADS', '12')))
D = os.environ.get('TEST_DIR', C + '/dataset/test')
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
names = sorted(os.listdir(D + '/images'))


def boxes(n, w, h):
    p = D + '/labels/' + os.path.splitext(n)[0] + '.txt'
    b = []
    if os.path.exists(p):
        for ln in open(p):
            v = ln.split()
            if len(v) >= 5:
                x, y, bw, bh = map(float, v[1:5])
                b.append((max(0, int((x - bw / 2) * w)), max(0, int((y - bh / 2) * h)), min(w, int(np.ceil((x + bw / 2) * w))), min(h, int(np.ceil((y + bh / 2) * h)))))
    return b


imgs = []
for n in names:
    im = cv2.imread(D + '/images/' + n)
    h, w = im.shape[:2]
    b = boxes(n, w, h)
    if b:
        M = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in b:
            M[y0:y1, x0:x1] = True
        imgs.append((n, im, b, M))
step = len(imgs) // 4
pick = [imgs[i * step + step // 2][0] for i in range(4)]
lb = LetterBox(new_shape=(640, 640), auto=False, stride=32)
res = json.load(open(out + '/align.json')) if os.path.exists(out + '/align.json') else {}
overlay = [x for x in os.environ.get('OVERLAY', '').split(',') if x]
tiles = {}
for spec in sys.argv[2:]:
    label, wpath = spec.split('=', 1)
    if label in res and label not in overlay:
        continue
    model = YOLO(wpath).model.float().eval()
    det = model.model[-1]
    k16 = [int(s) for s in det.stride.tolist()].index(16)
    feats = {}
    hook = det.register_forward_pre_hook(lambda mod, inp: feats.__setitem__('x', inp[0][k16].detach()))
    iou_l, aor_l, pt_l = [], [], []
    for n, im, b, M in imgs:
        h, w = im.shape[:2]
        x = lb(image=im)
        t = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1])).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            model(t)
        f = feats['x'][0].float().mean(0).numpy()
        f = cv2.resize(f, (640, 640), interpolation=cv2.INTER_LINEAR)
        r = min(640 / h, 640 / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        top, left = int(round((640 - nh) / 2 - 0.1)), int(round((640 - nw) / 2 - 0.1))
        f = cv2.resize(f[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_LINEAR)
        H = (f - f.min()) / (f.max() - f.min() + 1e-9)
        B = H >= 0.5
        iou_l.append(float((B & M).sum() / max((B | M).sum(), 1)))
        aor_l.append(float(100.0 * (H * M).sum() / max(H.sum(), 1e-9)))
        yy, xx = np.unravel_index(int(H.argmax()), H.shape)
        pt_l.append(100.0 * float(M[yy, xx]))
        if label in overlay and n in pick:
            col = cv2.applyColorMap((H * 255).astype(np.uint8), cv2.COLORMAP_JET)
            tiles.setdefault(label, {})[n] = cv2.addWeighted(im, 0.5, col, 0.5, 0)
    hook.remove()
    res[label] = {'IoU_HM': float(np.mean(iou_l)), 'AOR': float(np.mean(aor_l)), 'Pointing': float(np.mean(pt_l)), 'n_images': len(iou_l), 'weights': wpath}
    json.dump(res, open(out + '/align.json', 'w'), indent=1)
    print(label, {k: round(v, 3) for k, v in res[label].items() if isinstance(v, float)}, flush=True)

if overlay and all(lab in tiles for lab in overlay):
    TW, TH = 360, 240
    rows = []
    for n in pick:
        im = next(x[1].copy() for x in imgs if x[0] == n)
        for x0, y0, x1, y1 in next(x[2] for x in imgs if x[0] == n):
            cv2.rectangle(im, (x0, y0), (x1, y1), (255, 255, 0), 3)
        row = [cv2.resize(im, (TW, TH))] + [cv2.resize(tiles[lab][n], (TW, TH)) for lab in overlay]
        rows.append(np.concatenate([np.pad(t_, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t_ in row], axis=1))
    cv2.imwrite(out + '/heatmaps.png', np.concatenate(rows, axis=0))
    json.dump({'images': pick, 'columns': ['image with ground truth'] + overlay}, open(out + '/heatmaps.json', 'w'), indent=1)
    print('figure written', out + '/heatmaps.png')
