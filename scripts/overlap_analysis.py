"""Near-duplicate / source-clip overlap analysis for a YOLO-format dataset.

usage: python3 overlap_analysis.py DATASET_ROOT [--ref OTHER_ROOT]
DATASET_ROOT contains train/images, val/images, test/images.
File names are expected as <clip>_frame<NNN>.jpg (clip = source video id).
With --ref, every test image of DATASET_ROOT is also compared with all images
of OTHER_ROOT (e.g. TrashCan) to check cross-dataset overlap.
"""
import os, json, hashlib, argparse
import numpy as np
from PIL import Image


def thumb(p):
    a = np.asarray(Image.open(p).convert('L').resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def dhash(p):
    a = np.asarray(Image.open(p).convert('L').resize((9, 8), Image.BILINEAR), np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def listing(root, split):
    d = os.path.join(root, split, 'images')
    return [os.path.join(d, f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []


def feats(paths):
    return np.stack([thumb(p) for p in paths]), np.stack([dhash(p) for p in paths])


def nearest(Fa, Ha, Fb, Hb, same=False):
    c = Fa @ Fb.T / Fa.shape[1]
    d = (Ha[:, None, :] != Hb[None, :, :]).sum(2)
    if same:
        np.fill_diagonal(c, -1)
        np.fill_diagonal(d, 99)
    return c.max(1), d.min(1)


def clip(p):
    return os.path.basename(p).rsplit('_frame', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--ref')
    a = ap.parse_args()
    P = {s: listing(a.root, s) for s in ('train', 'val', 'test')}
    F, H = {}, {}
    for s in P:
        F[s], H[s] = feats(P[s])
    out = {'sizes': {s: len(P[s]) for s in P}}
    out['clips'] = {s: len({clip(p) for p in P[s]}) for s in P}
    tr = {clip(p) for p in P['train']}
    out['test_images_sharing_clip_with_train'] = sum(clip(p) in tr for p in P['test'])
    md5 = {s: {hashlib.md5(open(p, 'rb').read()).hexdigest() for p in P[s]} for s in P}
    out['exact_duplicates_test_in_train'] = len(md5['test'] & md5['train'])
    for t, r in (('val', 'train'), ('test', 'train'), ('test', 'val')):
        c, d = nearest(F[t], H[t], F[r], H[r])
        out[f'{t}->{r}'] = {'n': len(c), 'corr>=0.95': int((c >= .95).sum()), 'corr>=0.90': int((c >= .90).sum()),
                            'dhash<=5': int((d <= 5).sum()), 'dhash<=10': int((d <= 10).sum())}
    if a.ref:
        R = [p for s in ('train', 'val', 'test') for p in listing(a.ref, s)] or \
            [os.path.join(a.ref, f) for f in sorted(os.listdir(a.ref)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        Fr, Hr = feats(R)
        c, d = nearest(F['test'], H['test'], Fr, Hr)
        out['test->ref'] = {'n': len(c), 'ref_images': len(R), 'corr>=0.95': int((c >= .95).sum()),
                            'dhash<=5': int((d <= 5).sum())}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
