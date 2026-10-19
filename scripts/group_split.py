"""Source-clip-level re-split of a YOLO dataset (all frames of one clip -> one subset).

usage: python3 group_split.py DATASET_ROOT OUT_ROOT [--seed 0] [--link]
Pools train/val/test, groups by <clip> in <clip>_frame<NNN>.jpg, assigns whole clips to
train/val/test with ~60/20/20 image share, writes OUT_ROOT/{train,val,test}/{images,labels}.
"""
import os, random, shutil, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('out')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--link', action='store_true', help='symlink instead of copy')
a = ap.parse_args()

items = collections.defaultdict(list)
for s in ('train', 'val', 'test'):
    d = os.path.join(a.root, s, 'images')
    for f in sorted(os.listdir(d)):
        items[f.rsplit('_frame', 1)[0]].append((s, f))
clips = sorted(items)
random.Random(a.seed).shuffle(clips)
total = sum(len(v) for v in items.values())
target = {'train': .6 * total, 'val': .2 * total, 'test': .2 * total}
count = {k: 0 for k in target}
assign = {}
for c in clips:  # greedy fill, largest deficit first
    k = max(target, key=lambda k: target[k] - count[k])
    assign[c] = k
    count[k] += len(items[c])
for c, k in assign.items():
    for s, f in items[c]:
        for sub in ('images', 'labels'):
            name = f if sub == 'images' else os.path.splitext(f)[0] + '.txt'
            src = os.path.join(a.root, s, sub, name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(a.out, k, sub)
            os.makedirs(dst, exist_ok=True)
            (os.symlink if a.link else shutil.copy2)(os.path.abspath(src), os.path.join(dst, name))
print('images per split:', count, '| clips per split:', collections.Counter(assign.values()))
