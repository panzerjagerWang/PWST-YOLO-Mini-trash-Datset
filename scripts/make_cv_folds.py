"""Source-clip-grouped 4-fold cross-validation pool for module design (CPU only, symlinks).

Pool = images of the ORIGINAL train and validation splits whose source clip does not occur in the test split of the clip-level split
(revision_runs/clipsplit/test).  Hence neither the original test split (202 images, not in the pool) nor the clip-level test split
(its clips are excluded) is touched by any design decision.  Folds are made of whole clips (greedy balancing of image count and of
'bio' clips).  Output: revision_runs/cv/f{k}/{train,val,test}/{images,labels}; test is a link to val (the trainer needs a test key).
usage: python make_cv_folds.py [K=4] [SEED=0]"""
import os, sys, random, collections, shutil

ROOT = '/media/w/DATA1/Paper/JOE_paper'
SRC = ROOT + '/Code/dataset'
CLIP_TEST = ROOT + '/revision_runs/clipsplit/test/images'
OUT = ROOT + '/revision_runs/cv'
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def clip_of(f):
    return f.rsplit('_frame', 1)[0]


test_clips = {clip_of(f) for f in os.listdir(CLIP_TEST)}
items = collections.defaultdict(list)
for sp in ('train', 'val'):
    for f in sorted(os.listdir(f'{SRC}/{sp}/images')):
        if clip_of(f) in test_clips:
            continue
        items[clip_of(f)].append((sp, f))
clips = sorted(items)
random.Random(SEED).shuffle(clips)
clips.sort(key=lambda c: -len(items[c]))        # largest first, ties keep the shuffled order (stable sort)
load = [0] * K
bio = [0] * K
assign = {}
for c in clips:
    if c.startswith('bio'):
        k = min(range(K), key=lambda i: (bio[i], load[i]))
        bio[k] += len(items[c])
    else:
        k = min(range(K), key=lambda i: load[i])
    assign[c] = k
    load[k] += len(items[c])
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for k in range(K):
    for part in ('train', 'val'):
        for sub in ('images', 'labels'):
            os.makedirs(f'{OUT}/f{k}/{part}/{sub}', exist_ok=True)
    os.symlink(f'{OUT}/f{k}/val', f'{OUT}/f{k}/test')
for c, k in assign.items():
    for sp, f in items[c]:
        lab = os.path.splitext(f)[0] + '.txt'
        for j in range(K):
            part = 'val' if j == k else 'train'
            os.symlink(f'{SRC}/{sp}/images/{f}', f'{OUT}/f{j}/{part}/images/{f}')
            if os.path.exists(f'{SRC}/{sp}/labels/{lab}'):
                os.symlink(f'{SRC}/{sp}/labels/{lab}', f'{OUT}/f{j}/{part}/labels/{lab}')
n_pool = sum(len(v) for v in items.values())
print('pool: %d images from %d clips (excluded %d clip-level-test clips); fold images %s; bio images %s' % (n_pool, len(items), len(test_clips), load, bio))
