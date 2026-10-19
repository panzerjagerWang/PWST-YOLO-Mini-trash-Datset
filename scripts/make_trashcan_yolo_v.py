"""Video-level exclusion for the TrashCan pre-training set.
make_trashcan_yolo.py removed every TrashCan image whose thumbnail correlation with a mini-Trash validation or test image is >= 0.95 (trashcan_yolo/excluded.json).
TrashCan frames are named vid_<id>_frame<k>.jpg, and the frames of one video are strongly correlated, so here every frame of a video that contains at least one
such image is removed as well.  Output: revision_runs/trashcan_yolo_v/{train,val}/{images,labels}, trashcan_yolo_v.yaml, excluded_video.json.
usage: python make_trashcan_yolo_v.py"""
import json
import os
import re
import shutil

R = '/media/w/DATA1/Paper/JOE_paper/revision_runs'
SRC = R + '/trashcan_yolo'
OUT = R + '/trashcan_yolo_v'


def vid(f):
    return re.match(r'(?:train/|val/)?(vid_\d+)_', f).group(1)


ex = json.load(open(SRC + '/excluded.json'))['excluded']
bad = sorted({vid(k) for k in ex})
badset = set(bad)
stats = {'videos_with_near_duplicates': len(bad), 'frame_level_excluded': len(ex)}
dropped = []
for sp in ('train', 'val'):
    os.makedirs('%s/%s/images' % (OUT, sp), exist_ok=True)
    os.makedirs('%s/%s/labels' % (OUT, sp), exist_ok=True)
    kept = 0
    for f in sorted(os.listdir('%s/%s/images' % (SRC, sp))):
        if vid(f) in badset:
            dropped.append('%s/%s' % (sp, f))
            continue
        d = '%s/%s/images/%s' % (OUT, sp, f)
        if not os.path.lexists(d):
            os.symlink(os.readlink('%s/%s/images/%s' % (SRC, sp, f)), d)
        shutil.copy('%s/%s/labels/%s.txt' % (SRC, sp, os.path.splitext(f)[0]), '%s/%s/labels/%s.txt' % (OUT, sp, os.path.splitext(f)[0]))
        kept += 1
    stats['kept_' + sp] = kept
stats['additionally_dropped_kept_frames'] = len(dropped)
json.dump({'videos': bad, 'dropped_frames': dropped, 'stats': stats}, open(R + '/trashcan_yolo_v/excluded_video.json', 'w'), indent=1)
y = open(R + '/trashcan_yolo.yaml').read().replace('/trashcan_yolo\n', '/trashcan_yolo_v\n')
assert '/trashcan_yolo_v\n' in y
open(R + '/trashcan_yolo_v.yaml', 'w').write(y)
print(stats)
