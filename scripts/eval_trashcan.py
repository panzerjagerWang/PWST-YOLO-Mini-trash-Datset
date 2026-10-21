"""Evaluates the finished TrashCan runs (best.pt of each run directory) on the validation and test splits and writes summary_trashcan.jsonl.
usage on the slave: python eval_trashcan.py"""
import json
import os
import sys

R = '/media/w/DATA1/Paper/JOE_paper'
C = R + '/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

project = R + '/revision_runs'
data = project + '/trashcan_split.yaml'
out = project + '/summary_trashcan.jsonl'
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line)
        done.add((r['name'], r['seed']))
for name in ('tcdet_base', 'tcdet_pwst'):
    for seed in (0, 1, 2):
        if (name, seed) in done:
            continue
        d = '%s/%s_s%d' % (project, name, seed)
        rows = sum(1 for _ in open(d + '/results.csv')) - 1
        best = YOLO(d + '/weights/best.pt')
        res = {'name': name, 'seed': seed, 'epochs_run': rows, 'params_M': round(sum(p.numel() for p in best.model.parameters()) / 1e6, 3),
               'GFLOPs': round(get_flops(best.model, 640), 2)}
        for split in ('val', 'test'):
            r = best.val(data=data, split=split, imgsz=640, batch=8, verbose=False, plots=False)
            res[split] = {'P': round(float(r.box.mp) * 100, 2), 'R': round(float(r.box.mr) * 100, 2), 'mAP50': round(float(r.box.map50) * 100, 2),
                          'mAP50-95': round(float(r.box.map) * 100, 2)}
        open(out, 'a').write(json.dumps(res) + '\n')
        print(res, flush=True)
