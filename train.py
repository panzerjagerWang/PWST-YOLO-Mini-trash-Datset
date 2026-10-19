"""Portable training + evaluation script: the training recipe of the paper (identical to the experiment script in provenance/revision_run_exp3.py,
without workstation paths).

    python train.py --cfg configs/x_gin5.yaml --data data.yaml --name x_gin5_lightaug --seed 90 --light-aug

Recipe: COCO-pretrained YOLOv11n initialization (yolo11n.pt; weights of layers whose shapes match are loaded, the detection head is re-initialized),
300 epochs, early-stopping patience 100, batch 16, 640 x 640, Ultralytics default optimizer selection, mosaic 1.0 (closed in the last 10 epochs, the default),
flipud/fliplr 0.5; the paper recipe uses mixup 0.5 and shear 20, --light-aug sets both to 0.
After training the best checkpoint (by validation fitness) is evaluated on the validation and the test split and the result is appended to <project>/summary.jsonl.
The Ultralytics installation must contain the modules (see README: patch ultralytics_patch/pwst_ultralytics.patch)."""
import argparse
import json
import os
import time

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True, help='model yaml (e.g. configs/x_gin5.yaml or yolo11n.yaml)')
ap.add_argument('--data', required=True, help='dataset yaml with train / val / test')
ap.add_argument('--name', default='run')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--light-aug', action='store_true', help='no mixup, no shear')
ap.add_argument('--weights', default='yolo11n.pt', help='initial weights (COCO yolo11n.pt)')
ap.add_argument('--project', default='runs')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--device', default='0')
a = ap.parse_args()

m = YOLO(a.cfg)
if a.cfg.endswith('.yaml'):
    m.load(a.weights)
mixup, shear = (0.0, 0.0) if a.light_aug else (0.5, 20.0)
t0 = time.time()
run = '%s_s%d' % (a.name, a.seed)
m.train(data=a.data, epochs=a.epochs, patience=100, batch=16, imgsz=640, seed=a.seed, deterministic=False, mixup=mixup, shear=shear,
        flipud=0.5, fliplr=0.5, mosaic=1.0, project=a.project, name=run, exist_ok=True, workers=a.workers, device=a.device, plots=False)
best = YOLO(os.path.join(str(m.trainer.save_dir), 'weights', 'best.pt'))
res = {'name': a.name, 'seed': a.seed, 'epochs_run': a.epochs, 'train_min': round((time.time() - t0) / 60, 1)}
for split in ('val', 'test'):
    r = best.val(data=a.data, split=split, imgsz=640, batch=16, verbose=False, plots=False, device=a.device)
    res[split] = {'P': round(float(r.box.mp) * 100, 2), 'R': round(float(r.box.mr) * 100, 2),
                  'mAP50': round(float(r.box.map50) * 100, 2), 'mAP50-95': round(float(r.box.map) * 100, 2)}
os.makedirs(a.project, exist_ok=True)
with open(os.path.join(a.project, 'summary.jsonl'), 'a') as f:
    f.write(json.dumps(res) + '\n')
print(res)
