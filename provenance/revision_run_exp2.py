"""Training and evaluation on mini-Trash: YOLOv11n, PWST-YOLOv11 and the module variants under one protocol.
usage: python run_exp.py NAME MODEL(.yaml|.pt) SEED EPOCHS
Writes <OUT>/<NAME>_s<SEED>/ and a json line to <OUT>/summary.jsonl
"""
import sys, os, json, time
C = '/media/w/DATA1/Paper/JOE_paper/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO

name, model, seed, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
OUT = '/media/w/DATA1/Paper/JOE_paper/revision_runs'
os.makedirs(OUT, exist_ok=True)
ROOT = os.environ.get('DATA_ROOT', C + '/dataset')
TAG = os.environ.get('DATA_TAG', 'minitrash')
DATA = OUT + '/' + TAG + '.yaml'
open(DATA, 'w').write(f"path: {ROOT}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: debris\n  1: bio\n  2: machine\n")
m = YOLO(model if model.startswith('/') or model.endswith('.pt') else model)
if model.endswith('.yaml'):
    m.load(C + '/yolo11n.pt')          # same weight injection as main_pretrained_better.py
t0 = time.time()
m.train(data=DATA, epochs=epochs, patience=100, batch=16, imgsz=640, seed=seed, deterministic=False,
        project=OUT, name=f'{name}_s{seed}', exist_ok=True, workers=8, device=0, plots=False,
        flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.5, shear=20.0)       # paper's augmentation recipe
save = str(m.trainer.save_dir)
best = YOLO(save + '/weights/best.pt')
res = {'name': name, 'seed': seed, 'epochs_run': epochs, 'train_min': round((time.time() - t0) / 60, 1)}
for split in ('val', 'test'):
    r = best.val(data=DATA, split=split, imgsz=640, batch=16, verbose=False, plots=False)
    res[split] = {'P': round(float(r.box.mp) * 100, 2), 'R': round(float(r.box.mr) * 100, 2),
                  'mAP50': round(float(r.box.map50) * 100, 2), 'mAP50-95': round(float(r.box.map) * 100, 2)}
open(OUT + '/summary.jsonl', 'a').write(json.dumps(res) + '\n')
print(res)
