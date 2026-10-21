"""Training and evaluation on mini-Trash: YOLOv11n, PWST-YOLOv11 and the module variants under one protocol.
usage: python run_exp.py NAME MODEL(.yaml|.pt) SEED EPOCHS
Writes <OUT>/<NAME>_s<SEED>/ and a json line to <OUT>/summary.jsonl
"""
import sys, os, json, time
C = '/media/w/DATA1/Paper/JOE_paper/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO
if os.environ.get('DROP_LAST'):
    # PWAM_Lite has a BatchNorm on a 1x1 pooled map; the 1809-image degradation-aware train set leaves a final batch of 1 image (1809 % 16 == 1) -> ValueError.
    # Drop the incomplete last batch (one image per epoch, a different one each epoch because of shuffling).
    import ultralytics.models.yolo.detect.train as _dt
    _bd = _dt.build_dataloader
    _dt.build_dataloader = lambda dataset, batch, workers, shuffle=True, rank=-1, drop_last=False, pin_memory=True: _bd(
        dataset, batch, workers, shuffle, rank, drop_last=shuffle, pin_memory=pin_memory)

name, model, seed, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
OUT = '/media/w/DATA1/Paper/JOE_paper/revision_runs'
os.makedirs(OUT, exist_ok=True)
ROOT = os.environ.get('DATA_ROOT', C + '/dataset')
TAG = os.environ.get('DATA_TAG', 'minitrash')
DATA = OUT + '/' + TAG + '.yaml'
open(DATA, 'w').write(f"path: {ROOT}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: debris\n  1: bio\n  2: machine\n")
m = YOLO(model if model.startswith('/') or model.endswith('.pt') else model)
if model.endswith('.yaml'):
    m.load(os.environ.get('INIT_WEIGHTS', C + '/yolo11n.pt'))          # same weight injection as main_pretrained_better.py; INIT_WEIGHTS = TrashCan pre-training
t0 = time.time()
HP = dict(imgsz=int(os.environ.get('IMGSZ', 640)), mixup=float(os.environ.get('MIXUP', 0.5)), shear=float(os.environ.get('SHEAR', 20.0)))
if os.environ.get('OPT'): HP.update(optimizer=os.environ['OPT'], lr0=float(os.environ['LR0']))
for _k in ('hsv_h', 'hsv_s', 'hsv_v', 'scale', 'translate', 'degrees', 'flipud', 'fliplr', 'mosaic', 'copy_paste', 'erasing', 'close_mosaic', 'freeze'):
    if os.environ.get(_k.upper()):                      # optional recipe overrides (screening experiments); defaults unchanged
        HP[_k] = int(os.environ[_k.upper()]) if _k in ('close_mosaic', 'freeze') else float(os.environ[_k.upper()])
m.train(data=DATA, epochs=epochs, patience=100, batch=16, seed=seed, deterministic=False, **HP,
        project=OUT, name=f'{name}_s{seed}', exist_ok=True, workers=int(os.environ.get('WORKERS', 8)), device=0, plots=False,
        **{k: v for k, v in dict(flipud=0.5, fliplr=0.5, mosaic=1.0).items() if k not in HP})       # paper's augmentation recipe
save = str(m.trainer.save_dir)
best = YOLO(save + '/weights/best.pt')
res = {'name': name, 'seed': seed, 'epochs_run': epochs, 'train_min': round((time.time() - t0) / 60, 1)}
for split in ('val', 'test'):
    r = best.val(data=DATA, split=split, imgsz=HP['imgsz'], batch=16, verbose=False, plots=False)
    res[split] = {'P': round(float(r.box.mp) * 100, 2), 'R': round(float(r.box.mr) * 100, 2),
                  'mAP50': round(float(r.box.map50) * 100, 2), 'mAP50-95': round(float(r.box.map) * 100, 2)}
open(OUT + '/summary.jsonl', 'a').write(json.dumps(res) + '\n')
print(res)
