"""Training and evaluation on TrashCan 1.0 with the protocol of the manuscript: 200 epochs, SGD, batch 8, initial learning rate 0.01 with cosine decay, 640 x 640,
COCO-pretrained YOLOv11n initialization, the augmentation of the paper recipe (mixup 0.5, shear 20, mosaic 1.0, flips 0.5), no detector-specific tuning.
usage: python run_trashcan.py NAME MODEL.yaml SEED EPOCHS [DATA.yaml]      environment: WORKERS, PROJECT, LIGHT=1 for the light augmentation recipe (mixup 0, shear 0)
Appends the validation and test metrics to <project>/summary_trashcan.jsonl."""
import json
import os
import sys
import time

R = '/media/w/DATA1/Paper/JOE_paper'
C = R + '/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402

name, model, seed, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
data = sys.argv[5] if len(sys.argv) > 5 else R + '/revision_runs/trashcan_split.yaml'
project = os.environ.get('PROJECT', R + '/revision_runs')
m = YOLO(model)
m.load(C + '/yolo11n.pt')
t0 = time.time()
m.train(data=data, epochs=epochs, patience=epochs, batch=8, imgsz=640, seed=seed, deterministic=False, optimizer='SGD', lr0=0.01, cos_lr=True,
        mixup=0.0 if os.environ.get('LIGHT') == '1' else 0.5, shear=0.0 if os.environ.get('LIGHT') == '1' else 20.0, flipud=0.5, fliplr=0.5, mosaic=1.0, project=project, name='%s_s%d' % (name, seed), exist_ok=True,
        workers=int(os.environ.get('WORKERS', 4)), device=0, plots=False)
best = YOLO(str(m.trainer.save_dir) + '/weights/best.pt')
from ultralytics.utils.torch_utils import get_flops  # noqa: E402
res = {'name': name, 'seed': seed, 'epochs_run': epochs, 'train_min': round((time.time() - t0) / 60, 1), 'params_M': round(sum(p.numel() for p in best.model.parameters()) / 1e6, 3), 'GFLOPs': round(get_flops(best.model, 640), 2)}
for split in ('val', 'test'):
    r = best.val(data=data, split=split, imgsz=640, batch=8, verbose=False, plots=False)
    res[split] = {'P': round(float(r.box.mp) * 100, 2), 'R': round(float(r.box.mr) * 100, 2), 'mAP50': round(float(r.box.map50) * 100, 2), 'mAP50-95': round(float(r.box.map) * 100, 2)}
open(project + '/summary_trashcan.jsonl', 'a').write(json.dumps(res) + '\n')
print(res)
