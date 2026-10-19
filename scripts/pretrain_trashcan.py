"""Stage 1 of the TrashCan initialisation: YOLOv11n (COCO weights) trained on TrashCan 1.0 (22 classes, leakage-controlled, see make_trashcan_yolo.py and
make_trashcan_yolo_v.py).  The best checkpoint (by TrashCan validation fitness) is the initial weight of the mini-Trash runs (INIT_WEIGHTS of
revision_run_exp3.py; the detection head is re-initialised because the class count differs).
usage: EP=100 python pretrain_trashcan.py [DATA_NAME [RUN_NAME]]      (defaults: trashcan_yolo, trashcan_pretrain)"""
import os
import sys

R = '/media/w/DATA1/Paper/JOE_paper'
C = R + '/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402

OUT = R + '/revision_runs'
DATA = sys.argv[1] if len(sys.argv) > 1 else 'trashcan_yolo'
NAME = sys.argv[2] if len(sys.argv) > 2 else 'trashcan_pretrain'
m = YOLO('yolo11n.yaml')
m.load(C + '/yolo11n.pt')
m.train(data='%s/%s.yaml' % (OUT, DATA), epochs=int(os.environ.get('EP', 100)), patience=50, batch=32, imgsz=640, seed=0, deterministic=False,
        mixup=0.0, shear=0.0, flipud=0.5, fliplr=0.5, mosaic=1.0, project=OUT, name=NAME, exist_ok=True,
        workers=int(os.environ.get('WORKERS', 8)), device=0, plots=False)
print('done', m.trainer.save_dir)
