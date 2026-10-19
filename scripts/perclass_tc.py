"""Per-class test AP of the models of the TrashCan-initialisation experiment and of their COCO-initialised references (descriptive; run on the workstation).
Writes revision_runs/perclass_tc.json: {group: {seed: {'ap50': [3], 'ap': [3]}}} for the classes debris, bio, machine.   usage: python perclass_tc.py"""
import json
import os
import sys

R = '/media/w/DATA1/Paper/JOE_paper'
C = R + '/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
from ultralytics import YOLO  # noqa: E402

OUT = R + '/revision_runs'
DATA = OUT + '/minitrash.yaml'
GROUPS = {
    'paper_coco': ('yolo11n_base', range(0, 8)),
    'la_coco': ('base_lightaug', range(0, 10)),
    'S_coco_conf': ('x_A_amp_lightaug', range(10, 20)),
    'B_tc': ('tc_yolo11n_base', range(20, 28)),
    'L_tc': ('tc_base_lightaug', range(20, 28)),
    'S_tc': ('tc_x_A_amp_lightaug', range(20, 28)),
}
fn = OUT + '/perclass_tc.json'
res = json.load(open(fn)) if os.path.exists(fn) else {}
for g, (name, seeds) in GROUPS.items():
    for s in seeds:
        w = '%s/%s_s%d/weights/best.pt' % (OUT, name, s)
        if str(s) in res.get(g, {}) or not os.path.exists(w):
            continue
        r = YOLO(w).val(data=DATA, split='test', imgsz=640, batch=16, verbose=False, plots=False)
        res.setdefault(g, {})[str(s)] = {'ap50': [round(float(x) * 100, 2) for x in r.box.ap50], 'ap': [round(float(x) * 100, 2) for x in r.box.ap],
                                         'map50': round(float(r.box.map50) * 100, 2)}
        json.dump(res, open(fn, 'w'), indent=1)
        print(g, s, res[g][str(s)], flush=True)
print('done')
