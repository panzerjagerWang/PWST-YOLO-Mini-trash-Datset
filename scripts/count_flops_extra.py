"""Parameters / GFLOPs (nc=3, 640x640) of the configurations added in the last revision round, from their YAMLs."""
import sys, os, json
C = '/media/w/DATA1/Paper/JOE_paper/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
os.environ.setdefault('PWAM_MODE', 'full')
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

R = '/media/w/DATA1/Paper/JOE_paper/revision_runs/'
CFG = {'pw_W': R + 'cfg/pw_W_wpatm357.yaml', 'PWST-YOLOv11 (full PWAM)': C + '/dataset/yolov11n_pwstv8.yaml',
       'pw_F': R + 'cfg/pw_F_pw357.yaml', 'YOLOv11n': 'yolo11n.yaml'}
out = json.load(open(R + 'flops.json'))
for k, y in CFG.items():
    try:
        net = DetectionModel(y, nc=3, verbose=False).float().cpu().eval()
        n = sum(p.numel() for p in net.parameters())
        f = get_flops(net, 640)
        rec = {'params_M': round(n / 1e6, 3), 'GFLOPs': round(f, 2), 'GMACs': round(f / 2, 2), 'nc': 3}
        print(k, rec, 'stored:', out.get(k), flush=True)
        if k not in out:
            out[k] = rec
    except Exception as e:
        print(k, 'ERROR', str(e)[:200], flush=True)
json.dump(out, open(R + 'flops.json', 'w'), indent=1)
