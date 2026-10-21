"""Parameters and GFLOPs (ultralytics convention: 2 x MACs, 640x640) of the trained checkpoints (nc=3)."""
import sys, os, json
C = '/media/w/DATA1/Paper/JOE_paper/Code'
sys.path[:0] = [C, C + '/ultralytics_repo']
os.environ.setdefault('PWAM_MODE', 'full')
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
R = '/media/w/DATA1/Paper/JOE_paper/revision_runs/'
CK = {'YOLOv11n': 'yolo11n_base_s0', 'PWST-YOLOv11': 'pwst_v7final_s0', 'pw_A': 'pw_A_sraw57_s0', 'pw_B': 'pw_B_pw01_sraw57_s0',
      'pw_C': 'pw_C_sraw357_s0', 'pw_D': 'pw_D_sra57_s0', 'pw_E': 'pw_E_sraw7_s0', 'pw_F': 'pw_F_pw357_s0', 'pw_G': 'pw_G_fix_s0',
      'pw_H': 'pw_H_fix_s0', 'pw_I': 'pw_I_fix_s0'}
out = {}
for k, d in CK.items():
    try:
        net = YOLO(R + d + '/weights/best.pt').model.float().cpu().eval()
        n = sum(p.numel() for p in net.parameters())
        f = get_flops(net, 640)
        out[k] = {'params_M': round(n / 1e6, 3), 'GFLOPs': round(f, 2), 'GMACs': round(f / 2, 2), 'nc': int(net.nc)}
    except Exception as e:
        out[k] = {'error': str(e)[:150]}
    print(k, out[k], flush=True)
json.dump(out, open(R + 'flops.json', 'w'), indent=1)
