"""print up to N 'label=weights' specs of finished runs that are not yet complete in robust2.json (used by watch_robust.sh).
extra weights (e.g. trained on another machine) can be listed in <R>/robust_extra.txt as label=path lines."""
import json, os, sys
R, N = sys.argv[1], int(sys.argv[2])
MAP = {"x_L7gin5": "x_L7gin5", "x_Agin5": "x_Agin5", "x_Lgin5": "x_Lgin5", "x_L": "x_L", 'yolo11n_base': 'base', 'pw_F_pw357': 'F', 'pw_A_sraw57': 'A', 'pw_B_pw01_sraw57': 'B', 'pw_C_sraw357': 'C', 'pw_D_sra57': 'D',
       'pw_E_sraw7': 'E', 'pw_G_fix': 'G', 'pw_H_fix': 'H', 'pw_I_fix': 'I', 'pwst_v7final': 'P'}
LA = {"x_gin5": "x_gin5", "x_L7gin5": "x_L7gin5", "x_Agin5": "x_Agin5", "x_Lgin5": "x_Lgin5", "x_L": "x_L", 'x_amp': 'x_amp', 'x_A_amp': 'x_A_amp', 'base': 'base', 'pwF': 'F', 'pwA': 'A', 'pwB': 'B', 'pwC': 'C', 'pwD': 'D', 'pwE': 'E', 'pwG': 'G', 'pwH': 'H', 'pwI': 'I', 'pwst': 'P'}
rj = os.path.join(R, 'robust2.json')
done = json.load(open(rj)) if os.path.exists(rj) else {}
specs = []
seen = set()
for l in open(os.path.join(R, 'summary.jsonl')):
    d = json.loads(l)
    n = d['name']
    if n.startswith('clip_'):      # trained on the clip-level split: the frame-level test set overlaps its training clips
        continue
    if n in MAP:
        lab = MAP[n]
    elif n.endswith(('_uw', '_uwl', '_nomix', '_noshear')):
        lab = n
    elif n.endswith('_lightaug') and n[:-9] in LA:
        lab = LA[n[:-9]] + '_la'
    elif n.startswith('tc_') and n[3:] in MAP:      # TrashCan-initialised, paper augmentation
        lab = 'tc_' + MAP[n[3:]]
    elif n.startswith('tc_') and n.endswith('_lightaug') and n[3:-9] in LA:      # TrashCan-initialised, light augmentation
        lab = 'tc_' + LA[n[3:-9]] + '_la'
    else:
        continue
    lab = '%s_s%d' % (lab, d['seed'])
    w = os.path.join(R, '%s_s%d' % (n, d['seed']), 'weights', 'best.pt')
    if lab in seen or not os.path.exists(w):
        continue
    seen.add(lab)
    if lab not in done or len(done[lab]) < 22:
        specs.append('%s=%s' % (lab, w))
ex = os.path.join(R, 'robust_extra.txt')
if os.path.exists(ex):
    for l in open(ex):
        l = l.strip()
        if '=' in l:
            lab, w = l.split('=', 1)
            if os.path.exists(w) and (lab not in done or len(done[lab]) < 22) and lab not in seen:
                specs.append(l)
def prio(spec):
    lab = spec.split('=', 1)[0]
    if '_uw' in lab:      # degradation-aware models first
        return 0
    if lab.endswith(('_s0', '_s1', '_s2')) and lab.split('_')[0] in ('base', 'F', 'A', 'B', 'P'):
        return 1
    return 2


specs.sort(key=prio)
print(' '.join(specs[:N]))
