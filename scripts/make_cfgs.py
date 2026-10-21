"""Write the model YAMLs of the v3 modules (layer positions are unchanged, only the module class of selected layers is replaced, so the
COCO weights load exactly as for YOLOv11n).  usage: python make_cfgs.py   -> revision_runs/cfg/x_*.yaml"""
import copy, os, yaml

CFG = '/media/w/DATA1/Paper/JOE_paper/revision_runs/cfg'
base = yaml.safe_load(open(f'{CFG}/yolo11n_nc3.yaml'))
layers = base['backbone'] + base['head']
VARIANTS = {
    'x_amp':    {0: 'ConvAmp0'},                                  # phase-preserving Fourier amplitude mixing of the input (train only)
    'x_ms':     {2: 'C3k2MS', 4: 'C3k2MS'},                       # MixStyle after the first two C3k2 stages (train only)
    'x_ampms':  {0: 'ConvAmp0', 2: 'C3k2MS', 4: 'C3k2MS'},
    'x_gin':    {0: 'ConvGIN0'},                                  # gated per-image channel standardisation of the input, a0 = 0
    'x_gin5':   {0: 'ConvGIN5'},                                  # same, a0 = 0.5
    'x_ginf':   {2: 'C3k2FIN', 4: 'C3k2FIN'},                     # gated instance normalisation of early features (IBN-like)
    'x_ampgin': {0: 'ConvAmpGIN0'},
    'x_lk':     {6: 'C3k2LK', 8: 'C3k2LK'},                       # zero-gated 7x7 depth-wise branch (capacity control)
    'x_ampp1':  {0: 'ConvAmpP1'},                                 # AmpMix on every training sample
    'x_amplo':  {0: 'ConvAmpLo'},                                 # AmpMix, low-frequency band only
    'x_amphi':  {0: 'ConvAmpHi'},                                 # AmpMix, high-frequency band only
    'x_ampf':   {2: 'C3k2AmpF', 4: 'C3k2AmpF'},                   # feature-level amplitude mixing after the first two C3k2 stages
    'x_ush':    {0: 'ConvUSH0'},                                  # learned multi-scale local-contrast (unsharp) gain of the input
    'x_lcn':    {0: 'ConvLCN0'},                                  # local contrast normalisation of the input (blend a0 = 0.5)
    'x_sg':     {2: 'C3k2SG', 4: 'C3k2SG'},                       # phase-preserving spectral amplitude gain on early features
}
for name, sub in VARIANTS.items():
    d = copy.deepcopy(base)
    L = d['backbone'] + d['head']
    for i, mod in sub.items():
        assert L[i][2] in ('Conv', 'C3k2'), (i, L[i])
        L[i][2] = mod
    yaml.safe_dump(d, open(f'{CFG}/{name}.yaml', 'w'), sort_keys=False)
    print('wrote', name, sub)

# combinations with the SRAW-TM variants (pw_A / pw_B yaml of the ablation): AmpMix on the input of the first convolution
for name, srcname in (('x_A_amp', 'pw_A_sraw57_nc3'), ('x_B_amp', 'pw_B_pw01_sraw57_nc3')):
    p_ = f'{CFG}/{srcname}.yaml'
    if not os.path.exists(p_):
        continue
    d = yaml.safe_load(open(p_))
    L = d['backbone'] + d['head']
    assert L[0][2] in ('Conv', 'ConvPWAM'), L[0]
    if L[0][2] == 'Conv':
        L[0][2] = 'ConvAmp0'
    else:
        print('first layer is', L[0][2], 'not combined for', name)
        continue
    yaml.safe_dump(d, open(f'{CFG}/{name}.yaml', 'w'), sort_keys=False)
    print('wrote', name, 'from', srcname)
