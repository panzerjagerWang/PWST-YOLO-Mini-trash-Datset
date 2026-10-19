"""Latency and throughput of the network forward pass (image decoding and NMS excluded).

Protocol: batch size 1, 640x640 input, checkpoints with three classes, Conv+BN fused as in Ultralytics inference,
FP32 and FP16 (model.half()). Each model/precision is warmed up with 100 iterations and then timed in
BLOCKS blocks of ITERS CUDA-synchronized iterations; the configurations are interleaved (round 1: every model and
precision once, round 2: ... ), so that load on the shared machine affects all models alike, and the reported value is the
fastest block (the least disturbed one). The median over blocks is stored as well.
Two timings are taken: the eager PyTorch forward (fp32_ms, fp16_ms; includes the Python/kernel-launch overhead) and the
replay of a CUDA graph of the same forward pass (fp32_graph_ms, fp16_graph_ms; GPU execution time only).
Writes a JSON file {label: {params_M, fp32_ms, fp32_fps, fp16_ms, fp16_fps, fp32_graph_ms, fp16_graph_ms, *_med_ms, gpu}}.

usage (on an otherwise idle GPU; CODE_DIR = directory that contains ultralytics_repo, default below):
    python bench_latency.py OUT.json label=weights.pt [label=weights.pt ...]
"""
import json
import os
import statistics
import sys
import time

import torch

CODE = os.environ.get('CODE_DIR', '/media/w/DATA1/Paper/JOE_paper/Code')
sys.path[:0] = [CODE, CODE + '/ultralytics_repo']      # repository that contains PWAM / SRAW-TM
from ultralytics import YOLO  # noqa: E402

out_path, items = sys.argv[1], sys.argv[2:]
WARM, BLOCKS, ITERS = 100, int(os.environ.get('BENCH_BLOCKS', 25)), 100
DEVICE = os.environ.get('BENCH_DEVICE', 'cuda')
gpu = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'
torch.set_num_threads(1)


def sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


def block(fn, iters=ITERS):
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t) / iters


# build every configuration once
cfgs = []
params = {}
for it in items:
    label, w = it.split('=', 1)
    base = YOLO(w).model.float().eval().fuse(verbose=False)
    params[label] = round(sum(p.numel() for p in base.parameters()) / 1e6, 3)
    for tag, dt in (('fp32', torch.float32), ('fp16', torch.float16)):
        if DEVICE != 'cuda' and tag == 'fp16':
            continue
        m = YOLO(w).model.float().eval().fuse(verbose=False).to(dtype=dt).to(DEVICE)
        for p in m.parameters():
            p.requires_grad_(False)
        x = torch.rand(1, 3, 640, 640, device=DEVICE, dtype=dt)
        with torch.no_grad():
            for _ in range(WARM if DEVICE == 'cuda' else 2):
                m(x)
            sync()
        graph = None
        if DEVICE == 'cuda':
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s), torch.no_grad():
                    for _ in range(5):
                        m(x)
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    m(x)
                for _ in range(20):
                    graph.replay()
                sync()
            except Exception as e:            # a module that cannot be captured: report the eager timing only
                print('CUDA graph not available for', label, tag, repr(e)[:100], flush=True)
                graph = None
        cfgs.append(dict(label=label, tag=tag, m=m, x=x, graph=graph, eager=[], replay=[]))

rounds = BLOCKS if DEVICE == 'cuda' else 1
for r in range(rounds):
    for c in cfgs:
        with torch.no_grad():
            c['eager'].append(block(lambda: c['m'](c['x']), ITERS if DEVICE == 'cuda' else 3))
            if c['graph'] is not None:
                c['replay'].append(block(c['graph'].replay, ITERS * 2))

res = {}
for c in cfgs:
    r = res.setdefault(c['label'], {'params_M': params[c['label']], 'gpu': gpu, 'blocks': len(c['eager']), 'iters': ITERS})
    best = min(c['eager'])
    r[c['tag'] + '_ms'] = round(best * 1e3, 2)
    r[c['tag'] + '_fps'] = round(1.0 / best, 1)
    r[c['tag'] + '_med_ms'] = round(statistics.median(c['eager']) * 1e3, 2)
    if c['replay']:
        r[c['tag'] + '_graph_ms'] = round(min(c['replay']) * 1e3, 2)
for k, v in res.items():
    print(k, v, flush=True)
json.dump(res, open(out_path, 'w'), indent=1)
