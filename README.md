# PWST-YOLO and Gated Input Standardization (GIN) for Underwater Debris Detection

Code, weights and reproduction material for the manuscript

> **Enhancing Underwater Object Detection via Phase-Aware Wave-like Aggregation and Space-Reduction Attention**
> Jiaqing Wang, Baichuan Zeng, Hexiong Zhou\*, Zheng Zeng\* (Shanghai Jiao Tong University; The Chinese University of Hong Kong)
> submitted to *IEEE Journal of Oceanic Engineering* (manuscript 26-0110)

**Project page:** https://panzerjagerwang.github.io/PWST-YOLO-Mini-trash-Datset/ &nbsp;·&nbsp; **Weights and mini-Trash dataset:** [Releases](../../releases) &nbsp;·&nbsp; **Supplementary material:** [`docs/supplementary.pdf`](docs/supplementary.pdf)

## What this repository shows

Turbidity, backscatter and the absorption of red light change the mean and the contrast of each colour channel of an underwater image. We propose a **gated input standardization (GIN)** for YOLOv11n that standardizes each colour channel of the input image and blends the result with the image through a learnable gate, and we study two modules for the detector, **PWAM** (phase-aware amplitude and phase aggregation) and **SRAW-TM** (a transformer block with space-reduction attention). All models were trained under one protocol with several seeds. The main results:

* **GIN under degradation, mini-Trash.** With light augmentation (no mixup, no shear), GIN raised the mean mAP50 on the nine *named* corruptions (blur, contrast reduction, turbidity) from 82.7 % to 84.0 % (+1.3 points, Welch p = 0.008; hypothesis and level fixed before the runs, 10 new seeds). The gain grows with the strength of turbidity and contrast reduction and reaches +6.3 points under turbidity with t = 0.4 (p < 0.001). Under this turbidity GIN detects more objects than YOLOv11n on 97 of the 202 test images and fewer on 5, averaged over ten seeds of each model, and finds 79.2 % of the annotated objects against 69.0 %.
* **GIN under degradation, TrashCan 1.0.** Over five seeds (10–14), nine named corruptions 54.4 ± 2.8 vs. 45.1 ± 3.7 (+9.3 points, p = 0.002), with +21.9 points under turbidity with t = 0.6. GIN with light augmentation was chosen among three candidates on the validation split, and this corruption comparison was not fixed before the runs.
* **Clean accuracy unchanged.** Clean test mAP50 89.8 vs. 89.9 on mini-Trash (p = 0.87) and 76.8 ± 4.1 vs. 74.1 ± 3.7 on TrashCan (p = 0.30). GIN adds three parameters.
* **Where GIN does not help.** Blur, motion blur, defocus and JPEG compression are not affine changes of the colour channels. Over the twelve additional corruptions GIN changes the mean mAP50 by −0.8 points (p = 0.26), and under JPEG compression with quality 8 it is 4.5 points lower.
* **Modules.** In an ablation inside YOLOv11n, no variant of PWAM or SRAW-TM is significantly better than YOLOv11n in clean mAP50 or mAP50-95. PWST-YOLOv11 reaches 86.4 ± 1.0 test mAP50 against 89.3 ± 1.1 for YOLOv11n, with 57 % more GMACs.
* **Training recipe.** Removing mixup and shear raised the test mAP50-95 of YOLOv11n by 3.3 points, and AmpMix (Fourier amplitude mixing, training only) raised the mean mAP50 on twelve additional corruptions by 2.8 points.
* **Benchmark.** The frame-level split of mini-Trash puts near-duplicates of 24 % of the test frames into the training set; on a split by source clip the test mAP50 of all models is far lower (43.8–54.4 %).

All numbers come from the raw result files in [`results/raw`](results/raw); the pre-registration lines (time-stamped design, comparators, endpoints and levels, written before the runs) are in [`preregistrations.txt`](preregistrations.txt).

### GIN

The first convolution of YOLOv11n receives the image after a per-image, per-channel standardization that is blended with the identity by a learnable gate (3 parameters, initialized at 0.5, active at training and inference):

`x~_c = x_c + a_c * ( (x_c - mu_c) / sqrt(var_c + eps) * s + m - x_c )`  with `m = 0.4`, `s = 0.18`, `eps = 1e-3`

Turbidity, contrast reduction and red-channel attenuation are affine changes `x'_c = alpha_c * x_c + beta_c` of each channel, and the standardized term is unchanged by them. After training the gates are 0.29 ± 0.04 on mini-Trash and 0.18 ± 0.03 on TrashCan. GIN is implemented as `GatedIN` / `ConvGIN5` in [`modules_added_in_revision.py`](modules_added_in_revision.py); the model is [`configs/x_gin5.yaml`](configs/x_gin5.yaml) (YOLOv11n with `ConvGIN5` as layer 0, 2.59 M parameters).

## Repository layout

| Path | Content |
|---|---|
| `ultralytics_patch/pwst_ultralytics.patch` | patch against Ultralytics 8.4.2 that registers all modules (PWAM, SRAW-TM, AmpMix, GIN and the other candidates) |
| `pwst_modules.py`, `modules_added_in_revision.py` | readable copies of the module classes (the patch contains the same code) |
| `configs/` | model configurations (`x_gin5.yaml` is the GIN model; `pw_*.yaml` are the module variants of the ablation; `x_*.yaml` the other candidates) |
| `train.py` | portable training + evaluation script (the recipe of the paper) |
| `scripts/eval_corruptions.py` | evaluation under 21 synthetic corruptions (9 named + 12 additional) |
| `scripts/analyze_gin.py` | reproduces the pre-registered test of GIN from the raw results |
| `scripts/gen_deg_fig.py` | detections of YOLOv11n and GIN under one corruption, with the per-image counts over all seeds |
| `scripts/heatmap_align.py` | heat maps of the stride-16 input of the detection head and their alignment with the ground-truth boxes |
| `scripts/*` | split analysis (near-duplicates, clip-level split), TrashCan overlap, latency, FLOPs, cross-validation, statistics |
| `results/raw/` | one line per trained model (`summary.jsonl`), mAP50 under every corruption (`robust2.json`), cross-validation of all candidates (`cv_results.json`), per-image counts under turbidity (`deg_fig_*.json`), heat-map alignment (`heatmap_align.json`), learned gates (`gin_gates.json`) |
| `provenance/` | the experiment scripts as they were run on our workstation (hard-coded paths) |
| `docs/` | project page (GitHub Pages) |

## Quick start

```bash
# 1. environment (tested: Python 3.12, PyTorch 2.11, CUDA 12.8, Ultralytics 8.4.2)
pip install -r requirements.txt
git clone https://github.com/panzerjagerWang/PWST-YOLO-Mini-trash-Datset && cd PWST-YOLO-Mini-trash-Datset

# 2. register the modules in your Ultralytics installation (adds classes + parse_model entries)
SITE=$(python -c "import ultralytics,os;print(os.path.dirname(os.path.dirname(ultralytics.__file__)))")
patch -p1 -d "$SITE" < ultralytics_patch/pwst_ultralytics.patch

# 3. use the weights (Releases -> weights.zip; 28 mini-Trash and 10 TrashCan checkpoints, all seeds, with per-seed metrics and SHA-256 sums)
unzip weights.zip -d weights
python - <<'PY'
from ultralytics import YOLO
m = YOLO("weights/x_gin5_lightaug/x_gin5_lightaug_seed90.pt")
m.predict("your_underwater_image.jpg", imgsz=640, save=True)
PY
```

## Reproduce the experiments

**Data.** mini-Trash is `mini-Trash.zip` in the [Releases](../../releases): 1006 frames from J-EDI, classes *debris*, *bio*, *machine*, 603 / 201 / 202 images for training / validation / test, split at the frame level, in YOLO format with a `data.yaml`. The images are frames of JAMSTEC J-EDI deep-sea video, and their use is subject to the J-EDI terms of use; please cite J-EDI and the paper. The split analysis scripts need the image folders.

**Train GIN and its baseline** (the recipe of the paper: 300 epochs, patience 100, batch 16, 640 px, COCO-pretrained YOLOv11n initialization; `--light-aug` removes mixup and shear):

```bash
for s in 90 91 92 93 94 95 96 97 98 99; do
  python train.py --cfg configs/x_gin5.yaml --data data.yaml --name x_gin5_lightaug --seed $s --light-aug
done
for s in 0 1 2 3 4 5 6 7 8 9; do
  python train.py --cfg yolo11n.yaml --data data.yaml --name base_lightaug --seed $s --light-aug
done
```

**Evaluate under corruptions** (test split only, mAP50 in %, 9 named + 12 additional conditions):

```bash
python scripts/eval_corruptions.py data.yaml robust.json x_gin5_la_s90=runs/x_gin5_lightaug_s90/weights/best.pt  # one label=weights per model
```

**Qualitative figure under turbidity** (first weights of each list in the figure, all weights in the per-image counts):

```bash
TEST_DIR=dataset/test CODE_DIR=. python scripts/gen_deg_fig.py out_deg haze_0.4 BASE=runs/base_lightaug_s0/weights/best.pt,... GIN=runs/x_gin5_lightaug_s90/weights/best.pt,...
```

**Statistics.** `python scripts/analyze_gin.py` recomputes the pre-registered test (primary endpoint: mean mAP50 over the nine named corruptions, Welch two-sided test, level 0.05) from `results/raw`. With your own runs, put them in the same JSON formats.

**TrashCan 1.0.** `scripts/make_trashcan_yolo.py` converts the release to YOLO format, `scripts/run_trashcan.py NAME MODEL.yaml SEED 200` trains one model (200 epochs, SGD, batch 8, lr0 0.01 with cosine decay; `LIGHT=1` for light augmentation), `scripts/eval_trashcan.py` evaluates finished runs, and `scripts/tc_aggregate.py` and `scripts/tc_robust_aggregate.py` recompute the tables of Section IV-J and Supplementary S8 and S9 from `results/raw/summary_trashcan.jsonl` and `results/raw/robust_*.json`. The queue files `scripts/queue_tc_*.sh` list the runs in the order they were made; the selection rule of round 2 is in `preregistrations.txt`.

**Other analyses.** `scripts/heatmap_align.py` (heat maps and alignment metrics), `scripts/overlap_analysis.py` (near-duplicates between the splits), `scripts/group_split.py` (clip-level split), `scripts/trashcan_overlap.py` (mini-Trash vs. TrashCan 1.0), `scripts/bench_latency.py`, `scripts/count_flops.py`, `scripts/cv_eval.py` (clip-grouped cross-validation), `scripts/analyze_tc.py` (initialization from TrashCan).

## Limitations

* The GIN hypothesis was formulated after three other variants that contained GIN had shown the same direction on the nine named corruptions (as secondary, uncorrected observations) and after about 25 candidates had been screened; the supplementary material lists all of them. The confirmation uses new seeds and a single comparison, but the selection history should be kept in mind.
* The gain comes from turbidity, contrast reduction and red-channel attenuation; the clean accuracy and the mean over the twelve additional corruptions are unchanged, and strong JPEG compression lowers mAP50. On TrashCan the seed standard deviation is about 4 points, and the clean accuracies of GIN and YOLOv11n cannot be distinguished with five seeds.
* The corruptions are synthetic. The benchmark is small (202 test images) and frame-level split; see the data analysis in the paper before drawing conclusions about generalization to new sites.

## License

The code is released under the GNU AGPL-3.0 (it modifies Ultralytics, which is AGPL-3.0). The model weights are provided for research use.

## Citation

The manuscript is under review; until it is published, please cite this repository:

```bibtex
@misc{wang2026pwst,
  title  = {Enhancing Underwater Object Detection via Phase-Aware Wave-like Aggregation and Space-Reduction Attention},
  author = {Wang, Jiaqing and Zeng, Baichuan and Zhou, Hexiong and Zeng, Zheng},
  year   = {2026},
  note   = {Code and weights, manuscript under review at IEEE Journal of Oceanic Engineering},
  url    = {https://github.com/panzerjagerWang/PWST-YOLO-Mini-trash-Datset}
}
```

Contact: zhou_hexiong@sjtu.edu.cn
