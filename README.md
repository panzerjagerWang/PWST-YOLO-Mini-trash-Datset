# PWST-YOLO and Gated Input Standardization (GIN) for Underwater Debris Detection

Code, weights and reproduction material for the manuscript

> **Enhancing Underwater Object Detection via Phase-Aware Wave-like Aggregation and Space-Reduction Attention**
> Jiaqing Wang, Baichuan Zeng, Hexiong Zhou\*, Zheng Zeng\* (Shanghai Jiao Tong University; The Chinese University of Hong Kong)
> submitted to *IEEE Journal of Oceanic Engineering* (manuscript 26-0110, under revision)

**Project page:** https://panzerjagerwang.github.io/PWST-YOLO-Mini-trash-Datset/ &nbsp;·&nbsp; **Weights:** [Releases](../../releases) &nbsp;·&nbsp; **Supplementary material:** [`docs/supplementary.pdf`](docs/supplementary.pdf)

## What this repository shows

Underwater debris detection is hard because turbidity, backscatter and blur change the statistics of every frame. We studied two modules for YOLO detectors, **PWAM** (phase-aware amplitude/phase aggregation) and **SRAW-TM** (a space-reduction-attention Transformer block), and we re-ran all comparisons under one training protocol with several seeds. The honest summary of the revision:

* The gains that were first reported for PWST-YOLOv11 (88.1 vs. 79.2 mAP50) are **not reproduced** in a controlled re-run (YOLOv11n: 89.3 ± 1.1 mAP50 over 8 seeds; the original configuration is 2.8 points lower).
* **No module improved the clean test mAP50 of YOLOv11n significantly** in any pre-registered confirmation with new seeds (differences between −0.1 and +0.5 points, all p > 0.25).
* What did hold up is a **robustness effect** of a tiny input module, the **gated input standardization (GIN)**: with light augmentation (no mixup, no shear) it raised the mean mAP50 on the nine *named* corruptions (blur, contrast reduction, haze) from 82.7 % to 84.0 % (**+1.3 points, Welch p = 0.008**; hypothesis and level fixed before the runs, 10 new seeds), without changing the clean accuracy (89.8 vs. 89.9 mAP50, p = 0.87). It does not help on the twelve additional corruptions (−0.8 points, p = 0.26).
* On **TrashCan 1.0**, GIN with light augmentation was chosen among three candidates on the validation split and then compared with YOLOv11n under the same recipe over five new seeds (10–14): clean test mAP50 76.8 ± 4.1 vs. 74.1 ± 3.7 (+2.7 points, p = 0.30, not significant), nine named corruptions 54.4 ± 2.8 vs. 45.1 ± 3.7 (**+9.3 points, p = 0.002**), mostly under contrast reduction and haze; it is slightly lower under the strongest blur and defocus. This corruption comparison was not fixed before the runs. The re-run of the original PWST-YOLOv11 configuration on TrashCan (61.2 ± 5.7 vs. 63.8 ± 1.5 for YOLOv11n, three seeds) and all other TrashCan runs are in Supplementary S8.
* Removing mixup and shear from the augmentation recipe raised the test mAP50-95 of YOLOv11n by 3.3 points; AmpMix (Fourier amplitude mixing, training only) raised the mean mAP50 on twelve additional corruptions by 2.8 points.
* A data analysis of the benchmark: the frame-level split of mini-Trash puts near-duplicates of 24 % of the test frames into the training set; on a clip-level split the test mAP50 of all models is far lower (43.8–54.4 %).

All numbers come from the raw result files in [`results/raw`](results/raw); the pre-registration lines (time-stamped design, comparators, endpoints and levels, written before the runs) are in [`preregistrations.txt`](preregistrations.txt).

### GIN in one paragraph

The first convolution of YOLOv11n receives the image after a per-image, per-channel standardization that is blended with the identity by a learnable gate (3 parameters, initialized at 0.5, active at training and inference):

`x~_c = x_c + a_c * ( (x_c - mu_c) / sqrt(var_c + eps) * s + m - x_c )`  with `m = 0.4`, `s = 0.18`, `eps = 1e-3`

It is implemented as `GatedIN` / `ConvGIN5` in [`modules_added_in_revision.py`](modules_added_in_revision.py); the model is [`configs/x_gin5.yaml`](configs/x_gin5.yaml) (YOLOv11n with `ConvGIN5` as layer 0, 2.59 M parameters).

## Repository layout

| Path | Content |
|---|---|
| `ultralytics_patch/pwst_ultralytics.patch` | patch against **Ultralytics 8.4.2** that registers all modules (PWAM, SRAW-TM, AmpMix, GIN and the other candidates) |
| `pwst_modules.py`, `modules_added_in_revision.py` | readable copies of the module classes (the patch contains the same code) |
| `configs/` | model configurations (`x_gin5.yaml` is the GIN model; `pw_*.yaml` are the module variants of the controlled re-run; `x_*.yaml` the other candidates) |
| `train.py` | portable training + evaluation script (the recipe of the paper) |
| `scripts/eval_corruptions.py` | evaluation under 21 synthetic corruptions (9 named + 12 additional) |
| `scripts/analyze_gin.py` | reproduces the pre-registered test of GIN from the raw results |
| `scripts/*` | split analysis (near-duplicates, clip-level split), TrashCan overlap, latency, FLOPs, cross-validation, statistics |
| `results/raw/` | one line per trained model (`summary.jsonl`), mAP50 under every corruption (`robust2.json`), cross-validation of all candidates (`cv_results.json`) |
| `provenance/` | the original experiment scripts (hard-coded workstation paths) kept for transparency |
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

**Data.** mini-Trash (1006 frames from J-EDI, classes *debris*, *bio*, *machine*; 603 / 201 / 202 images for training / validation / test, split at the frame level) in YOLO format with a `data.yaml` (`train: train/images`, `val: val/images`, `test: test/images`). The frames are derived from the JAMSTEC J-EDI archive; please check the J-EDI terms of use before redistributing images. The split analysis scripts need the image folders.

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

**Statistics.** `python scripts/analyze_gin.py` recomputes the pre-registered test (primary endpoint: mean mAP50 over the nine named corruptions, Welch two-sided test, level 0.05) from `results/raw`. With your own runs, put them in the same JSON formats.

**TrashCan 1.0.** `scripts/make_trashcan_yolo.py` converts the release to YOLO format, `scripts/run_trashcan.py NAME MODEL.yaml SEED 200` trains one model (200 epochs, SGD, batch 8, lr0 0.01 with cosine decay; `LIGHT=1` for light augmentation), `scripts/eval_trashcan.py` evaluates finished runs, and `scripts/tc_aggregate.py` and `scripts/tc_robust_aggregate.py` recompute the tables of Section IV-J and Supplementary S8 and S9 from `results/raw/summary_trashcan.jsonl` and `results/raw/robust_*.json`. The queue files `scripts/queue_tc_*.sh` list the runs in the order they were made; the selection rule of round 2 is in `preregistrations.txt`.

**Other analyses.** `scripts/overlap_analysis.py` (near-duplicates between the splits), `scripts/group_split.py` (clip-level split), `scripts/trashcan_overlap.py` (mini-Trash vs. TrashCan 1.0), `scripts/bench_latency.py`, `scripts/count_flops.py`, `scripts/cv_eval.py` (clip-grouped cross-validation), `scripts/analyze_tc.py` (initialization from TrashCan).

## Notes on what the evidence supports

* The GIN hypothesis was formulated after three earlier variants that contained GIN had shown the same direction on the nine named corruptions (as secondary, uncorrected observations) and **after about 25 candidates had been screened**; the supplementary material lists all of them. The confirmation uses new seeds and a single comparison, but the selection history should be kept in mind.
* The effect is limited to blur, contrast reduction and haze; clean accuracy and the twelve additional corruptions are unchanged. On TrashCan the gain comes from contrast reduction and haze, the seed standard deviation is about 4 points, and the clean accuracies of GIN and YOLOv11n cannot be distinguished with five seeds.
* The benchmark is small (202 test images) and frame-level split; see the data analysis in the paper before drawing conclusions about generalization to new sites.

## License

The code is released under the **GNU AGPL-3.0** (it modifies Ultralytics, which is AGPL-3.0). The model weights are provided for research use.

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
