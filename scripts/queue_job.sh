#!/bin/bash
# one training job.  usage: queue_job.sh KEY:SEED:RECIPE
#   KEY    base|F|A|B|C|D|E|G|H|I|P|W|V8full|V8amp|V8phase        (P = released PWST-YOLOv11 configuration, P2 head + PWAM_Lite)
#   RECIPE uw    train set = original + 2 corrupted copies, paper augmentation (mixup .5, shear 20), 100 epochs
#          uwl   same train set, light augmentation (mixup 0, shear 0), 100 epochs
#          la    original train set, light augmentation, 300 epochs
#          orig  original train set, paper augmentation, 300 epochs
#          corig clip-level split (revision_runs/clipsplit), paper augmentation, 300 epochs
#          cuwl  clip-level split + 2 degraded copies of every training image, light augmentation, 100 epochs
#          cla   clip-level split, light augmentation (mixup 0, shear 0), 300 epochs
#          nomix    original train set, mixup 0 / shear 20 (decomposition of 'la'), 300 epochs
#          noshear  original train set, mixup 0.5 / shear 0 (decomposition of 'la'), 300 epochs
#          cv[_V] fold SEED of the clip-grouped CV pool (revision_runs/cv_V, V=raw by default), light augmentation, 100 epochs
cd /media/w/DATA1/Paper/JOE_paper
PY=/home/w/miniconda3/envs/isaacsim6/bin/python
R=revision_runs
export OMP_NUM_THREADS=2 PWAM_MODE=full WORKERS=4
IFS=: read K S RC <<< "$1"
FOLD=$((S % 4))      # CV recipes: SEED 0..3 = folds, 4..7 and 8..11 = replicate trainings of the same folds
case $K in
  base) Y=yolo11n.yaml;                              SN=base;            ON=yolo11n_base;;
  F)    Y=$R/cfg/pw_F_pw357.yaml;                    SN=pwF;             ON=pw_F_pw357;;
  A)    Y=$R/cfg/pw_A_sraw57.yaml;                   SN=pwA;             ON=pw_A_sraw57;;
  B)    Y=$R/cfg/pw_B_pw01_sraw57.yaml;              SN=pwB;             ON=pw_B_pw01_sraw57;;
  C)    Y=$R/cfg/pw_C_sraw357.yaml;                  SN=pwC;             ON=pw_C_sraw357;;
  D)    Y=$R/cfg/pw_D_sra57.yaml;                    SN=pwD;             ON=pw_D_sra57;;
  E)    Y=$R/cfg/pw_E_sraw7.yaml;                    SN=pwE;             ON=pw_E_sraw7;;
  G)    Y=$R/cfg/pw_G_c3k2pw468.yaml;                SN=pwG;             ON=pw_G_fix;;
  H)    Y=$R/cfg/pw_H_neck.yaml;                     SN=pwH;             ON=pw_H_fix;;
  I)    Y=$R/cfg/pw_I_c3k2pwsra68.yaml;              SN=pwI;             ON=pw_I_fix;;
  P)    Y=/media/w/DATA1/Paper/JOE_paper/Code/dataset/yolov11n_pwstv7final.yaml; SN=pwst; ON=pwst_v7final;;
  W)    Y=$R/cfg/pw_W_wpatm357.yaml;                 SN=pwW;             ON=pw_W_wpatm357;;   # WaveMLP PATM at the pw_F positions
  Famp)   Y=$R/cfg/pw_F_pw357.yaml;                    SN=pwFamp;          ON=pwF_amp;    export PWAM_MODE=amp;;     # pw_F with the phase of every PWAM fixed to zero (amplitude only)
  Fphase) Y=$R/cfg/pw_F_pw357.yaml;                    SN=pwFphase;        ON=pwF_phase;  export PWAM_MODE=phase;;   # pw_F with the amplitude of every PWAM fixed to one (phase only)
  V8full|V8amp|V8phase) Y=/media/w/DATA1/Paper/JOE_paper/Code/dataset/yolov11n_pwstv8.yaml; SN=v8_${K#V8}; ON=v8_${K#V8}; export PWAM_MODE=${K#V8};;   # full-PWAM configuration, amplitude / phase ablation
  x_*)  Y=$R/cfg/$K.yaml;                            SN=$K;              ON=$K;;      # v3 modules (scripts/make_cfgs.py)
esac
case $RC in
  uw)   export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/Code/dataset_uw DATA_TAG=uw;                    NAME=${K}_uw;       EP=100;;
  uwl)  export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/Code/dataset_uw DATA_TAG=uw MIXUP=0.0 SHEAR=0.0; NAME=${K}_uwl;      EP=100;;
  la)   export MIXUP=0.0 SHEAR=0.0;                                                                       NAME=${SN}_lightaug; EP=300;;
  orig) NAME=$ON; EP=300;;
  corig)   export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/clipsplit DATA_TAG=clip;          NAME=clip_${SN};      EP=300;;   # clip-level split, paper recipe
  cla)     export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/clipsplit DATA_TAG=clip MIXUP=0.0 SHEAR=0.0;         NAME=clip_${SN}_la;   EP=300;;   # clip-level split, light augmentation (no DA), 300 epochs
  tcorig)  export INIT_WEIGHTS=/media/w/DATA1/Paper/JOE_paper/revision_runs/trashcan_pretrain_v/weights/best.pt;                          NAME=tc_${ON};          EP=300;;   # TrashCan-initialised, paper augmentation
  tcla)    export INIT_WEIGHTS=/media/w/DATA1/Paper/JOE_paper/revision_runs/trashcan_pretrain_v/weights/best.pt MIXUP=0.0 SHEAR=0.0;  NAME=tc_${SN}_lightaug; EP=300;;   # TrashCan-initialised, light augmentation
  cuwl)    export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/clipsplit_uw DATA_TAG=clipuw MIXUP=0.0 SHEAR=0.0; NAME=clip_${SN}_uwl; EP=100;;   # clip-level split, degradation-aware + light augmentation
  nomix)   export MIXUP=0.0 SHEAR=20.0; NAME=${SN}_nomix;   EP=300;;
  cv_hsv)  export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/cv_raw/f$FOLD DATA_TAG=cvhsv$FOLD MIXUP=0.0 SHEAR=0.0 HSV_H=0.05 HSV_S=0.9 HSV_V=0.6; NAME=cv_hsv_${SN}; EP=100;;
  cv_scale) export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/cv_raw/f$FOLD DATA_TAG=cvscale$FOLD MIXUP=0.0 SHEAR=0.0 SCALE=0.9; NAME=cv_scale_${SN}; EP=100;;
  cv_nomosaic) export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/cv_raw/f$FOLD DATA_TAG=cvnomosaic$FOLD MIXUP=0.0 SHEAR=0.0 MOSAIC=0.0; NAME=cv_nomosaic_${SN}; EP=100;;
  cv|cv_*) V=${RC#cv}; V=${V#_}; V=${V:-raw}          # clip-grouped 4-fold CV on the design pool; SEED = fold index, light augmentation, 100 epochs
           export DATA_ROOT=/media/w/DATA1/Paper/JOE_paper/revision_runs/cv_${V}/f$FOLD DATA_TAG=cv${V}$FOLD MIXUP=0.0 SHEAR=0.0; NAME=cv_${V}_${SN}; EP=100;;
  noshear) export MIXUP=0.5 SHEAR=0.0;  NAME=${SN}_noshear; EP=300;;
esac
case $RC in cv*) export WORKERS=2;; esac     # short jobs, keep the host memory low
[ "$K" = P ] && case $RC in uw|uwl) export DROP_LAST=1;; esac   # see revision_run_exp3.py
L=$R/log_${NAME}_s$S.txt
[ "$2" = "--log" ] && { echo $L; exit 0; }
[ -e $L ] && exit 0
$PY revision_run_exp3.py $NAME $Y $S $EP > $L 2>&1
