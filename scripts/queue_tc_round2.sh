#!/bin/bash
# TrashCan round 2 (pre-registered in preregistrations.txt before the first run): three arms, seeds 0-2, at most four jobs at a time
R=/media/w/DATA1/Paper/JOE_paper
PY=/home/w/miniconda3/envs/isaacsim6/bin/python
cd $R/Code
export WORKERS=3
job() {
  arm=$1; model=$2; seed=$3; light=$4
  LIGHT=$light $PY run_trashcan.py $arm $model $seed 200 > $R/revision_runs/log_${arm}_s${seed}.txt 2>&1
}
export -f job
export R PY
printf '%s\n' \
 "tcdet_gin0 $R/revision_runs/cfg/x_gin.yaml 0 0" "tcdet_gin0 $R/revision_runs/cfg/x_gin.yaml 1 0" "tcdet_gin0 $R/revision_runs/cfg/x_gin.yaml 2 0" \
 "tclight_base yolo11n.yaml 0 1" "tclight_base yolo11n.yaml 1 1" "tclight_base yolo11n.yaml 2 1" \
 "tclight_gin5 $R/revision_runs/cfg/x_gin5.yaml 0 1" "tclight_gin5 $R/revision_runs/cfg/x_gin5.yaml 1 1" "tclight_gin5 $R/revision_runs/cfg/x_gin5.yaml 2 1" \
 | xargs -P 4 -L 1 bash -c 'job $0 $1 $2 $3'
