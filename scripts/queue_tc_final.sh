#!/bin/bash
# final TrashCan seeds 10-14 for the light-recipe pair, then the rerun of the diverged tcdet_gin0 seed 1.
# Started for scheduling only after queue_tc_round2c.sh has started all its jobs; their numbers are read only after
# the round-2 selection on validation (preregistrations.txt). Same rule as round2c: start one job when fewer than six
# run and more than 6000 MB of RAM is available.
R=/media/w/DATA1/Paper/JOE_paper
PY=/home/w/miniconda3/envs/isaacsim6/bin/python
cd $R/Code
while pgrep -f "[q]ueue_tc_round2c.sh" > /dev/null; do sleep 120; done
JOBS=()
for s in 10 11 12 13 14; do
  JOBS+=("tclight_gin5 $R/revision_runs/cfg/x_gin5.yaml $s 1" "tclight_base yolo11n.yaml $s 1")
done
JOBS+=("tcdet_gin0 $R/revision_runs/cfg/x_gin.yaml 1 0")
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  n=$(ps -eo cmd | grep "^$PY run_trashcan" | awk '{print $3, $5}' | sort -u | wc -l)
  avail=$(free -m | awk 'NR==2 {print $7}')
  if [ "$n" -lt 6 ] && [ "$avail" -gt 6000 ]; then
    set -- ${JOBS[$i]}
    d=$R/revision_runs/${1}_s${3}
    [ -d $d ] && mv $d ${d}_diverged
    [ -f $R/revision_runs/log_${1}_s${3}.txt ] && mv $R/revision_runs/log_${1}_s${3}.txt $R/revision_runs/log_${1}_s${3}_diverged.txt
    echo "$(date '+%m-%d %H:%M') start $1 s$3 light=$4 (running $n, avail $avail)"
    (WORKERS=3 LIGHT=$4 setsid nohup $PY run_trashcan.py $1 $2 $3 200 > $R/revision_runs/log_${1}_s${3}.txt 2>&1 < /dev/null &)
    i=$((i + 1))
    sleep 600
  else
    sleep 120
  fi
done
echo "$(date '+%m-%d %H:%M') all started"
