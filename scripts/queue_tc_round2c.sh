#!/bin/bash
# last three jobs of TrashCan round 2: start one when fewer than six jobs run and more than 6000 MB of RAM is available
R=/media/w/DATA1/Paper/JOE_paper
PY=/home/w/miniconda3/envs/isaacsim6/bin/python
cd $R/Code
JOBS=("tclight_gin5 $R/revision_runs/cfg/x_gin5.yaml 1" "tclight_base yolo11n.yaml 2" "tclight_gin5 $R/revision_runs/cfg/x_gin5.yaml 2")
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  n=$(ps -eo cmd | grep "^$PY run_trashcan" | awk '{print $3, $5}' | sort -u | wc -l)
  avail=$(free -m | awk 'NR==2 {print $7}')
  if [ "$n" -lt 6 ] && [ "$avail" -gt 6000 ]; then
    set -- ${JOBS[$i]}
    rm -rf $R/revision_runs/${1}_s${3}
    echo "$(date '+%m-%d %H:%M') start $1 s$3 (running $n, avail $avail)"
    (WORKERS=3 LIGHT=1 setsid nohup $PY run_trashcan.py $1 $2 $3 200 > $R/revision_runs/log_${1}_s${3}.txt 2>&1 < /dev/null &)
    i=$((i + 1))
    sleep 600
  else
    sleep 120
  fi
done
echo "$(date '+%m-%d %H:%M') all started"
