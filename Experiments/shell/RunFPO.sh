#!/bin/sh

pathname="./FPO.py"

for VARIABLE in {0..29}
do
python3 -u -W ignore $pathname --alpha 0.01 --batch_choice $(($VARIABLE)) --epochs 150000 > FPO_log/fpo_bc10_a001_time.log 2>&1
done