#!/bin/sh

#./RunVanillaRLMedium.sh 200
#cd ..
pathname="./PPPO.py"

for VARIABLE in {0..29}
do
python3 $pathname --batch_choice $(($VARIABLE)) --clip_eps 0.1 --safety_requirement 0.05 --lr 0.001 --epochs 3100
done