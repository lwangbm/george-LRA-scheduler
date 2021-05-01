#!/bin/sh

#./RunVanillaRLMedium.sh 200
#cd ..
pathname="./CPO.py"

for VARIABLE in {0..29}
do
python3 $pathname --batch_choice $(($VARIABLE)) --safety_requirement 0.05 --epochs 3100
done