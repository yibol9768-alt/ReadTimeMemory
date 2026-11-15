#!/bin/bash
# tiny check, then S1: 5 full worlds in parallel
cd ~/data/rtm/sel
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
bash wtiny.sh > wtiny.log 2>&1
grep -q WTINYDONE wtiny.log || { echo "FAIL tiny"; exit 1; }
echo TINYOK
one(){ i=$1; for s in plan render verify build gates; do python3 world.py $s --out ws1 --start $i --n 1 --rounds 240 --par 24 $A > ws1_$i.$s.log 2>&1 || { echo "FAIL world $i $s"; exit 1; }; done; echo "world $i done"; }
mkdir -p ws1
for i in 10 11 12 13 14; do one $i & done
wait; echo S1DONE
