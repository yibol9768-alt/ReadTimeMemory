#!/bin/bash
cd ~/data/rtm/sel
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
rm -rf wt
for s in plan render verify build gates; do python3 world.py $s --out wt --start 0 --n 3 --rounds 80 $A || { echo "FAIL $s"; exit 1; }; done; echo WTINYDONE
