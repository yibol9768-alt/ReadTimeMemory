#!/bin/bash
# B0 (10-31, audit fixes applied): can a conditioned selector use its own imperfect selections? Trained on BEAM-100K, evaluated on BEAM-500K and LongMemEval.
#  SA fresh 0.6B, prefixes up to 20; SB fresh 0.6B, prefixes up to 8; SC continue from T1, prefixes up to 20; A0 untrained, probes only. 15% empty prefixes.
# Pass rule (registered 10-31 15:50, PROGRESS): at least one arm has MAIN self5-plc top18 lower bound >0 on BEAM hard, wrong-prefix harm smaller than untrained
# (paired lower bound >0), and cond hard 16k lower bound above T1. LongMemEval and the far/masked lines say whether the gain is a BEAM position artifact.
set -u
cd ~/data/rtm/sel
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 14400 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
waitp(){ sleep 20; for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
probes(){ tag=$1; ports=$2
  python3 cond_probe2.py beam_ex.json hard_ids.txt pool --port $ports --out cond_${tag}b.json $3 > cond_${tag}b.log || return 1
  python3 cond_probe2.py q_lmes150.json ids_lmes150.txt pool_lmes150 --port $ports --out cond_${tag}l.json --min-gold 2 $3 > cond_${tag}l.log || return 1; }
if [ "${1:-}" = tiny ]; then
  bash serve_sel.sh rtm-sel4 4 18347 $B06; waitp 18347 || { log FAIL serve; exit 1; }
  probes t 18347 "--limit 3" || { log FAIL probes; docker rm -f rtm-sel4; exit 1; }
  docker rm -f rtm-sel4; cat cond_tb.log cond_tl.log; log TINYDONE; exit 0
fi
arm(){ n=$1; g=$2; tr=$3; dv=$4; base=$5; lr=$6; ep=$7; p1=$8; p2=$9; g1=${g%,*}; g2=${g#*,}
  trap "docker rm -f rtm-sel$g1 rtm-sel$g2 >/dev/null 2>&1" EXIT
  log "train $n"; TRAIN $n $g 2 --model $base --data /w/$tr --dev /w/$dv --out /w/ckpt/$n --lr $lr --epochs $ep --eval-every 100 --max-d 1100 --max-q 5200 || { log "FAIL $n train"; exit 1; }
  m=/home/centuser/data/rtm/sel/ckpt/$n
  bash serve_sel.sh rtm-sel$g1 $g1 $p1 $m; bash serve_sel.sh rtm-sel$g2 $g2 $p2 $m; waitp $p1 $p2 || { log "FAIL $n serve"; exit 1; }
  probes $n $p1,$p2 "" || { log "FAIL $n probes"; exit 1; }; log "$n probes done"
  python3 infer.py beam_ex.json --port $p1,$p2 --tag ${n^^}P_H --cond 0 --par 8 || { log "FAIL $n plain infer"; exit 1; }
  python3 infer.py beam_ex.json --port $p1,$p2 --tag ${n^^}C_H --cond 1 --per-step 2 --par 8 || { log "FAIL $n cond infer"; exit 1; }
  for t in ${n^^}P_H ${n^^}C_H; do [ "$(python3 -c "import json;print(len({json.loads(l)['qid'] for l in open('ctx_$t.jsonl')}))")" = "$(python3 -c "import json;print(len(json.load(open('beam_ex.json'))))")" ] || { log "FAIL $n $t incomplete"; exit 1; }; done
  log "$n done"; }
a0(){ trap "docker rm -f rtm-sel4 >/dev/null 2>&1" EXIT
  bash serve_sel.sh rtm-sel4 4 18347 $B06; waitp 18347 || { log "FAIL a0 serve"; exit 1; }
  probes a0 18347 "" || { log "FAIL a0 probes"; exit 1; }; log "a0 probes done"; }
rm -f ctx_SAP_H.jsonl ctx_SAC_H.jsonl ctx_SBP_H.jsonl ctx_SBC_H.jsonl ctx_SCP_H.jsonl ctx_SCC_H.jsonl
a0 &
arm sa 0,1 tr_b0a.jsonl dv_b0a.jsonl /models/qwen/Qwen3-Reranker-0.6B 1e-5 2 18341 18342 &
arm sb 2,3 tr_b0b.jsonl dv_b0b.jsonl /models/qwen/Qwen3-Reranker-0.6B 1e-5 2 18343 18344 &
arm sc 6,7 tr_b0a.jsonl dv_b0a.jsonl /w/ckpt/t1 5e-6 1 18345 18346 &
wait; log B0DONE
