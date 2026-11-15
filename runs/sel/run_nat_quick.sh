#!/usr/bin/env bash
# H20, run from ~/data/rtm/sel. prepare retains the generation servers for LLM Qs.
# After data is ready an operator stops rtm-v3-q4/q5/g/o and creates nat_gpus47.ok.
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export TZ=Asia/Singapore
MODE=${1:-all}
[[ $# == 0 ]] || shift
case "$MODE" in all|prepare|rebuild|train|trainj) ;; *) echo 'Usage: bash run_nat_quick.sh [all|prepare|rebuild|train|trainj] [mine3 pilot directories...]'; exit 2 ;; esac
ARM=nat
[[ "$MODE" != trainj ]] || ARM=natj
exec 9>/tmp/rtm_nat_quick.lock
flock -n 9 || { echo 'Another native quick launcher is running'; exit 2; }
mkdir -p s23
exec > >(tee -a s23/run_nat_quick.log) 2>&1
log(){ echo "[$(date '+%F %T SGT')] $*"; }
[[ $(md5sum s23.py | cut -d' ' -f1) == 31359d7eef61432441c21b66894af99b ]] || { log 'FAIL s23.py differs from audited H20 source'; exit 1; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
M=/home/centuser/data/models
ROOT=s23/all_rr8_mr4_2e437ddad635/lobo
T=(--seed 0 --teacher rr8+mr4 --teacher-config s23/teachsel/teacher_mr4.json)
RUN_ID="$(date +%m%d%H%M%S)-$$"
export S23_NAT_OWNER="rtm-nat-$RUN_ID"
export NAT_RUNTIME_CONFIG="$PWD/nat_runtime.json"
PIDS=()
WATCHDOG=''
PID_FILE=$(mktemp /tmp/rtm_nat_children.XXXXXXXX)
start_child(){
  [[ ! -f "$PID_FILE.expired" ]] || exit 143
  setsid "$@" 9>&- & CHILD_PID=$!
  local statline rest
  local -a fields
  if IFS= read -r statline 2>/dev/null < "/proc/$CHILD_PID/stat"; then
    rest=${statline##*) }
    read -r -a fields <<< "$rest"
    printf '%s %s\n' "$CHILD_PID" "${fields[19]}" >> "$PID_FILE"
  fi
}
run_child(){
  start_child "$@"
  wait "$CHILD_PID"
}
remove_own(){
  local name
  while IFS= read -r name; do
    case "$name" in "$S23_NAT_OWNER"-*) docker rm -f "$name" >/dev/null 2>&1 || true ;; esac
  done < <(docker ps -a --format '{{.Names}}')
}
cleanup(){
  local code=$?
  trap - EXIT INT TERM
  if [[ ! -f "$PID_FILE.expired" ]]; then
    [[ -z "$WATCHDOG" ]] || kill "$WATCHDOG" 2>/dev/null || true
    python3 -B nat_watchdog.py --cleanup --owner "$S23_NAT_OWNER" --pid-file "$PID_FILE" --parent "$$" || true
  fi
  # On expiry the independent watchdog retains the registry for its KILL sweep.
  log "EXIT rc=$code; removed only this invocation's $S23_NAT_OWNER-* containers"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# This process directly removes owned containers and signals every recorded
# child/group. Bash cannot defer it while waiting for pool/planning/training.
setsid python3 -B nat_watchdog.py --owner "$S23_NAT_OWNER" --pid-file "$PID_FILE" --parent "$$" 9>&- & WATCHDOG=$!
(( $(python3 -B nat_runtime.py remaining) > 0 )) || { log 'REFUSE: 06:45 SGT deadline has passed'; exit 1; }
# Reuse the launcher's exports, without executing any of its jobs/container code.
if [[ -f run_lobo1107b.sh ]]; then
  source <(python3 - <<'PY'
import os, shlex
from pathlib import Path
for line in Path('run_lobo1107b.sh').read_text().splitlines():
    try: tokens = shlex.split(line, comments=True)
    except ValueError: continue
    for token in tokens:
        key, sep, value = token.partition('=')
        if sep and key in ('S23_STAGE1_PROVENANCE', 'INNER_RANK_PAR'):
            value = os.path.expanduser(os.path.expandvars(value))
            if '$' not in value:
                print('export ' + key + '=' + shlex.quote(value))
PY
  )
fi
export S23_STAGE1_PROVENANCE=${S23_STAGE1_PROVENANCE:-s23dev/stage1_provenance.json}
export INNER_RANK_PAR=${INNER_RANK_PAR:-16}
log "provenance=$S23_STAGE1_PROVENANCE INNER_RANK_PAR=$INNER_RANK_PAR owner=$S23_NAT_OWNER"
[[ -f "$S23_STAGE1_PROVENANCE" ]] || { log 'FAIL missing stage-1 provenance'; exit 1; }
run_child python3 -B nat_runtime.py estimate --root "$ROOT"
waitp(){
  local ok p
  for _ in $(seq 90); do
    ok=1
    for p in "$@"; do curl -sf -m 3 "http://127.0.0.1:$p/v1/models" >/dev/null || ok=0; done
    [[ $ok == 0 ]] || return 0
    sleep 5
  done
  return 1
}
if [[ "$MODE" == all || "$MODE" == prepare ]]; then
  if [[ $# -gt 0 ]]; then PILOTS=("$@"); else
    PILOTS=()
    for d in mine3_pilot*; do
      [[ -d "$d" && ( -f "$d/mine.jsonl" || -d "$d/checkpoints/mine" ) ]] && PILOTS+=("$d")
    done
  fi
  (( ${#PILOTS[@]} > 0 )) || { log 'FAIL no mine3 pilot directories'; exit 1; }
  # Existing seed3_models pools: family 0 writer, family 2 independent intent judge.
  Q27_SPEC=${Q27_SPEC:-Qwen3.8-27B-FP8@18344,Qwen3.8-27B-FP8@18345}
  GEM_SPEC=${GEM_SPEC:-gemma-4-31b-it@18346}
  OSS_SPEC=${OSS_SPEC:-gpt-oss-120b@18347}
  log "native question generation: ${PILOTS[*]}"
  run_child python3 -B mine3_to_env.py "${PILOTS[@]}" --q27 "$Q27_SPEC" --gem "$GEM_SPEC" --oss "$OSS_SPEC" --par 8
  python3 - <<'PY'
import json
r=json.load(open('env_nat.json.report.json'))
assert r['single_fact_questions'] > 0, 'no LLM-accepted native single-fact questions; refusing training'
PY
  # Pool cache identity must match this exact env, including paired-round gold.
  python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('pool_nat');p.mkdir(exist_ok=True)
h=hashlib.md5(Path('env_nat.json').read_bytes()).hexdigest()
f=p/'env_md5.txt'
if f.exists():
    assert f.read_text()==h, 'env_nat changed; use a fresh pool_nat directory'
else:
    assert not list(p.glob('*.json')), 'unmanifested pool_nat cache; move it aside first'
    f.write_text(h)
PY
  log 'serve native embedding on GPU 0 and RR8 on GPUs 1,2,3 (only own containers)'
  run_child docker run -d --name "$S23_NAT_OWNER-emb" --gpus '"device=0"' --ipc=host --network host \
    -v "$M/qwen/Qwen3-Embedding-0.6B:/model:ro" "$IMG" /model \
    --served-model-name Qwen3-Embedding-0.6B --host 127.0.0.1 --port 18511 \
    --runner pooling --convert embed --max-model-len 8192 --gpu-memory-utilization 0.85 >/dev/null
  for g in 1 2 3; do run_child bash serve_sel.sh "$S23_NAT_OWNER-rr$g" "$g" "$((18511+g))" "$M/qwen/Qwen3-Reranker-8B"; done
  waitp 18511 18512 18513 18514 || { log 'FAIL pool servers'; exit 1; }
  # Match W20 labelled groups: RR8 probabilities, not ensemble logits. The
  # existing real teacher sources still consume rr8+mr4 via teacher_mr4.json.
  run_child python3 -B pool.py env_nat.json --out pool_nat --emb-port 18511 --rr-port 18512,18513,18514 --rr-model sel --P 600 --T 300 --par 8
  remove_own
fi
if [[ "$MODE" != train && "$MODE" != trainj ]]; then
  TOKENIZER_ARGS=()
  if [[ -n "${NAT_TOKENIZER:-}" ]]; then
    TOKENIZER_ARGS=(--tokenizer "$NAT_TOKENIZER")
  elif [[ -d "$M/qwen/Qwen3-Reranker-0.6B" ]]; then
    TOKENIZER_ARGS=(--tokenizer "$M/qwen/Qwen3-Reranker-0.6B")
  else
    log 'FAIL: reranker tokenizer directory missing; refusing char proxy'; exit 1
  fi
  log 'offline refilter: serialized support protection and training-visible gold'
  run_child python3 -B mine3_to_env.py --refilter env_nat.json --out env_nat2.json "${TOKENIZER_ARGS[@]}" "$@"
  # pool_nat belongs to env_nat (parent), not the filtered env hash. Verify
  # lineage/subset/text identity before reusing question-id pool entries.
  run_child python3 -B nat_artifacts.py validate
  HS=${HS:-$(cat hs.txt 2>/dev/null || echo 10)}
  log "mkdata from env_nat2: HS=$HS; reuse pool_nat; skip slow fnchk"
  run_child python3 -B mkdata.py env_nat2.json pool_nat tr_nat.rebuild.jsonl --no-set --ignore-kinds repeat,fu --hard-skip "$HS" --boost 3 --steps 8 --ids nat_tr2.txt
  run_child python3 -B nat_artifacts.py publish --root "$ROOT"
fi
run_child python3 -B nat_artifacts.py check
# Freeze full plans before consuming GPU time. trainj uses jia start, no stage 1;
# repetition factors, teacher fill deltas and exact FLOPs are in plan.json.
for fold in loc ev; do
  run_child python3 -B s23nat.py lobo --arm "$ARM" --fold "$fold" "${T[@]}" --dry-run
 done
run_child python3 -B nat_runtime.py check --root "$ROOT" --arm "$ARM" --phase all
if [[ "$MODE" == prepare || "$MODE" == rebuild ]]; then
  log 'PREPARED: next stop generation containers manually, touch nat_gpus47.ok, then run train'
  exit 0
fi
log 'WAIT: operator must stop rtm-v3-q4 rtm-v3-q5 rtm-v3-g rtm-v3-o, then touch nat_gpus47.ok. This script never stops those servers.'
while :; do
  running=$(run_child docker ps --format '{{.Names}}')
  if [[ -f nat_gpus47.ok ]] && ! printf '%s\n' "$running" | grep -Eq '^rtm-v3-(q4|q5|g|o)$'; then break; fi
  # No training starts while waiting; abort as soon as the remaining path no longer fits.
  run_child python3 -B nat_runtime.py check --root "$ROOT" --arm "$ARM" --phase all
  sleep 10
 done
if [[ "$ARM" == nat ]]; then
run_child python3 -B nat_runtime.py check --root "$ROOT" --arm "$ARM" --phase s1
log 'TRAIN native stage 1: two passes, worlds 4 then 2, shared by both folds'
start_child python3 -B s23nat.py s1 --arm nat --gpus 0,1,2,3 "${T[@]}"; PIDS=("$CHILD_PID")
wait "${PIDS[0]}"; PIDS=()
fi
run_pair(){
  local smoke=$1
  local extra=()
  [[ "$smoke" == 0 ]] || extra=(--smoke-steps 50)
  start_child python3 -B s23nat.py lobo --arm "$ARM" --fold loc --gpus 0,1,2,3 "${T[@]}" "${extra[@]}" > "s23/${ARM}_loc_${smoke}.log" 2>&1; PIDS=("$CHILD_PID")
  start_child python3 -B s23nat.py lobo --arm "$ARM" --fold ev --gpus 4,5,6,7 "${T[@]}" "${extra[@]}" > "s23/${ARM}_ev_${smoke}.log" 2>&1; PIDS+=("$CHILD_PID")
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  PIDS=()
}
run_child python3 -B nat_runtime.py check --root "$ROOT" --arm "$ARM" --phase smoke
log 'SMOKE 50 steps/stage: loc GPUs 0-3, ev GPUs 4-7'
run_pair 50
run_child python3 -B nat_runtime.py check --root "$ROOT" --arm "$ARM" --phase full
log 'FULL LOBO: loc GPUs 0-3, ev GPUs 4-7'
run_pair 0
run_child python3 -B nat_summary.py --root "$ROOT" --arm "$ARM"
log 'NAT_QUICK_DONE'
