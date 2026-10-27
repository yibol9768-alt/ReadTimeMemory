# Source this: AMB with luna as answerer, judge and (for hindsight) extraction model. The key is read, never printed.
export SOLKEY=$(sed -n 13p ~/.local/share/codex-workflow/llm-proxy-private/config-pro.yaml | sed -E 's/^\s*-\s*"?([^"]+)"?\s*$/\1/')
export OPENAI_API_KEY=$SOLKEY OPENAI_BASE_URL=http://127.0.0.1:18318/v1
export OMB_ANSWER_LLM=openai OMB_ANSWER_MODEL=gpt-6-luna OMB_JUDGE_LLM=openai OMB_JUDGE_MODEL=gpt-6-luna
export AMB_HINDSIGHT_LLM_PROVIDER=openai AMB_HINDSIGHT_LLM_MODEL=gpt-6-luna
export AMB_HINDSIGHT_LLM_BASE_URL=http://127.0.0.1:18318/v1 AMB_HINDSIGHT_LLM_API_KEY=$SOLKEY
export BEAM_DATA_PATH=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/V3/data/beam/beam100k_amb.json
export PATH=$HOME/v3venvs/hindsight/bin:$HOME/.local/bin:$PATH
export GEMINI_API_KEY=unused-not-called   # AMB checks it exists; answer/judge/extraction all go to luna
export HINDSIGHT_EMBED_DAEMON_STARTUP_TIMEOUT=1200   # first start downloads bge-small and the cross-encoder
# Self-hosted Hindsight server (hindsight-api 1.0.0 from this env) for AMB's hindsight-http provider
export HINDSIGHT_API_LLM_PROVIDER=openai HINDSIGHT_API_LLM_MODEL=gpt-6-luna
export HINDSIGHT_API_LLM_BASE_URL=http://127.0.0.1:18318/v1 HINDSIGHT_API_LLM_API_KEY=$SOLKEY
export HINDSIGHT_HTTP_URL=http://127.0.0.1:8888
