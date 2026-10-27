# stateful-memory

Non-parametric long-term memory for LLM agents: an append-only observation store whose state (current values,
value histories, dependency rules with cascades, unknown and retracted slots) is maintained by a program, and is
given to the reader next to the raw history.

Work in progress. Docs (`docs/`) are internal research notes, mostly in Chinese.

## Layout
- `mem/` — method code. `stateq.py`: MEME pipeline (extraction -> state engine -> reader; official MEME judge adapter).
  `uistore.py`: closed-world observation store for LongMemEval-V2 web-agent trajectories (program-built, no LLM).
  `llm.py`: model calls with a disk cache.
- `runs/` — run scripts and result files (answers, judged outputs). `runs/lmev2/run_lme.py`: LongMemEval-V2 runner
  using the official prompts and scoring functions.
- `docs/` — plans, progress log, reviews. `paper/` — draft.

## Data (not included)
- MEME: https://huggingface.co/datasets/meme-benchmark/MEME (put `meme_nofiller.json`, `meme_filler32k.json` in `runs/meme/data/` as `nofiller.json`, `filler32k.json`).
- LongMemEval-V2: https://huggingface.co/datasets/xiaowu0162/longmemeval-v2 (into `runs/lmev2/data/`); evaluation code from https://github.com/xiaowu0162/LongMemEval-V2 (into `runs/lmev2/repo/`).
- BEAM / LongMemEval: see `docs/SURVEY.md`.
