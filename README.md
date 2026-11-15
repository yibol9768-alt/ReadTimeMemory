<p align="center"><img src=".github/assets/repository-cover.jpg" alt="ReadTimeMemory" width="100%"></p>

# ReadTimeMemory

**Long-term memory, evidence retrieval and reranking for AI agents.**

Former repository name: `stateful-memory`. **ReadTimeMemory and stateful-memory are one project.**

## What is here

- **Stateful memory:** append-only observations, value histories, dependency cascades, unknown values and explicit retractions.
- **Evidence retrieval:** memory selection, query-time reconstruction and benchmark adapters.
- **Reranking experiments:** synthetic-world construction, set-conditioned selector training and inference.

## Code map

| Path | Purpose |
| --- | --- |
| [`mem/`](mem/) | Memory representations, retrieval and evaluation modules |
| [`mem/stateq.py`](mem/stateq.py) | Program-maintained state over an observation history |
| [`mem/uistore.py`](mem/uistore.py) | Closed-world observation-store implementation |
| [`runs/sel/train.py`](runs/sel/train.py) | Selector training with group-wise ranking objectives |
| [`runs/sel/infer.py`](runs/sel/infer.py) | Reranking and evidence selection |
| [`runs/sel/world.py`](runs/sel/world.py) | Synthetic-world generation and evidence interventions |
| [`runs/lmev2/run_lme.py`](runs/lmev2/run_lme.py) | LongMemEval-V2 experiment runner |
| [`results/`](results/) | Small recorded development-result snapshots |

## Recorded results

[Development recall snapshot · 2026-11-02](results/development-recall-20261102.csv): E0’s mean recall across the 1k/2k/4k/8k/16k evidence budgets was 90.8 on LongMemEval, 87.8 on EpBench, 72.6 on EverMem, 93.7 on LoCoMo and 39.5 on BEAM. These are percentages from an earlier development-stage record, not final-test claims or proof of general superiority.

**Limits:** BEAM and LongMemEval development labels informed recipe selection; LoCoMo covers only five development conversations; the recorded analysis did not establish the project’s frozen success criterion. Do not interpret a non-significant difference as equivalence. The CSV is a transcribed summary, not raw per-question predictions or an independently rerun result.

## Getting oriented

Start with the module docstrings and each script’s argument parser. The memory modules and experiment runners have different dependencies; training runners require PyTorch/Transformers, compatible model weights and benchmark data. Some research runners still use environment-specific paths and are not a packaged drop-in SDK. Supply API credentials through your own local environment; no credentials are distributed here.

## Scope and status

This is a research-code snapshot. Model weights, raw datasets, private operational notes and manuscript files are not included. Recorded results are provided as artifacts, not a claim of independent reproduction or production readiness.

Project artwork was generated with AI for repository presentation.
