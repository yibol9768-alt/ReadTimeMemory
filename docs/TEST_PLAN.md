# 测试集计划（2026-10-20 写；方法冻结后、Pro 额度重置后执行，约 10-27 02:45 ET 之后）

## 冻结的方法与对照
- 方法：note3_deep~m0（BEAM，tuned 提示）/ note3_deep（LME，极简提示）。笔记 5 部分保持原样（消融显示“未找到”无作用，但测试前不改方法）。
- 对照：hindsight~m0、cf_hindsight~m0（仅点状题，路由器同开发集 runs/h35c_route.py）；LME 对照 lychee、hindsight。
- 读者：gpt-6-luna，3 次生成；判分 gpt-6-luna（官方提示）。强读者（DeepSeek 2 并发）视余量补 1 次。

## 测试集（事先固定，不看内容）
- BEAM-100K：开发集挑选的补集，10 段 = 1,4,5,6,7,8,12,13,15,19（200 题）。
- BEAM-1M：HF Mohammadta/BEAM 的 1M 共 35 段，开发 = 0–4；**测试固定为第 5–14 段（10 段、200 题）**，原因是额度（30 段全建 Hindsight 约需数千万 token 的 luna 提取）。剩余 15–34 不使用。
- LongMemEval-S：lme_split 的 test，151 题（与开发集 62 题无重叠；论文写 151）。

## 步骤（V3_TEST=1，输出 runs/{beam,beam1m,lme}_test/）
1. 下载 BEAM-1M 第 5–14 行 → data/beam/1M_test_rows.json（只存，不看问题）。
2. 建我们的记录（luna）：`V3_TEST=1 V3_BEAM_SPLIT=<s> python3 -m mem.beam build`（LME 每题一个库）。
3. 建 Hindsight 库：AMB hindsight-http，100K 测试单元、1M 第 5–14 行、LME 151 题；需重启 Hindsight 服务（8888），3 路并行（内存）。收集为 ext/hindsight/ctx.jsonl。
4. 路由：`V3_TEST=1 python3 runs/h35c_route.py`。
5. 回答 3 次 + 判分（同开发集脚本，加 V3_TEST=1）。
6. 分析：同 runs/cmp_main.py / lme_cmp.py，结果整体写进论文 sec:test，不再改方法。

## 额度估计（粗）
- 最贵的是第 3 步 1M 部分（10 段 × 约 1M token 的 luna 提取）和第 2 步 1M 记录。开跑前先用 1 段测一次额度消耗，再决定 1M 是否缩到 5 段。
