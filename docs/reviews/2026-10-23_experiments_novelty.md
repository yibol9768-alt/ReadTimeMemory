# 模拟审稿：实验充分性 + 方法新颖性（2026-10-23 01:40 ET）

四位审稿人独立评审（主编、方法学、反方、领域/新颖性），只读论文与 PROGRESS 10-23 01:20 条。约束：不再花 GPT 的钱，免费算力为 H20 上的 DeepSeek-V4.1-Flash 与 Qwen3.8-27B；Pro 10-29 重置可做少量 luna 判分。

## 总体判断
- 主编：实验强度 borderline 偏下（约 ICML 4/10，weak reject）。强项：受控协议、冻结后的测试集、4 读者扫描、组件级消融、读时 token 少于 HS+CC。
- 新颖性（领域审稿人）：单看方法不够主会；最可能被概括为 "LycheeMemory + structured Chain-of-Note" 或 "training-free DeferMem/LazyMem"。能撑住论文的是实证部分（受控审计、1M 规模、跨读者）。
- 四人一致的最大风险：唯一显著增益在 luna 上，而 luna 同时建库、作答、判分（判分自偏好）；测试集上 sol/astra 持平或略低。

## 补实验清单（按优先级；全部可在 H20 免费完成，除注明外）
### P0
1. 换判分者重判全部已有答案（开发+测试；luna/sol/astra 读者；所有系统）：DeepSeek 与 Qwen 各一遍。报告一致率（κ）、各判分者下 Δ 是否保持、差值对答案长度的回归。10-29 后用 sol 抽 100–200 条复核；可请 3 名博士生人工标 100 条。【四人都提】
2. Qwen3.8-27B 作弱读者（计划、编译、CC、作答全用 Qwen），BEAM-100K/1M 测试集 3 次，开源判分。用来把"弱读者受益"变成一条梯度（Qwen < luna < DeepSeek < sol/astra）。DeepSeek 测试集作答已排在 10-23 11:05，判分改用 Qwen，不必等 Pro。【主编、反方】
3. 同等调用的通用两步对照：同一份记录、同一次调用、同推理强度，把五段笔记换成自由形式"提炼相关证据"（DeferMem/LazyMem 式）；以及给 Hindsight 加同等输出预算的通用整理调用。拆开"多一次深思调用 / 五段结构 / 账本存储"。【主编、反方、领域】
4. 写时 vs 读时同操作对照（新颖性最关键）：同一编译提示在写时按主题/实体预生成与问题无关的笔记，对比读时按问题生成，token 配平。【领域】
### P1
5. 等长对照：Hindsight 上下文截到 60K 再编译（或账本补到 124K），全题、DeepSeek 读者。【方法学、反方】
6. 基线对等：记录检索 60K + CC；Hindsight 原生读者提示；Hindsight 自带 reflect 操作。【方法学、反方、领域】
7. 更多写时系统：Mem0、A-Mem（开源）用 DeepSeek 建库和作答，单独出"DeepSeek 全流程"表；有代码则加 MemoryLACE/LazyMem。Zep 闭源，说明即可。【三人提】
8. 用 DeepSeek 重建记录库和 Hindsight 库（BEAM-100K 测试或开发），既测"建库者是 luna"的依赖，也估计建库方差（现在只重复了读时调用）。【方法学、主编】
9. 10-29 后：补完记录检索+CC 1M 测试集另外 2 次、100K 第 3 次剩余 37 题判分；DeepSeek 测试作答的 luna 判分。【Pro 少量】
### P2
10. LoCoMo（真实对话）用开源读者+开源判分跑一遍，回应"BEAM 是合成数据"。
11. 笔记忠实度：DeepSeek 逐条核对笔记中的"变化/冲突"是否虚构。
12. 编译者×读者交叉（Qwen 编译 DeepSeek 作答及反之）。
13. noteparts 补到 3 次（DeepSeek/Qwen）；嵌入模型与分段长度敏感性。
14. 写时成本与延迟实测；输出/推理 token 计入成本。
15. 统计：按对话分层置换检验、5 段对话用 BCa 区间、事先指定主假设+多重比较校正、BEAM/LME 重采样单位说明（不用新跑）。

## 新颖性：最接近的工作（已核实存在）
- DeferMem https://arxiv.org/abs/2605.22411 （已引）：查询时 RL 训练的证据蒸馏。
- LazyMem https://arxiv.org/abs/2607.22690 （2026-07，未引）：写时不构建，查询时广泛检索 + SFT/RL 训练的小模型按问题压缩。标题即 "Retrieve Broadly, Construct Selectively"。
- MemoryLACE https://arxiv.org/abs/2609.03201 （2026-09，未引）：原子记忆+出处，按当前/历史/支持/冲突组装证据单元；也跑 BEAM、也比 Hindsight。
- Theanine https://arxiv.org/abs/2406.10996 （NAACL'25，未引）：不删旧记忆、时间线、回答时提炼。
- Hindsight reflect https://arxiv.org/abs/2512.12818 ：查询时推理操作，我们未对比。
- 组件评级：原子账本=已有；读时编译笔记=部分新（新在五段结构尤其"长期指令"一段，以及笔记+记录同给读者）；路由=不新（且已去掉）；CC=不新；"写时不整合、读时整合"立场=不新。
- 其他漏引（领域审稿人给出，未逐篇核实）：TRACE 2607.00339、EGMemory 2609.23465、Scroll 2608.21690、RecMem 2605.16045、REALM 2609.16053、MemDream 2609.34545、Useful Memories Become Faulty 2605.12978、TriMem 2605.19952、SeCom 2502.05589、TReMu 2502.01630、MemoRAG 2409.05591。

## 文字层面（作者处理，列出备查）
- 摘要/tab:main 仍是"仅开发集、每格最高"，测试集 sol/astra 结果未写入正文。
- tab:test 表注按"路由路径"拆点状/面状，与方法节"已去掉路由"矛盾。
- DeepSeek 开发集 1017 权重、测试集 v9 权重；astra 只跑 1 次；LME 两段被内容过滤。
- 反方认为只能靠表述处理的：所有库都由 luna 建；"弱读者"事后定义；读时 vs 写时是整系统对比；每题重编译的部署成本。
