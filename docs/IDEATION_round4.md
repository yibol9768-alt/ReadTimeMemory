# 构思第 4 轮（2026-10-23）

## 可认证的否定（候选 Y）——偏窄，降为算子
- 查新：带保证的弃答/事实性（Yadkori 2405.01563、Mohri & Hashimoto 2402.10978、C-RAG、TRAQ、Conformal-RAG）都不在记忆场景、不估计漏抽率；2606.30247 已证明只看观测证据无法区分“没观察到但为真”与“为假”（占掉一半理论）；CROWN-QA、MemTrace、MEME 是评测。
- 题量：LME-V2 拒答 128/451；LME-S 30/500；BEAM 10%；LoCoMo adversarial 22.5%（不计分）；CUE-Mem 拒答 30.6%（默认拒答已饱和）。可提升：BEAM +3–5、LME-V2 +5–11。对话中多数事实只说一次，漏抽率估不准，证书难发出。→ 作为候选 X 的否定算子，不单列主线。

## 强读者在 BEAM 上到底丢在哪（已有数据，不花钱）
- 测试集 note3_deep 按能力的丢分：1M 上 luna/sol/astra 丢分最多的都是**时间推理（6.0–6.5 分）**，其次拒答（4.0–4.7）。
- 原因核查：BEAM-1M 每批对话只有首轮一个 time_anchor（每两周一个）；金标日期（如 1 月 18 日、2 月 3 日）**在对话原文中不存在**，来自隐藏的 conversation_plan（金标里写“inferred from session timing / batch anchor”）。→ 1M 时间推理题大量不可从对话推出，是基准标注噪声；强读者在 BEAM 上的真实可修空间比表面小。

## 新线索：依赖/级联推理（MEME）
- MEME（2605.12477）：Cascade 平均 3%、Absence 1%；提示优化、更深检索、更强 LLM 均无效，仅 Opus 4.7 文件智能体以约 70 倍成本部分解决；无方法。相关：MAB FactConsolidation 多跳公开最好约 7%（CAR 30–51%），Deleted Is Not Forgotten（93% 复活），STALE 2605.06527。
- 撞车：PACMI 2610.05732（10-05）出处图 + 依赖边 + 四态有效性 + 级联传播，但仅自建 100 例，级联消融不显著，未测公开基准；Kumiho 2603.17244（AGM 信念修正图记忆）。
- 正在：独立评分 + MEME 数据/题型核查。

## 依赖感知记忆（TMS-Memory）独立评分：上限约 6，改了再做
- PACMI 全文：依赖边由基准人工标注（自动抽取列为未来工作），仅两跳固定拓扑自建 100 例，级联消融 p=0.057，同家族读者/判分，无代码。另有 2608.10502（出处图回滚修复）、MoM/P-Mem 2609.25054、CUPMem（STALE 原型）、A-TMA 2607.01935、NeurIPS25 Truth-Maintained Memory Agent。
- 基准：MEME 级联规则是原文明说的（“组长换了，周报改发 James”），Opus 文件智能体正是记下“条件条目+父节点”才有效；FC-MH 不是依赖传播问题（先选当前版本再串联，CAR 已覆盖）；STALE 是隐式失效，依据需 LLM 推断。
- 打分：新颖性 4.5、单一方法 6、MEME 提升可信 7、FC-MH 3、STALE 4、BEAM/LME 2、强读者有效 7、可行 8、上限约 6。
- 到 7–8 需：从原文自动抽依据（报召回/精度）+ 定理；MEME 全量 Cascade/Absence ≥0.6 且成本 ≤3 倍、不掉其他题型；STALE 比 CUPMem +10；自动边打平 PACMI 标注边；BEAM/LME 不掉；三读者一致。
- 第一个试验：MEME 20 episode，DeepSeek-Flash 抽“实体值+条件规则+父节点”，程序 JTMS 传播；门槛 Cascade ≥0.5 且 Absence ≥0.5、成本 ≤3 倍、其他题型掉分 ≤3、依据召回 ≥80%；过了再测 STALE 100 题（比 CUPMem <+8 就停）。

## MEME 核查
- 数据/代码已公开（HF meme-benchmark/MEME：nofiller/32k/128k；GitHub SeokwonJung-Jay/MEME-public）；100 episode、694 题（Cas 164、Abs 130、Tr/Del/Agg/ER 各 100）；GPT-4o 判分（κ=0.965）。
- 分数（gpt-4.1-mini；ER/Agg/Tr/Del/Cas/Abs）：MD-flat .94/.45/.77/.25/.06/.05；Mem0 .67/.35/.43/.21/.03/.00；全上下文 Sonnet 4.6 Cas/Abs .05/.35；只喂相关事实上限 Opus 4.7 .93/.72。Opus MD-flat：Cas .32、Abs .59（成本约 70 倍，ER 掉到 .60、Tr .20）。
- 失败：规则与变化都已存下，Cas 一半没取到变化事件一半回答失败；Abs 83–86% 两条证据都在仍答旧值。
- 294 道 Cas/Abs 中 291 道依赖由固定句式明说 → 依赖感知记忆基本可直接解决，但会被说成贴合基准写法。FC-MH 不是依赖传播；STALE Type II 依赖需常识推断，CUPMem（68.0）已占写时传播。

## 候选 Z（统一）：记忆的封闭世界状态查询
共同规律：答案取决于记忆“整体状态”的题——全部/计数/顺序/最新、一个都没有/前提不成立、某事变了之后的当前状态——相似度取前 k + 读者阅读原理上答不了，读者再强也无效（MEME 换更强 LLM 无效；LME-V2；OAKS；BEAM 穷举型题）。
方法：维护一个只追加的观察库，在其上提供三类由程序精确执行的状态算子：(1) 集合算子（列举/计数/最新/之前/排序）；(2) 可认证否定（覆盖充分才答“没有/前提不成立”，否则回退阅读或弃答）；(3) 依赖维护（依据变化时级联更新或标为待确认）。读者只负责把问题翻译成状态查询并读结果。
评测：LME-V2（列举+拒答）、OAKS（追踪/计数）、MEME（Cas/Abs/Agg/Tr）、BEAM 穷举型与拒答题；读者含强读者；对照 grep 智能体、便宜模型全库扫描、Hindsight、Mem0/MD-flat、PACMI/CUPMem 式做法。

## 候选 Z 反方（上限 5.5–6）
- 拼装：集合算子（ChatDB、xmemory、Lemmalog、UltraMem、TAG/LOTUS）、否定理论（2606.30247）/评测（CROWN-QA）、依赖维护（PACMI、CUPMem、Kumiho）。四个基准各有取巧空间（MEME 固定句式、OAKS-B 模板、LME-V2 网页框架、BEAM 时间题金标不在原文）。漏抽 → 自信的错误“没有”。
- 能推到 7–8 的唯一结果：同一套不针对基准的通用流程，在开放对话、抽取会漏（删 30% 原文也成立）时误否定率实测在理论界内；且在 LME-V2 拒答+前提、BEAM 拒答、改写句式后的 MEME 上，以 ≤1/5 耗时赢 grep/Codex 智能体 ≥5 分，三个读者一致。

## 候选 Z 严格评审（6–6.5，改了再做）
- 近邻补充：TallyMem 2610.19474（事件账本 + 确定性代码执行，MINTEval +12.8–13.9，三读者一致）；StateMem/StateMemBench 2608.19652（DeepSeek-V4-Flash 0.205→0.363）。
- 机制主张应改为“检索接口的问题”而非“读者的问题”（强智能体 + grep 能解 LME-V2）。
- 到 7–8：统一成“三值（真/假/未知）封闭世界查询语言”（带聚合的 Datalog + 分层否定；否定只在覆盖证书成立时输出假，否则未知；级联 = 规则重新求值带出处）；分离定理（依赖完整支撑集的查询，任何 k < 支撑集大小的前 k 读取存在反例；漏抽率 ε 下误否定率 ≤ δ）；同一抽取提示不按基准定制。
- 小试 1：MEME 20 episode + OAKS-B 一篇，共用抽取器与查询引擎；门槛 Cas ≥0.5 且 Abs ≥0.5、成本 ≤3 倍、其他题型掉分 ≤3、规则召回 ≥80%；OAKS 子集比 RAG 32.4 高 ≥20；需按基准改提示才能过就停。
- 小试 2：LME-V2 Small 拒答 128 + 前提约 49；抽取召回 ≥90%（<85% 停）、证书“不存在”精确率 ≥95% 且覆盖 ≥60%、比 DeepSeek grep 智能体 ≥+5 且耗时 ≤1/5。

## 第 4 轮结论
想法层面的上限稳定在 6–6.5；所有评审都指向“靠结果和形式化推到 7–8”。下一步：用免费算力跑小试 1、2（门槛如上），过了再做形式化。

## 小试 1 开始（2026-10-23 04:50 ET）
- 代码 mem/stateq.py：通用抽取（OBS/RULE/RETRACT，提示不含 MEME 句式）→ 程序状态引擎（值历史、状态 KNOWN/UNKNOWN/RETRACTED、规则级联）→ 读者映射槽位并按状态作答；判分用官方 MEME judge.py（含“变化前也要答对”的过滤），判分模型经 llm.call。
- 开发调试只用 pl_001、sw_001（两处通用修正：抽取时给出已有槽位值与规则、变化写回原槽位；规则值只存新值本身）。开发集上 Cas/Abs/Del/Tr/ER 全对。
- 正式：20 个留出 episode（pl_002–011、sw_002–011，nofiller），读者 luna（Plus），state vs full；先 luna 判分，H20 窗口后加 Qwen 判分与 DeepSeek 读者。runs/meme_pilot.sh，日志 runs/meme/pilot1.log。

## 小试 1 初步信号（04:35 ET；luna 读者 + luna 判分，仅 11/20 个留出 episode 判完，未判 full 对照）
- 状态查询：Cas 12/16（0.75）、Abs 12/13（0.92）、Del 9/11、Tr 9/11、ER 10/11、Agg 8/11（Cas/Abs/Del 为“变化前也答对”的 real pass）。对比论文：最好系统 Cas .06 / Abs .05，Opus 文件智能体（约 70 倍成本）.32/.59。已过门槛（Cas、Abs ≥0.5），但读者=判分者、无同读者全上下文对照、样本小。
- 用户 04:35：先不用 luna。已停 luna 判分。runs/meme_pilot_h20.sh（PID 263127）11:05 ET 起：Qwen3.8 判 luna 两组答案；DeepSeek-V4.1-Flash（v9）作读者跑 state/full，Qwen 判分。
