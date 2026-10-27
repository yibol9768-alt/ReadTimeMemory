# V3 调研：非参数化记忆（2026-10-13）

V3 的起点是哈工大（深圳）李东方老师团队（Lychee Team，合作者户保田、张民）的 agent 记忆工作。这一页记三件事：他们做了什么，这个领域现在是什么状态，我们可以从哪里接着做。

标了“未核实”的是只在搜索摘要里看到、没读原文的内容。

## 1. 李东方团队的工作

### 1.1 非参数化记忆（不训练，只调 API）

| 论文 | 时间 | 核心做法 | 模型 | 结果 |
|---|---|---|---|---|
| LycheeMemory V2（arXiv 2608.12990，李东方共同一作） | 2026-08 | 按语义边界把多轮对话攒成一段，一段只调一次 LLM 整理 | GPT-4.1-Mini、GPT-4o-Mini；text-embedding-3-small；bge-reranker-v2-m3 | LoCoMo 89.22%，LongMemEval-S 92.20% |
| SEEM（arXiv 2601.06411，二作） | 2026-01 | 图记忆层存关系事实，情景事件帧存叙事，都带指回原文的指针 | Qwen3-Next-80B、GPT-OSS-120B、Pangu-7B | LoCoMo F1 61.1，LongMemEval 65.0% |

**LycheeMemory V2 的流程**

1. 切段：每轮对话只算 embedding，用语义突变、内聚度下降、长度压力算边界概率，超过 0.50 就结束一段。不调 LLM。
2. 编码：一段结束调一次 LLM，产出带类型、实体、主题、时间范围、来源链接的独立记录，另带一个消歧状态传给下一段。
3. 索引：记录写入向量库（LanceDB）和结构化库（SQLite FTS5），建实体、主题、时间、事件帧节点。不调 LLM。
4. 查询：调一次 LLM 做查询规划，四路召回（记录向量、证据节点、时间过滤、原始对话），融合、重排、去冗余后生成答案。

**LycheeMemory V2 的消融（LoCoMo，完整版 89.22%）**

| 改动 | 准确率 |
|---|---|
| 每轮都整理（eager） | 81.88% |
| 固定窗口切段 | 82.40% |
| 记录换成摘要 | 80.78% |
| 只用记录向量召回 | 81.75% |
| 去掉融合、重排、去冗余 | 66.62% |

成本：每个对话的构建 token 是 204.1K（LoCoMo）和 304.7K（LongMemEval-S），A-Mem 是 1459.9K 和 1264.3K。

**SEEM 的流程**

1. 每段原文抽成情景事件帧（参与者、动作、时间、地点、原因、方式）。
2. 新帧和最相似的旧帧比较，LLM 判断是否同一事件，是就合并，并合并来源指针。
3. 同时抽（主、谓、宾、时间）四元组进图，相近节点合并。
4. 检索时从图里排出原文，再沿事件帧的指针把相关原文都拉回来（反向溯源扩展）。

### 1.2 参数化 / 隐式记忆（要训练）

| 论文 | 时间 | 做法 |
|---|---|---|
| CMT（AAAI 2025，一作，arXiv 2412.07393） | 2024-12 | 新文档压缩进记忆库，主模型不动，压缩模块要训练 |
| 压缩记忆 + 端到端强化学习（arXiv 2602.08382，二作） | 2026-02 | 分块压缩成记忆表示，门控选择后迭代推理，压缩器和推理器联合训练。搜索结果称它是 ACL 2026 的 LycheeMemory V1（未核实） |
| LycheeDecode（ICLR 2026）、LycheeCluster | 2026-02、03 | KV cache 的长上下文推理加速，和 agent 记忆关系较远 |

### 1.3 他们自己承认的缺口

LycheeMemory V2：
- 偏好类问题弱：LongMemEval-S 上 90.00%，MemoryOS 是 100%；换 GPT-4o-Mini 掉到 70.00%。
- 冲突的记录不在存储时解决，留给检索和回答。
- 只测了纯文本对话，没测延迟、存储增长、线上反馈。
- 结果依赖托管 API，换开源模型要重测。

SEEM：
- 抽取和融合出错会永久污染记忆库。
- 预定义的事件槽位会漏掉不符合框架的信息。
- 抽取和融合的 LLM 调用多，成本高。

### 1.4 代码

两篇论文都没给代码链接。GitHub 上 `owoakuma/LycheeMemory` 存在（4 星，2026-03 后无更新，内容未看）；`LycheeMem/LycheeMem` 查不到。复现要按论文自己写。

## 2. 领域现状

### 2.1 对话记忆的榜

LycheeMemory V2 对比的基线：Full Context、Naive RAG、Mem0、A-Mem、MemoryOS、MemOS、Nemori、LightMem、TiMem、MemU。LoCoMo 上 Full Context 就有 84.80%，最好的系统接近 90%。

对这两个榜的质疑（都是二手转述，未核实原始出处）：
- LoCoMo 答案约 6.4% 有错；LLM 判分器接受了约 63% 故意写错的答案。
- 同一个系统换评测脚本，LoCoMo 分数从 38% 到 92%。
- 题目和原文措辞高度重合，靠匹配问法就能过。
- LoCoMo-Plus（arXiv 2602.10715）指出现有榜主要测明说的事实，不测隐含的约束。
- BEAM（ICLR 2026）更难，1000 万 token 档最好成绩 68.0%。

### 2.2 对话问答之外的记忆评测

- Proced-Mem（ICLR 2026）：ALFWorld 和 OSWorld 上的程序性记忆检索，换到新情境 MAP 掉 30–42%。
- EvoMemBench（arXiv 2605.18421）：按记忆范围（单回合内 / 跨回合）和内容（知识 / 执行）分类，比了 15 种记忆方法和长上下文基线。**和我们最可能撞车，要读全文。**
- V2 的 `POSITIONING.md` 里已经读过的：How Memory Management Impacts LLM Agents（ACL 2026）、Belief Memory、Memory Trust Gap、Ground Truth First、Handoff Debt。

## 3. 和 V2 的关系

V2 研究的是交接：模型玩交互世界，把学到的规则写成纸条传给下一代。V2 的纸条就是一种非参数化记忆，两边的发现能对上。

| V2 的发现（`THEORY.md` v0.2） | LycheeMemory V2 / SEEM 的对应 |
|---|---|
| 损失最大的一段是反复转述：写的人看不到前几代的原始记录 | 每轮都整理比攒成一段再整理低 7 个点 |
| 写的人看得到全部记录时，300 字符就能传下大部分 | 记录换成摘要掉 8 个点；记录要带来源链接 |
| 纸条放长能补回损失，是因为顺带带下了更多原始观察 | SEEM 的反向溯源扩展：检索后把原文拉回来 |
| 写错的规则很少被纠正 | SEEM 承认抽取错误会永久污染记忆库 |

两边的差别：

| | 现有记忆系统 | V2 |
|---|---|---|
| 记忆的内容 | 对话里别人说过的事实 | 模型自己从交互里归纳的规则 |
| 写入时内容是否确定 | 确定，原文里有 | 不确定，可能还没看懂 |
| 评测 | 问答，LLM 判分 | 世界里的实际得分，加逐条 fact 的真值 |
| 解释 | 消融表 | 三段分解（换人、一次压缩、反复转述） |

他们的结论“整理的粒度决定准确率和成本”只有消融表支持，没有解释。V2 的假说给出了一个解释：归纳需要把多段原始记录并排放在一起，整理得太早太碎，就拿不到并排的原始记录。

## 4. 候选方向

### A. 把非参数化记忆系统放进 V2 的交互世界（推荐）

**问题**：现有记忆系统是为“记住别人说过的事实”设计的。当要记的是自己归纳的、还没定论的规则时，它们还管用吗？

**做法**：在 V2 的世界上，把纸条换成记忆系统（Mem0、A-Mem、LycheeMemory V2 式的分段记录、SEEM 式的带溯源指针），和已有的五种玩法并排比。

**V2 假说给出的可否定预测**：
- 每条 life 结束就整理成记录、之后只读记录的系统，成绩接近普通 relay，因为跨 life 的归纳仍然拿不到原始记录。
- 能沿指针取回原始记录的系统，成绩接近“完整记录交接”。
- 所以决定成绩的是读的时候拿不拿得到原始记录，而不是记录的结构做得多精细。

**为什么值得做**：
- 现成资产：56 个带逐条 fact 真值的世界、五种玩法的基线、三段分解的方法，都已经跑过。
- 只用 API，记忆系统本身不用训练。
- 避开 LoCoMo 这个已经刷满、判分有争议的榜。
- 给 V2 的论文补一节“现有记忆方法救不救得回交接损失”，也可以独立成文。

**风险**：EvoMemBench 可能已经做了跨回合的执行类记忆对比，要先读全文确认差异。我们的差异点预计在逐条真值和三段分解上。

### B. 整理时机的受控实验

把 A 里的假说拿回对话数据上验证：在 LoCoMo / LongMemEval-S 上只改两个变量，整理的时机，以及整理时能看到多少原始对话，看 V2 的三段分解在对话记忆上是否同样成立。可以作为 A 的第二个数据集，不单独立项。

### C. 记错了怎么改

SEEM 承认抽取错误会永久留在库里，LycheeMemory V2 不处理冲突记录，V2 也看到写错的规则很少被纠正。可以做带可信度、可撤回的记忆记录。竞争多（Belief Memory、Memory Trust Gap 已在做），建议等 A 出结果、确认错误确实是主要损失来源之后再定。

### D. 在 LoCoMo / LongMemEval 上复现并超过他们（不推荐）

榜接近饱和，判分有争议，要复现十个基线，A-Mem 一类每个对话要一百多万 token，吃额度多，新意少。

## 5. 建议的第一步

用最小代价检验方向 A 的预测，再决定要不要铺开。

1. 读 EvoMemBench 全文，确认没有撞车。
2. 自己写两个最小的记忆系统，各几百行：
   - 只存记录：每条 life 结束整理成带类型的记录，下一代只能检索记录。
   - 记录加溯源：同上，但检索到记录后能取回对应的原始屏幕和操作。
3. 在 V2 的 10 个高损失世界上跑 Sol 和 Astra，和已有的 relay、压缩、完整记录交接三个数并排。
4. 看结果是否符合 4.A 的预测。符合就接入 Mem0、A-Mem 等现成系统并扩到全部世界；不符合就修改 `THEORY.md`。

额度：10 个世界、两个模型、两种系统，和 V2 的一次压缩条件实验是同一个量级，在 Pro 额度留 10% 的约束之内。

## 新颖性核查：“账本 + 派生视图”（2026-10-14）

| 我们的组成部分 | 已有工作 | 结论 |
|---|---|---|
| 原始记录只追加、带溯源、冲突时作废不删 | Zep/Graphiti（arXiv 2501.13956）：原始 episode 作为真值，事实边带双时间戳，作废而不删除，可追溯到来源；SEEM：记录带指回原文的指针 | 不新 |
| 摘要/视图从源头重新生成并带引用 | GraphRAG 社区摘要、RAPTOR 树摘要都从源文本生成；SSGM（arXiv 2603.11768，2026-03）提出不可变日志 + 周期性“对账”可回滚，但只是概念框架，没有实验，也没说是重建还是打补丁 | 不新 |
| “反复改写会让记忆变坏” | Zhang 等（arXiv 2605.12978，2026-05/08）：在 ARC-AGI 上，LLM 持续整合的经验记忆先升后降，可低于不用记忆；只保留原始轨迹的对照组不差。对象是过程性经验，不是对话事实；没有做“从原始记录重建”这一臂 | 论点已被提出，但对话事实记忆上的受控实验暂未找到 |
| 题型分流（视图 vs 检索） | 各系统普遍有类似的多路检索 | 不新 |

结论：整套设计是已有部件的组合，不能当方法创新卖。可能还空着的点：在对话事实记忆（LoCoMo/LongMemEval）上，同等字数预算下，受控比较“在旧记忆上改写”和“从账本重建”，并画出早期事实随会话推进的遗忘曲线；再拿同一把尺子去量现有系统（Mem0 的 UPDATE/DELETE、A-Mem 的记忆演化、滚动摘要）。这更像测量/分析型贡献，要等漂移实验结果再定。检索只做了两轮，可能有漏。

## 新方向调研（2026-10-14）

已经挤满、不建议做：记忆投毒/溯源洗白（Memory Provenance Laundering 2607.29167、MemoryGraft、MemSecBench 等）；自进化记忆架构（MemEvolve ICML'26、EvolveMem 2605.13941、Evo-Memory）；检索驱动的再巩固（REALM 2609.16053，LoCoMo 75.97%）；记忆的删除/替换操作评测（ForgetEval 2606.15903）；遗忘曲线式衰减（FSFM、FadeMem、MemoryBank）。

可能还空着（两轮检索内没找到同类）：
1. 记忆自测复习（测试效应）：定期从账本出题考压缩记忆，答不上的事实优先写回。心理学里“自己出题”的效果弱于“别人出题”（Myers 等 2023），LLM 是否一样可以顺带检验。
2. 写读算力分配：强模型写、弱模型读 vs 反过来。Eywa（2605.30771）分别换了写和答的模型但没有隔离这个对比；Harness the Memory（2608.15008）指出 LLM 开销只有用在检索瓶颈上才有回报。
3. 无题评测：让模型只凭记忆复原对话里的事实，和账本逐条比对，得到每条事实的存活情况，不依赖 benchmark 的题目数量。

### 本轮调研的出处

新颖性核查与新方向：
- Zep: Temporal Knowledge Graph for Agent Memory — https://arxiv.org/abs/2501.13956 ；Graphiti — https://help.getzep.com/graphiti/getting-started/overview
- SSGM（不可变日志 + 对账，纯概念框架，无实验）— https://arxiv.org/abs/2603.11768
- Useful Memories Become Faulty When Continuously Updated by LLMs（ARC-AGI，持续整合后记忆先升后降，仅保留原始轨迹的对照不差）— https://arxiv.org/abs/2605.12978
- Memory for Autonomous LLM Agents（综述，提到反复压缩会丢低频细节）— https://arxiv.org/abs/2603.07670
- ProMem（抽取后对照原对话迭代核验）— https://arxiv.org/abs/2601.04463
- RecMem（ACL Findings 2026，推迟整合，LoCoMo/LongMemEval-S）— https://arxiv.org/abs/2605.16045
- Memory Provenance Laundering — https://arxiv.org/abs/2607.29167 ；MemoryGraft — https://arxiv.org/abs/2512.16962 ；MemSecBench — https://arxiv.org/abs/2607.27080 ；Origin-Bound Authority — https://arxiv.org/abs/2606.24322 ；From Untrusted Input to Trusted Memory — https://arxiv.org/abs/2606.04329
- MemEvolve（ICML'26）— https://arxiv.org/abs/2512.18746 ；EvolveMem — https://arxiv.org/abs/2605.13941 ；Evo-Memory — https://arxiv.org/abs/2511.20857 ；EvoMemBench — https://arxiv.org/abs/2605.18421
- REALM（检索驱动再巩固，LoCoMo 75.97%，LongMemEval 65.11%）— https://arxiv.org/abs/2609.16053
- ForgetEval / Control-Plane Placement Shapes Forgetting（13 种配置，删除/替换失败）— https://arxiv.org/abs/2606.15903
- FSFM（选择性遗忘）— https://arxiv.org/abs/2604.20300 ；FOREVER（遗忘曲线式回放，持续学习）— https://arxiv.org/abs/2601.03938
- 自己出题的测试效应弱（Myers, Hausman, Rhodes 2023）— https://hausman.sites.ucsc.edu/files/2023/08/2023-99019-001.pdf
- Eywa（带溯源的长期记忆，换写/答模型，LoCoMo 90.19%）— https://arxiv.org/abs/2605.30771 ；Harness the Memory（记忆底座评测）— https://arxiv.org/abs/2608.15008

## 工具调研：反代模型指纹（2026-10-14）

目的：确认反代背后实际服务的是不是请求的模型。
- 采用：Ikaleio/lm-detector（npm `lmfpd`，Node ≥22）— https://github.com/Ikaleio/lm-detector 。让模型写约 300 个 1–355 的随机整数，按分布在 59 个参考模型（含 gpt-6-astra、gpt-6-luna、gpt-5.6 系列）里给出校准过的概率；每次 3 条请求。实测结果见 PROGRESS.md 2026-10-14。
- 同类：OneTokenIsEnough（单 token 回答分布 + JS 散度，arXiv 2607.10252）— https://github.com/PhaerisWakfu/OneTokenIsEnough ；ToseaAI/llm-fingerprint-detector — https://github.com/ToseaAI/llm-fingerprint-detector ；fpverify（序贯检验，误报 ≤1%）— https://github.com/Mohamed7415/fpverify ；llm-verify（32 条行为探针）— https://github.com/mintesnot-teshome/llm-verify ；KBF（知识边界指纹）— https://arxiv.org/abs/2605.29524
- 教训：上游自报的模型名不可信。

## 2026-10-14 美东晚　换更难的数据集：调研

起因：DeepSeek 在 LoCoMo 80 题上已有 80–85 分，每组只错 12–16 题，新方法的提升在小样本里分不出来。厂商自报的 LoCoMo 分数在 92–96%（有争议），基本到顶。

| 数据集 | 规模 | 测什么 | 难度（已发表数字） | 对我们的成本 | 结论 |
|---|---|---|---|---|---|
| **BEAM**（ICLR 2026，arXiv 2510.27246） | 100 段对话，2000 题；分 100K/500K/1M/10M 四档，100K 档 20 段 | 10 种能力各 2 题/段：弃答、矛盾处理、事件排序、信息抽取、遵循指令、知识更新、多会话推理、遵循偏好、总结、时间推理 | 论文 Table 1：长上下文/RAG/LIGHT 在 100K 档整体只有 0.24–0.36；矛盾处理全线 0.01–0.05，知识更新 0.26–0.41，总结 0.06–0.32。Mem0 自报 BEAM-1M 64.1、10M 48.6 | 100K 档每段 3 个会话约 200 轮、约 50 万字（多是助手长回答），20 题共用一段对话，编码一次答 20 题，便宜 | **选用，作为主数据集** |
| LongMemEval-S（Wu et al. 2024） | 500 题，每题自带约 115K token 的历史（约 50 个会话） | 单会话、多会话、时间、知识更新、弃答 | 厂商自报 82–94%，也在接近顶 | 每题的历史各不相同，抽 60 题也要编码约 3000 个会话，贵 | 暂不用；后期抽少量做交叉验证 |
| LongMemEval-M | 每题约 500 个会话、约 1M token | 同上 | — | 更贵 | 不用 |
| LongMemEval-V2（arXiv 2605.12493） | 451 题，每题最多 500 条网页 agent 轨迹，最多 115M token | 环境经验：界面、状态变化、流程、坑 | 最好 72.5%，最强 RAG 48.5% | 轨迹巨大，和对话记忆不是一类 | 不用（方向不同） |
| MemoryAgentBench（arXiv 2507.05257） | 4 种能力，约 2000 题，103K–1.44M token | 准确检索、测试时学习、长程理解、冲突处理（v3 改叫选择性遗忘） | — | 多数由长文本数据集改造，不是对话 | 不作主数据集，冲突处理部分可参考 |

选 BEAM 的理由：
- 难，分数远没到顶。
- 题型和我们的方向正好对上：多会话推理、总结、事件排序都要“把散落的事实收拢”，对应 WP3 的汇总记录；知识更新和矛盾处理对应漂移、更新问题，对应 WP4/WP5。
- 用户对助手的对话，比 LoCoMo 两人闲聊更接近真实 agent 记忆。
- 开源：数据 CC BY-SA 4.0，代码 MIT，判分提示词公开。

判分：照搬官方代码。逐条要点用官方提示词（`unified_llm_judge_base_prompt`，已原样存到 `mem/beam_judge_prompt.txt`）打 0/0.5/1，按题取平均。事件排序题官方另算 Kendall tau，我们先只报要点分。判分模型用 DeepSeek（官方没写死判分模型），所以只做组内对比，不和论文数字直接比。

小坑：HF 上的分档名是 100K（论文写 128K）；数据用 datasets-server 的 rows 接口取 JSON（本机没有 pyarrow）。

来源：
- [BEAM 论文](https://arxiv.org/abs/2510.27246)、[BEAM 代码](https://github.com/mohammadtavakoli78/BEAM)、[BEAM 数据](https://huggingface.co/datasets/Mohammadta/BEAM)
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493)
- [MemoryAgentBench](https://arxiv.org/abs/2507.05257)、[数据卡](https://huggingface.co/datasets/Robin076/MemoryAgentBench/blob/main/README.md)
- [Mem0：LoCoMo vs LongMemEval vs BEAM](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)（厂商自报，仅作参考）
- [Cognee：AI Memory Benchmarks 2026](https://www.cognee.ai/ai-memory-benchmarks)（厂商）
- [MemDelta：记忆评测里的混杂因素](https://arxiv.org/pdf/2606.29914)（未细读）

### 附：写入时处理矛盾，是否还有空位（快速查新）
BEAM 上所有方法的矛盾处理分都只有 0.01–0.05，看起来空间很大，但这个方向已经有人做：
- [TOKI](https://arxiv.org/pdf/2606.06240)：双时间轴的算子代数，处理持久记忆里的矛盾。
- [CAAP（arXiv 2608.13921）](https://arxiv.org/pdf/2608.13921)：发现模型能察觉冲突却不主动澄清，提出“提交/澄清/核实/推迟/可逆尝试”的策略，主张记忆保留冲突证据。
- Zep/Graphiti 的边失效机制（见前文）。
- 综述 [2603.07670](https://arxiv.org/pdf/2603.07670) 认为“矛盾检查作为质量关口”仍不成熟。

结论：不把矛盾处理作为主方法，只看汇总记录在 BEAM 的矛盾和知识更新题上有没有顺带效果。另外，BEAM 的矛盾题要求回答“你说法前后矛盾，请澄清”，这部分很依赖读者提示词，和记忆本身关系不大。

## 2026-10-15 凌晨美东　查新：“改写记忆会漂移”是否已知
- 定性上已知：SSGM 把反复摘要归为语义漂移、细节逐步流失（[2603.11768](https://arxiv.org/html/2603.11768v1)），并提出“对照事实锚定”；综述 [2603.07670](https://arxiv.org/pdf/2603.07670) 称之为 summarization drift；工程文章也普遍建议保留原始记录、别只靠摘要（[DEV](https://dev.to/gabrielanhaia/agent-memory-in-2026-summarization-vector-recall-or-both-239h)、[TDS](https://towardsdatascience.com/a-practical-guide-to-memory-for-autonomous-llm-agents/)）。
- 还没细读、可能相近：RecMem（ACL Findings 2026，[链接](https://aclanthology.org/2026.findings-acl.1619/)，记忆巩固）、EMBER（[2606.05894](https://arxiv.org/pdf/2606.05894)，按预算保留证据）、Memory-Induced Tool-Drift（[2605.24941](https://arxiv.org/pdf/2605.24941)）。
- 对我们的含义：WP4 的“漂移”本身不能当新发现，只能算受控的定量证据。真正可能新的是 WP5：用测试效应（自测、只补答错的）来挡漂移，并和“重读”、“整本重建”比效果和成本。要写成论文，需要先细读 SSGM 的锚定、RecMem、EMBER，确认没人做过“按自测结果有选择地补写”。

## 2026-10-15 凌晨美东　查新：测试驱动记忆（自测、间隔复习、回归修复）
- **MemDream**（[2609.34545](https://arxiv.org/html/2609.34545)，2026-09-28，UTS）：最接近的工作。
  - 做法：离线“做梦”周期里，Dreamer 给薄弱节点写前瞻性探针问题，Analyst 把检索失败分成埋没、分散、冗余、噪声四类，Consolidator 做抽取、合并、综合或剪枝（用 GRPO 训练）。
  - 结果：LoCoMo F1 55.1（GPT-4.1 读者），MemoryAgentBench 54.2。
  - 关键区别：
    1. 它的探针答案“逐字取自已存的记忆”，所以**发现不了已经丢掉的事实**；我们的探针来自不可改的账本。
    2. 没有间隔复习调度。
    3. 没把改写当作需要回归测试的操作。它自己附录的压力测试显示，连续 20 个周期后，罕见事实保留率和 QA F1 掉了 12–13%，也就是它自己也在漂移。
  - 对我们的意义：这正好是我们的切入口，“只用记忆考自己，查不出遗忘；要用账本出题”。WP5 的设计可以直接回答这个问题。它的四类失败和我们 WP2 的 (b)(c) 几乎一样，说明这个诊断大家都认，但我们能拿出账本驱动的解法。
- **SSGM**（[2603.11768](https://arxiv.org/html/2603.11768v1)）：有不可改账本，用嵌入相似度量漂移、定期“对齐”，但没有探针、没有逐条修复、没有调度，**没有任何实验**。只是概念框架，我们可以引用它作为动机。
- **MemoryBank**：用 Ebbinghaus 曲线让记忆衰减，用途是主动遗忘，不是靠复习保住事实。
- 综述 [2603.07670](https://arxiv.org/pdf/2603.07670) 把“给 Ebbinghaus 加间隔复习”列为未做的方向；工程项目 ZenBrain 用了 FSRS 间隔复习，但没有论文和评测。
- [LECTOR（2508.03275）](https://arxiv.org/pdf/2508.03275)：用 LLM 帮人类做间隔复习，方向相反，不冲突。
- 结论：“账本出题 + 只修复答错的 + 间隔调度 + 改写前后回归测试”的组合目前没人做。必须和 MemDream 正面对比，最好复现它的“从记忆出题”作为消融。
- 待读：RecMem、EMBER、[Retrieval-Driven Memory Reconsolidation（2609.16053）](https://arxiv.org/pdf/2609.16053)（即 REALM，已在拥挤方向里）、[Useful Memories Become Faulty（2605.12978）](https://arxiv.org/pdf/2605.12978)（已记）。

## 2026-10-15 美东凌晨　细读最接近的 6 篇，做成对比表（PLAN_ICML 第 1 周）

| 工作 | 记忆形态 | 不可改原始记录 | 出题/测试 | 定向修补 | 复习调度 | 量化漂移 | 数据 |
|---|---|---|---|---|---|---|---|
| MemDream [2609.34545](https://arxiv.org/html/2609.34545) | 三层图 | 保留对话页，但不当真值 | 有，从记忆出题 | 有（抽取/合并/综合/剪枝，GRPO 训练） | 无 | 附录压力测试：20 轮后罕见事实 −12–13% | LoCoMo、MemoryAgentBench |
| REALM [2609.16053](https://arxiv.org/abs/2609.16053) | 异构认知图 | 未提 | 无，靠检索反馈 | 检索后再巩固 | 无 | 无 | LoCoMo 76.0、LongMemEval 65.1 |
| Useful Memories Become Faulty [2605.12978](https://arxiv.org/abs/2605.12978) | LLM 改写的经验库 | 主张保留原始轨迹 | 无 | 无（建议“控制何时巩固”） | 无 | 有：巩固后 GPT-5.4 在 54% 原本会做的 ARC 题上失败 | ARC-AGI |
| RecMem [2605.16045](https://arxiv.org/pdf/2605.16045) | 潜意识/情景/语义三层 | 潜意识层存原始交互 | 无 | 语义精炼找回细节 | 按“相似交互复现”触发巩固 | 无 | 多个记忆基准，构建成本 −87% |
| EMBER [2606.05894](https://arxiv.org/abs/2606.05894) | 原文证据胶囊 | 无 | 无 | 学习“保留什么” | 无 | 无 | LongMemEval-RR |
| SSGM [2603.11768](https://arxiv.org/html/2603.11768v1) | 治理框架 | 有账本 | 无（嵌入相似度） | 概念上“对齐” | 无 | 无实验 | 无 |
| **我们** | 任意（滚动、记录、Mem0 式） | **有，账本当真值** | **从账本出题** | **只补考砸的** | **Leitner 间隔调度** | **测出遗忘曲线** | LoCoMo、BEAM、LongMemEval |

对我们定位的影响：
- “改写/巩固会坏事”已有强证据（2605.12978 在 ARC 上，MemDream 自己的附录）。我们的现象部分要做的是**对话记忆上的定量曲线**，加上**不可自检**这一点，不再声称“首次发现漂移”。
- 没人把“账本当真值的测试 + 只补失败 + 调度”放在一起，空位还在。
- RecMem 的“按复现触发巩固”是调度的另一种思路，可以作为相关工作和消融的参照（触发时机 vs 我们的测试驱动）。
- 2605.12978 的“只保留原始轨迹就和巩固差不多”是对我们的挑战：要在主表里放“纯原文检索”（我们的 rag），并说明测试驱动的巩固记忆在预算受限时更好。

## 2026-10-15 13:15 美东　“记忆作为账本的索引”和“持续改写会坏”的相关工作
- **Useful Memories Become Faulty When Continuously Updated by LLMs**（Zhang, …, Hao Peng；arXiv 2605.12978，2026-05，v2 2026-08）：<https://arxiv.org/abs/2605.12978>
  - 设定：LLM 把轨迹改写成文本记忆并持续更新（consolidation），对照只保留原始轨迹（episodic）。ARC-AGI 和自建 ARC-AGI Stream。
  - 结论：持续改写的记忆效用先升后降，可低于无记忆；GPT-5.4 用合并后的记忆在 54% 原本能做对的题上失败；不同更新时间表从同样的轨迹得到不同记忆；保留原始轨迹就有竞争力；建议把原始片段当一等证据、对合并加门控。
  - **和我们的关系：直接重叠**。“改写会漂移、原始账本更可靠、合并要有节制”这条主线他们已经在 ARC-AGI 上提出。我们的差别：对话记忆（LoCoMo、BEAM-1M）、固定字数预算、漂移随模型能力/预算/时长的相图、“修补在固定预算下是零和”、时间平均评测。单靠“发现漂移”不够新，必须有自己的方法。
- **LeanMem**（arXiv 2608.03463，2026-08）：<https://arxiv.org/abs/2608.03463>　画像/事件/原始记录三类记忆，只更新事件类，记录不反复合并，按问题需要分配检索预算。LoCoMo、LongMemEval-S，比最强基线高至多 15.1。和“记忆当索引、按需展开原文”重叠。
- **Agent Zero Memory / Provenance-Aware Long-Term Memory**（arXiv 2608.29606）：<https://arxiv.org/html/2608.29606v1>　时间线 + 实体事件图 + 文档记忆，智能体按需打开原始来源。
- **CoMIC**（arXiv 2606.00756）：<https://arxiv.org/pdf/2606.00756>　旧片段压成摘要，按需按索引重新展开。
- **ProMem**（arXiv 2601.04463）：<https://arxiv.org/pdf/2601.04463>　抽取时用自问反馈补回一次性摘要会漏的信息。
- **Adaptive Memory Distillation**（arXiv 2508.03341）：<https://arxiv.org/pdf/2508.03341>　每段存摘要、线索、原文、向量，两种检索模式。
- **Higram**（arXiv 2608.05095）：<https://arxiv.org/pdf/2608.05095>、**MemForest**（arXiv 2605.23986）：<https://www.alphaxiv.org/abs/2605.23986>　局部改写代替整体改写以减少维护成本。
- ReadAgent（gist + 查原文）、MemGPT（分层、换页）是更早的同类。
- 结论：“摘要 + 指向原文 + 按需展开”已经很多人做，不能作为我们的方法卖点。

## 2026-10-15 14:55 美东　分层 / 日志式记忆查新
- 没找到把 LSM 树压缩直接用于对话记忆的工作。相近的：
  - Hierarchical Aggregate Tree（arXiv 2406.06124）：<https://arxiv.org/html/2406.06124v1>　递归聚合对话上下文的树。
  - MOOM（arXiv 2509.11860）：<https://arxiv.org/pdf/2509.11860>　超长角色扮演对话分阶段压缩（关键事件 → 故事主线）。
  - Cloudflare Agents 记忆文档：<https://developers.cloudflare.com/agents/concepts/conversation-state-and-memory/>　宏压缩（汇总旧消息，原文不删）+ 微压缩；新摘要在旧摘要上更新。
  - Microsoft Agent Framework compaction：<https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction>
  - Context Compaction Theory（arXiv 2608.01326）：<https://arxiv.org/html/2608.01326v1>　需要细读，可能有理论重叠。
  - 流式记忆预算（HF 2608.09043）：<https://huggingface.co/papers/2608.09043.md>　在严格预算下比较近因、摘要、检索三类记忆。
  - 博客提醒：反复对摘要再摘要会把小错误固化（<https://aridanemartin.dev/blog/conversation-compaction-techniques/>）。
- 判断：分层摘要是常见工程做法；我们的区别要落在“每次都从账本重压、改写次数有界、预算分配、遗忘曲线解释 + 同成本下显著优于定期重建”。
- 补读 Context Compaction Theory（Tirmazi, Markelon, Bishop, Mitzenmacher；arXiv 2608.01326，2026-08）：<https://arxiv.org/abs/2608.01326>　把单次压缩形式化为“选择”和“生成”两种博弈，最小压缩预算 = 单向通信复杂度；存在生成比选择省预算的查询集。没有讨论多次压缩、时间调度或从日志重压。**可借作我们理论部分的框架**（反复压缩 = 多轮有损信道，从账本重压 = 每次从信源重新编码），不重叠。

## 2026-10-15 16:11 美东　系统调研第 1 轮：主流记忆方法、数据集现状、可复现的基线
（起因：用户要求按“调研 → 定基线并复现 → 在基线上做方法 → 迭代”的流程走，见 PROGRESS 16:09。）

### A. 数据集现状
| 数据集 | 规模 | 现状 | 对我们的意义 |
|---|---|---|---|
| LoCoMo（arXiv 2402.17753） | 10 段（公开版），每段约 9K tokens、19–35 会话 | 厂商自报 88–96%，接近饱和；数字互相矛盾（Zep 自报 94.7，ByteRover 测得 75.1） | 只适合当辅助；我们用的 luna 早期结果 80–85 |
| LongMemEval（ICLR 2025，arXiv 2410.10813） | 500 题，-S 约 115K tokens/题 | 厂商自报 92–95%；单会话类 98%+ 已饱和，多会话 88、知识更新 93.6 还有空间 | 必须报；重点看多会话、知识更新、时间推理 |
| **BEAM**（ICLR 2026，arXiv 2510.27246） | 100 段、2000 题，100K/500K/1M/10M | **没饱和**：Hindsight 1M 73.9、10M 64.1；Mem0 1M 64.1、10M 48.6；论文的 LIGHT/RAG 1M 约 31–34 | **最适合做主战场**；我们已有适配器和官方判分 |
| LongMemEval-V2（arXiv 2605.12493） | 451 题，网页智能体轨迹，最长 115M tokens | 新，指标是相对参考前沿的“准确率-延迟”增益，官方判分 gpt-5.2 | 设定偏智能体工作流，暂不作主战场 |
| MemoryAgentBench（ICLR 2026，arXiv 2507.05257） | 把长上下文数据改成增量多轮 | 指出 LoCoMo 太短 | 可作补充 |

链接：Mem0 汇总 <https://mem0.ai/blog/ai-memory-benchmarks-in-2026>；Hindsight BEAM <https://hindsight.vectorize.io/blog/2026/04/02/beam-sota>；LongMemEval-V2 <https://github.com/xiaowu0162/LongMemEval-V2/>。

### B. 方法（按可复现性排）
| 方法 | 出处 | 核心做法 | 报告成绩 | 代码 |
|---|---|---|---|---|
| **Hindsight**（Vectorize） | 博客 + arXiv 2512.12818（待读） | 改进事实抽取 + 检索管线 + “observations”（从事实归纳更高层认识） | BEAM 100K 73.4 / 500K 71.1 / 1M 73.9 / 10M 64.1（自报） | 开源 github.com/vectorize-io/hindsight（约 5 万星） |
| **LightMem** | ICLR 2026，arXiv 2510.18866 | 感觉记忆预压缩过滤 → 按主题分组的短期记忆 → 离线“睡眠时”更新长期记忆 | LongMemEval 比 A-Mem 高 2–7.7 分，LoCoMo 最多 +29.3；token 少 20–38 倍 | 开源 github.com/zjunlp/LightMem |
| A-Mem | NeurIPS 2025，arXiv 2502.12110 | Zettelkasten 式笔记互链、记忆演化 | LoCoMo（F1/BLEU） | 开源 github.com/WujiangXu/A-mem |
| MemoryOS | EMNLP 2025 Oral，arXiv 2506.06326 | 短/中/长期三级存储（类操作系统分页） | LoCoMo F1 相对基线 +49%（GPT-4o-mini） | 开源 github.com/BAI-LAB/MemoryOS |
| Mem0 | arXiv 2504.19413 | 抽取事实 + 增删改合并 + 向量检索（+图） | 自报 LoCoMo 92.5、LongMemEval 94.4、BEAM-1M 64.1；独立复测开源版 LongMemEval 只有 32.4（单一来源） | 开源（平台版更强） |
| Zep / Graphiti | arXiv 2501.13956 | 时间知识图谱 | LongMemEval 71.2；LoCoMo 自报 94.7 / 他测 75.1 | 开源 Graphiti |
| LycheeMemory V2 | arXiv 2608.12990 | 分段 → 类型化自包含记录 → 多路检索 | 见 SURVEY 前文 | 我们有 Lychee-lite 复现 |
| Agent Zero Memory | arXiv 2608.29606 | 时间线 + 实体事件图 + 文档记忆，三路智能体检索，带出处 | LongMemEval 95.6，LoCoMo 93.6 | 无代码 |
| AtomMem | arXiv 2606.19847 | 抽取高价值原子事实 → 层级事件结构 + 时间画像 → 联想图检索 | LoCoMo “SOTA”（摘要无数字） | 无代码。**和我们的编译式记忆有重叠（原子事实），要细读** |
| EMem（Simple Yet Strong Baseline） | arXiv 2511.17208 | 会话拆成带实体、出处的基本语篇单元（EDU），异构图 + 稠密检索/LLM 过滤 | LoCoMo、LongMemEval-S“持平或超过强基线、上下文更短” | 待发布 |
| MemReader | arXiv 2604.07877 | GRPO 训练的主动抽取器（写/缓/查/弃） | LoCoMo、LongMemEval、HaluMem | 模型开放，集成在 MemOS |
| LeanMem | arXiv 2608.03463 | 画像/事件/原始记录三类，只更新事件 | 比最强基线最多 +15.1 | 待查 |

### C. 评测方法学（重要）
- **MemDelta**（arXiv 2606.29914，<https://arxiv.org/abs/2606.29914>）：单作者，LongMemEval-S，一次只改一个部件。逐字 RAG 47.2 vs 全上下文 49.8（不显著）；**只换向量模型就差 6.2 分**；模型家族能让 RAG 和全上下文的排名反转；Mem0 输给云端 RAG 1.2 分、成本约 50 倍；智能体自管记忆 42 < 基础检索 47。
- 结论：这个领域大量“提升”来自换模型、换向量、换判分，而不是记忆方法本身。**我们的基线必须在同一个读者模型、同一个向量模型、同一个判分下复现**，否则没有意义。这也是我们的机会：用严格对照的方式证明方法有效。

### D. 初步判断（给定基线用）
1. **主战场选 BEAM（100K 起步，再 1M）**：没饱和、ICLR 2026 新数据集、强系统也只有 64–74、我们已有适配器和官方判分（luna 上 rag 66.0、lychee 69.0，120 题）。辅以 LongMemEval-S 和 LoCoMo。
2. **基线候选**（都开源、可在同一读者模型下复现）：
   - Hindsight（BEAM 上最强的开源系统）
   - LightMem（ICLR 2026 正刊，代码开源）
   - 加上我们自己的简单基线：全上下文（装得下时）、逐字 RAG、记录 RAG（Lychee-lite）
3. 下一步：读 Hindsight 论文（2512.12818）和 LightMem 全文；确认两者能否接 luna（OpenAI 兼容接口）和我们的 H20 向量服务；在 BEAM-100K 开发集上复现。

## 2026-10-15 16:18 美东　精读：Hindsight 与 LightMem（复现用）
### Hindsight（arXiv 2512.12818，<https://arxiv.org/html/2512.12818>，代码 <https://github.com/vectorize-io/hindsight>，MIT）
- **写入（retain）：** LLM 把对话转成叙述式事实，每段对话 2–5 条粗粒度事实，含指代消解、时间规范化、参与者、类型（world / experience / opinion）、实体。每条有发生区间和提及时间。实体消解用字符串、共现、时间接近度加权。事实之间连四种边：时间、语义、实体、因果（LLM 抽）。
- **观察（observations）：** 对每个实体，把提到它的所有事实汇总成实体摘要，有新事实时在后台重新生成。
- **检索（recall）：** 语义（pgvector）、BM25、图上扩散激活、时间区间四路并行，RRF（k=60）融合，交叉编码器 ms-marco-MiniLM-L-6-v2 重排，按调用方给的 token 预算装填。
- **回答（reflect）：** 检索后加上“记忆库人设”（怀疑度、字面度、共情度）生成回答，并可形成新观点。
- **成绩：** LongMemEval-S 83.6（OSS-20B）/ 89.0（OSS-120B）/ 91.4（Gemini-3）；同一个 20B 模型的全上下文只有 39.0。LoCoMo 83.2 / 85.7 / 89.6。判分用 OSS-120B，和基线不是同一判分（论文自己也承认）。
- **没有消融**，不知道哪个部件起作用。**这是我们可以做的地方。**
- **部署：** `pip install hindsight-all`，本机带嵌入式库；`HINDSIGHT_API_LLM_BASE_URL` / `HINDSIGHT_API_LLM_MODEL` 接 OpenAI 兼容接口；向量模型默认本地 BAAI/bge-small-en-v1.5（必须 384 维）。官方评测改用 AMB（agent-memory-benchmark）跑 LoCoMo、LongMemEval、BEAM-100K。配置文档：<https://hindsight.vectorize.io/developer/configuration>、<https://hindsight.vectorize.io/developer/models>。

### LightMem（ICLR 2026，arXiv 2510.18866，代码 <https://github.com/zjunlp/LightMem>，MIT）
- **感觉记忆：** LLMLingua-2 按 token 保留概率压缩（保留率 0.4–0.8）；512 token 缓冲满后分话题（注意力峰值和相邻句向量相似度骤降两者取交集，只切用户轮）。
- **短期记忆：** 话题组累积到阈值（0–1024 token）后，LLM 对每组做摘要。
- **长期记忆：** 条目 = {话题, 摘要向量, 用户轮, 助手轮}，向量模型 all-MiniLM-L6-v2；检索用余弦相似度。在线只插入（软更新），离线“睡眠时”用 LLM 对相似条目做增删改合并。
- **成绩（ACC，判分 GPT-4o-mini）：**
  - LongMemEval-S（GPT-4o-mini 作读者）：全上下文 56.8、简单 RAG 61.0、A-Mem 62.6、Mem0 53.6、MemoryOS 44.8、LightMem 68.6。
  - LoCoMo：全上下文 71.8、简单 RAG 63.6、A-Mem 64.2、Mem0 61.7、LightMem 73.0（和全上下文基本打平）。
- **看点：** 他们自己的表里，很多“记忆系统”不如简单 RAG 甚至全上下文；LightMem 主要优势在成本（token 少 20–38 倍）和 LongMemEval。仓库还带了 Mem0、A-Mem、LangMem 的统一基线框架（`src/lightmem/memory_toolkits`）。
- **部署：** Python 3.11，`pip install -e .`；需下载 LLMLingua-2（bert-base 多语言版）和 all-MiniLM-L6-v2；向量库 qdrant 本地路径；LLM 配置可填 base_url。

### 复现计划（luna，只跑子集，见 PROTOCOL.md）
1. BEAM-100K 开发集 6 段、120 题：全上下文（若装得下）、逐字 RAG、记录 RAG（已有）、Hindsight、LightMem。
2. LongMemEval-S 开发子集 60 题：同上。
3. 目标是在同一读者、同一判分下看清相对排序；能否接近论文绝对数不是重点（读者和判分都不同）。
4. 本机没有 venv、pip、sudo，用 uv 装独立环境（`~/v3venvs/`）。

## 2026-10-15 18:39 美东　统一评测框架：AMB（agent-memory-benchmark）
- 代码 <https://github.com/vectorize-io/agent-memory-benchmark>（Vectorize，Hindsight 团队），本地克隆在 `~/v3venvs/amb`。
- 自带数据集：BEAM（100k/500k/1m/10m）、LoCoMo、LongMemEval、LifeBench、MemBench、MemSim、PersonaMem、PrecisionMemBench。
- 自带记忆适配器：hindsight（本地 / 云 / HTTP）、mem0、mem0_cloud、cognee、mastra、mastra_om、supermemory、bm25、hybrid_search、none。
- 回答和判分的 LLM 可选 gemini / groq / openai（`OMB_ANSWER_LLM`/`OMB_ANSWER_MODEL`、`OMB_JUDGE_LLM`/`OMB_JUDGE_MODEL`）；openai 走官方 SDK，可用 `OPENAI_BASE_URL` 指到我们的 Pro 代理。BEAM 判分按评分点打分，默认用 gemini-3.5-flash，作者注明“判分会影响分数，比较时必须固定判分模型”。
- Hindsight 适配器里记着针对 BEAM 的调整，例如按记忆 id 去重而不是按原文块去重，单这一项在 BEAM-100K 上 +0.070（100 题配对，40 好 12 坏）。说明 Hindsight 的成绩里有不少工程细节。
- **决定：** 用 AMB 作为基线复现和后续方法评测的统一框架。所有方法用同一个回答模型和同一个判分模型（都用 luna），我们的方法也写成 AMB 适配器。这样对照最干净，别人也能复现。我们自己的 `mem/beam.py` 结果保留作对照（判分提示是 BEAM 官方原文）。

## 2026-10-16 14:35 ET — 读取时冲突检测（H39 的新颖性核查）
- 结论：“答题前在检索到的上下文里找冲突”在 RAG 里已有成熟工作，H39 作为单独部件不新，只能作为整套系统里的一个部件，或者需要在记忆场景里找到不同点（冲突来自用户自己前后说法，而非来源可信度）。
- Astute RAG（Wang et al., ACL 2025）：引出模型内部知识，与检索内容做来源感知的整合，标出一致和冲突的信息，按一致组分别作答再比较。https://arxiv.org/abs/2410.07176
- ConflictRAG（2026）：显式冲突检测阶段，嵌入 MLP 分类器 + LLM 复核，检测 F1 88.7%。https://arxiv.org/pdf/2605.17301
- Does RAG Know When Retrieval Is Wrong?（2026）：诊断知识冲突下对上下文的依从。https://arxiv.org/pdf/2605.14473
- 记忆侧：LatticeMind（2608.08236）把矛盾处理作为记忆的一级操作（多智能体）https://arxiv.org/pdf/2608.08236 ；STALE（2026-05）基准：后来的观察在没有明说否定时使早先记忆失效 https://arxiv.org/abs/2605.06527
- BEAM 第三方：ReMe 报告 100K 矛盾类 0.438 https://reme.agentscope.io/en/benchmarks/beam ；cognee 声称 BEAM SOTA https://www.cognee.ai/blog/deep-dives/benchmarking-cognee-on-beam

## 2026-10-16 17:08 ET — 别人在 BEAM 上的高分是怎么来的（用户要求：分析别人做法，在其上改进）
| 系统 | BEAM 100K / 500K / 1M / 10M | 做法要点 | 可比性问题 |
|---|---|---|---|
| Hindsight（Vectorize） | 73.4 / 71.1 / 73.9 / 64.1（博客）；另一处 86.2 / 80.1 / 79.1 / 64.1（past.dev 表） | 叙述式事实抽取（每段 2–5 条，带时间、实体、类型）；四路检索（向量、BM25、图扩散、时间）RRF + 交叉编码器重排；observations（按实体汇总、后台重生成）；按 token 预算装填 | 无消融；两个来源数字不一致；答题、判分模型未写。我们在 luna 读者 + 官方判分下复现 1M 65.8、100K 66.6 |
| past.dev | 92.1 / 89.6 / 90.7 / 85.0 | 页面不讲架构 | **答题和判分都是 gpt-5.6-luna（自己判自己）**；用“未改动的公开提示，含来自数据集题目标注的指引”；对手数字取自各自发布 |
| Exabase M-1 | 76.9 / — / 75.0 / 68.0 | 自称“检索架构比模型规模重要” | 自报 |
| Honcho | 63.0 / 64.9 / 63.1 / 40.6 | — | 自报 |
| Scroll（arXiv 2608.21690） | 10M 73.1；LongMemEval-S 94.8 | 会话 = 只追加事件日志 + 持久 Python 内核；模型写代码检索、物化、变换状态，只把打印的投影放进工作视图；满了就驱逐，留下指向日志地址的“地标” | 多轮 agent（我们的非目标）；Qwen3.8-Max |
| BEAM 论文 LIGHT / RAG | 约 31–36 | 论文自带基线 | 老模型 |
- AMB 里 mem0 的 BEAM 答题提示（verbatim，在 ~/v3venvs/amb/src/memory_bench/dataset/beam.py）：要求“扫完所有记忆、交叉比对、间接指代先搜再下结论、找不到就说固定的拒答句、给出具体名字日期数字、不编造”。AMB 作者注明：另一种做法是按题目的真实类别换提示措辞（即“题目标注的指引”），他们没用。past.dev 用了。
- LongMemEval 侧（Memanto 2604.22085 渐进消融）：召回深度 k=10→40 一步 +20.4，提示优化 +2.2。说明“给多少、怎么检索”的杠杆远大于写入侧。
- 判断：
  1. 公开高分里混着三种与记忆方法无关的因素：读者 / 判分模型（含自判）、按题型改提示、召回深度。我们一直用同一读者、同一官方判分、极简读者提示，所以绝对分数低于他们，相对比较是干净的。
  2. 最强的可复现系统是 Hindsight。它的弱点（我们的 erran）：读者“信息在却没用上”19.6/100 题（1M），上下文约 12.4 万字；矛盾题 30–47 分。
  3. 可行路线：以 Hindsight 为底座做延伸，把我们验证过的读取侧部件加上去：读取时冲突检查（已测：100K 矛盾 + 知识更新 +8.5，p=0.022）；把它 12 万字的召回编译成更短、按方面组织的视图（目标：减少“没用上”的丢分并降成本）。
- 来源：https://hindsight.vectorize.io/blog/2026/04/02/beam-sota ；https://past.dev/benchmarks ；https://arxiv.org/abs/2608.21690 ；https://arxiv.org/html/2604.22085v1 ；https://www.cognee.ai/ai-memory-benchmarks ；https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/REPORT.md

## 2026-10-19 21:08 ET — 查新：“读时整理”这一主张已有人做（影响论文定位）
- **DeferMem**（arXiv 2605.22411，2026-05）：明确提出预先建好的记忆单元粒度在问题未知时决定，因而改为“查询时高召回检索 + 按问题蒸馏证据”；蒸馏器用 RL（DistillPO）训练，作用于原文段落；LoCoMo、LongMemEval-S。**与我们的总主张几乎相同**。区别：我们不训练、在 LLM 生成的记录之上编译、笔记有固定结构（变化 / 冲突 / 未找到 / 长期指令）、笔记与原记录一起给读者、BEAM-1M、配 Hindsight 审计。https://arxiv.org/abs/2605.22411
- **Pull**（2609.14773）：查询时按需展开原文轮次（可逆），零 LLM 写入；BEAM-1M 14 段 263 题。读时按需取原文，不编译笔记。https://arxiv.org/abs/2609.14773
- **RENDER**（2608.23568）：同一段历史以不同“给读者的形式”呈现，分数差 24.6–48.8；主张记忆评测要控制读者所见材料。与我们审计部分（读者 / 提示 / 预算决定排名）同向，须引用并区分。https://arxiv.org/abs/2608.23568
- **Presentation, Not Mechanism**（2607.16019）：结构化“修订账本”的增益几乎全来自呈现方式；控制呈现后细粒度机制贡献≈0。与我们“写时整理不必要”同向。https://arxiv.org/abs/2607.16019
- 写时冲突处理的代表：TOKI（2606.06240，双时态算子）、TEPA（2608.07429，撤销旧记忆并保留历史）、MOSAIC（2607.16211，保存时检测冲突）、MemConflict（2605.20926，冲突评测）。
- 结论：“organize at read time, not write time”不能作为新意。可保留的新意：(1) 免训练、结构化的读时编译，且笔记与原记录并给（消融已表明两者都需要）；(2) 长期指令这一部分（指令遵循 62.5→85.8）；(3) BEAM-1M 上同读者、同判分、同提示的系统审计；(4) 三个读者家族上一致。论文引言和相关工作需改写，正面对比 DeferMem、RENDER、Presentation-not-Mechanism、Pull。

## 2026-10-23 03:20 ET — 查新：长期指令 / 隐式约束记忆（方法候选初筛）
- 问题已被多篇评测指出：问题本身不提约束时，按相似度检索会漏掉用户早先立下的规则/偏好。
  - CUE-Mem（2609.32574）：记忆系统常漏掉隐式线索、退回拒答。https://arxiv.org/abs/2609.32574
  - When Users Don't Ask（2609.03467）：上下文驱动的记忆检索；PersonaMem 隐式个性化前沿模型仅 37–48%。https://arxiv.org/html/2609.03467
  - Beyond Static Dialogues（2605.31086）：“陷阱”请求与用户更新后的状态隐式冲突。https://arxiv.org/html/2605.31086v2
  - Towards Root Memories（2606.23283）：隐式逻辑记忆检索，对比 PrefEval。https://arxiv.org/pdf/2606.23283
  - Memory Retrieval for Changing Preferences（2606.02976）。https://arxiv.org/pdf/2606.02976
  - AlpsBench（2603.26680）；LoCoMo-Plus（潜在约束一致性）。https://arxiv.org/html/2603.26680v2
- 初步判断：评测很多，方法侧还没看到一个专门“读时激活适用约束”的做法（未读全文，需细查）。我们的数据：笔记加“长期指令”一段，BEAM 指令遵循 62.5→85.8，是所有组件里最大的效应。

## 2026-10-23 04:00 ET — 查新：现方法各部件均已被占；新候选“穷举型证据”
- 读时按问题编译：**MemCompile**（2610.23313，10-16）按问题分配证据预算、不可变 SourcePacket、单独渲染读者视图，LoCoMo 80.52%、LME-S。与我们几乎一致。https://arxiv.org/abs/2610.23313 ；另有 DeferMem、LazyMem、MemoryLACE（见上）。
- 隐式约束激活：**RootMem**（2606.23283）写时提炼“根记忆”（决策逻辑）+ 读时 LLM 路由激活，基准 IMLogic。https://arxiv.org/abs/2606.23283 ；**User as Code**（2606.16707）只追加日志 + 类型化 Python 状态 + 规则函数，聚合题约 99%（检索式 6–43%），能主动提醒（新药 vs 过敏）。https://arxiv.org/abs/2606.16707 ；Memory Retrieval for Changing Preferences（2606.02976，训练式）。https://arxiv.org/abs/2606.02976 ；LoCoMo-Plus（2602.10715，ACL'26，潜在约束一致性）。https://arxiv.org/html/2602.10715 ；LoCoMo-Conv（2609.03467）测了多面查询改写 + RRF。
- 只追加账本：SSGM（2603.11768）、Springdrift（2604.04660）、TARL（2608.03699）、MemTxn（2607.27834）、MRAgent（2606.06036）。可引用笔记：Agent Zero Memory（2608.29606）、Eywa（2605.30771）。
- 新候选（穷举型证据）相关：top-k 在重复历史上塌缩、计数需每个实例都在——xMemory / Beyond RAG（2602.02007）https://arxiv.org/html/2602.02007v1 ；Don't Ask the LLM to Track Freshness（2606.01435，确定性 max() 比 LLM 判断高约 10.8）https://arxiv.org/html/2606.01435v1 ；MemFail（2605.26667）https://arxiv.org/html/2605.26667v1 ；Recall@k 在 k≈40 后饱和。未见“对计数/全部/从未/最新值这类题给出完整性保证”的非参数方法；最近的是 User as Code（需要预先定义的类型化状态）。
