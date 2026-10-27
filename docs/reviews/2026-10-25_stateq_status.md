# 独立评审：封闭世界状态查询当前状态（2026-10-25，独立 Agent，只读，未用 GPT）

评分（ICML 1–10）：新颖性 4.5；方法性 4（做出查询语言+覆盖证书+定理可到 6）；证据强度 2.5；PLAN 全部完成且过门槛的上限 6.5–7；今天写论文 3；到 7–8 的概率约 15%。

核对：文档数字与 state_p1.judged 一致（Cas 12/16、Abs 12/13、Del 9/11、Tr 9/11、ER 10/11、Agg 8/11）；但已判 11 个 episode 中 10 个是 pl、只有 1 个 sw；full_p1.judged 为 0 行（对照未判）；子串粗查 20 个 episode 的 Cas：state 18/32、full 20/32，full 常找到新值但加保留语——差距可能主要来自 ANS_SYS 的“UNKNOWN 就说不确定”口径贴合金标写法。

实现差距：stateq.py 只有槽位历史 + 规则级联 + 映射 6 槽位；查询语言、覆盖证书、三值否定、count/list_all 均未实现，本质是规则传播 JTMS + 提示。

查新：PACMI 已核实（自建 100 例，无公开基准）；未见在 MEME 上报 Cascade/Absence 方法结果的工作（机会）；组合各部分有近邻（TallyMem、ChatDB、PACMI、CUPMem）。

风险：1) 增益来自口径对齐；2) 开放对话漏抽→自信错答，LME-V2/BEAM 可能不涨；3) 被看作拼装，分离定理近乎显然。

优先（均不用 GPT）：1) Qwen3.8 判 20 个 episode 的 state/full，full 加同样 UNKNOWN 作答指引、另加“full+显式规则”对照，公平口径下仍 ≥30 点才算；2) DeepSeek 读者重跑，记规则召回、成本、改写与删 20% 证据后的误否定率；3) LME-V2 拒答+前提约 177 题先验证跨基准，不过就停。
