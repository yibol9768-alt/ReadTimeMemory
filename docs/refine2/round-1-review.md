# Round 1 Review（Codex，gpt-6.1-sol，Pro；2026-10-16 16:00 ET）

- 总分 6.35/10，结论 REVISE
- 分项：问题忠实 9，方法具体 6，贡献质量 4，前沿利用 8，可行 8，验证聚焦 6，投稿就绪 4
- Codex 线程 id：01a14645-3421-7291-b36e-a3e79ff72525

<details>
<summary>原文</summary>

Codex 评审的完整原文如下。

**Codex 线程 id:** `01a14645-3421-7291-b36e-a3e79ff72525`。这是线程 id,后续几轮要在同一线程继续,用 `--resume-last`。本轮的 turn id 是 `01a14645-3675-7420-9115-0a66749a71a3`。

---

**Verdict: REVISE. Overall score: 6.35/10.** CTR addresses the anchored problem and is feasible to build. Its main weakness is that the implemented rule does not yet establish the claimed contribution of scope-dependent resolution selection. It currently resembles query-conditioned summary selection added to ordinary retrieval.

I read the Problem Anchor first, the complete proposal, and the relevant supporting evidence. No files were modified.

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 9/10 | Preserves the fixed-reader, long-conversation setting and both point and broad questions. |
| Method Specificity | 6/10 | Implementable outline, but budget allocation, coverage, and scope semantics remain inconsistent. |
| Contribution Quality | 4/10 | Focused ambition; substantial overlap with existing gist-guided and hierarchical retrieval. |
| Frontier Leverage | 8/10 | Appropriate use of an LLM for compilation and selection; training is unnecessary here. |
| Feasibility | 8/10 | Existing representations and development infrastructure make a small implementation realistic. |
| Validation Focus | 6/10 | Sensible datasets and controlled reader, but the central comparisons do not fully isolate the mechanism. |
| Venue Readiness | 4/10 | The exact new mechanism is unvalidated, and the novelty argument overstates differences from prior work. |

Weighted score: 0.15(9)+0.25(6)+0.25(4)+0.15(8)+0.10(8)+0.05(6)+0.05(4)=6.35.

**Method Specificity — 6/10**

**Weakness:** The thesis promises to cover the required scope, then spend the *remaining* budget on finer representations. The actual [assembly rule](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine2/round-0-initial-proposal.md:69) reserves 8K characters for fine retrieval and adds up to 30K characters of summaries. It does not allocate a shared budget between resolutions. Moreover, truncating selected aspects by RRF directly breaks the stated coverage guarantee.

Including every selected summary also guarantees neither that the selector found every necessary aspect nor that the summaries retained the answer evidence.

**Concrete fix — CRITICAL:** Specify one deterministic assembler under a total reader-token budget. Start with a compact dated representation of each selected aspect, then spend the remaining tokens on ranked L1/L0 evidence while removing duplication. Define what happens when even the initial representations exceed the budget, and explicitly report omitted aspects. Describe the guarantee as representation of selected aspects, conditional on fitting, not complete answer-evidence coverage.

**Weakness:** A(q) has conflicting meanings. If it denotes the aspects containing necessary evidence, a point question generally has a nonempty scope. If point questions should produce an empty set, it instead denotes aspects needing an aggregate view. This still entails an implicit decision about whether coarse context is useful. The identity fallback at A(q)=∅ does not guarantee point-question safety when the selector makes mistakes.

**Concrete fix — IMPORTANT:** Define A(q) as "aspects requiring aggregate context." Preserve ordinary fine retrieval as the fallback, specify malformed-output handling, and measure erroneous coarse-context activation on point questions. Replace the blanket "point questions are unharmed" claim with a conditional statement.

**Contribution Quality — 4/10**

**Weakness:** The related-work argument understates the closest mechanisms:

- **RAPTOR:** Its collapsed-tree retrieval is accurately described, but RAPTOR also includes top-down traversal with explicit breadth and depth controls. "Coarse to fine" alone is therefore insufficient differentiation. [RAPTOR paper](https://arxiv.org/html/2401.18059v1)
- **ReadAgent:** ReadAgent-P selects multiple detail passages from gist memory in a single selection prompt. Avoiding a multi-round agent does not distinguish CTR from this variant. [ReadAgent paper](https://arxiv.org/html/2402.09727v1)
- **GraphRAG:** The claim that it only has separate local/global pipelines and does not refine omits DRIFT, which starts with community reports and uses local search for refinement. Its iterative operation remains a legitimate distinction from CTR. [DRIFT documentation](https://microsoft.github.io/graphrag/query/drift_search/)
- **LightRAG and Hindsight:** LightRAG extracts broad and specific query keywords for dual-level retrieval. Hindsight retrieves observations alongside raw facts and documents multiple abstraction levels. Calling these systems uniformly single-granularity is inaccurate. [LightRAG paper](https://arxiv.org/html/2410.05779v1), [Hindsight observations](https://hindsight.vectorize.io/developer/observations)
- **Adaptive routing and hierarchical conversational memory:** Query-dependent retrieval strategy selection and hierarchical conversation representations already exist. Neither the selector nor the three-layer representation independently establishes novelty. [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/), [MemTree](https://arxiv.org/abs/2410.14052)

**Concrete fix — CRITICAL:** Center the contribution on one precise question: **Does explicitly preserving coverage across selected conversational aspects produce a better allocation of limited detail tokens than gist-guided relevance selection?**

Use the existing aspect membership, timestamps, and source links to implement that allocation. Demonstrate why preserving distributed historical evidence matters beyond selecting relevant summaries. A coverage constraint or knapsack formulation alone would still be standard machinery; the contribution must lie in its necessary role and demonstrated behavior in long-term conversational memory.

**Weakness:** The strongest supporting evidence does not yet identify scope coverage as the remaining bottleneck. In [H37](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/PROGRESS.md:1055), giving all aspect summaries still trailed Hindsight by **5.7 points on broad 1M questions**. Among 23 scoring points Hindsight recovered and this view missed, the diagnostic labeled eight PRESENT and ten PARTIAL. That leaves both reader usability and summary detail loss plausible.

This does not disprove CTR: selecting fewer distracting aspects and adding fine evidence could help. It does mean that the existing +2.5 overall improvement over Lychee does not establish the proposed explanation.

**Concrete fix — IMPORTANT:** Use those existing failures to specify CTR's expected repair. Distinguish recovering absent evidence, completing partial evidence, and helping the reader use present evidence. The proposed selection rule should predict which failure category it improves.

**Validation Focus — 6/10**

**Weakness:** Sharing representations and matching context length is valuable, but the proposed comparison still changes several things together: LLM scope selection, vector-only versus hybrid ranking, aspect prioritization, and packing. A win over collapsed-tree-style cosine retrieval would establish the package's value, not uniquely the cover-then-refine rule.

**Concrete fix — CRITICAL:** Keep the benchmark plan focused and strengthen the central controls:

1. **Same predicted A(q), same candidates, same ranking and token budget:** compare coverage-preserving assembly against ordinary relevance packing. This isolates the proposed assembly rule.
2. **Same total budget:** retain the planned binary-router comparison to test whether selecting particular aspects improves over choosing a broad view.
3. Include one **ReadAgent-P-style single-selection baseline** using the same stored representations and comparable selection-call budget. This addresses the nearest conceptual alternative without requiring another large system.

Apply the common conflict checker identically, count its context and calls, and distinguish it from CTR's incremental gain. Existing single-resolution baselines can remain; a larger benchmark checklist is unnecessary.

**Weakness:** Five 1M conversations provide limited independent evidence. Three generations measure sampling variability but do not create additional independent histories. Development-set discoveries and question-level significance should not be treated as confirmation of cross-conversation improvement.

**Concrete fix — IMPORTANT:** Average repetitions within questions, retain conversation-level paired uncertainty and per-conversation differences, and use the untouched test split for the frozen primary comparison. Predefine a noninferiority margin for the 100K "does not degrade" claim. Report oracle-aspect selection as a diagnostic reference rather than a guaranteed answer-quality upper bound.

**Venue Readiness — 4/10**

**Weakness:** The proposal currently joins a plausible retrieval heuristic to a stronger, insufficiently supported resolution-allocation narrative. Previous variants' results do not validate the exact new selector and assembler.

**Concrete fix — IMPORTANT:** Freeze the corrected budgeted mechanism and a narrower claim before scaling evaluation. The decisive result should show that explicit coverage allocation beats both ordinary packing under the same scope and the closest one-call gist-guided alternative. If gains instead come from selecting fewer summaries, describe the contribution accordingly; do not attribute them to demonstrated resolution optimization.

**Simplification Opportunities**

1. Treat aspect timelines as the fixed representation used to study selection, rather than an optional second contribution.
2. Initially remove "all in-scope records first" prioritization. It changes retrieval independently and can displace a highly relevant out-of-scope record.
3. Use one total token cap, one scope-selection call, and one deterministic assembly policy.

**Modernization Opportunities**

1. Use structured selector output with stable aspect IDs and validation. If titles are ambiguous, add short descriptors and date ranges within the same call.
2. Otherwise, **NONE**. LLM compilation and selection are appropriate modern primitives. RL, diffusion, multimodal components, and trained routing would add no clear value to this anchored task.

**Drift Warning:** **NONE relative to the supplied Problem Anchor.** The anchor permits more than 8K context provided the Hindsight-relative limit holds. The earlier method tree's strict 8K objective is therefore historical context, not the governing requirement here. The recommended shared-budget assembler preserves the current problem.

**Verdict: REVISE.** Preserve the focused direction, but make the coverage-allocation mechanism explicit and test it against the closest one-call alternative before claiming a new resolution-selection method.
</details>
