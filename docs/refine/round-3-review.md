# Round 3 Review（Codex，同一线程；2026-10-16）

- 总分 7.20/10，结论 REVISE（已到最大轮数）
- 分项：问题忠实 9，方法具体 8，贡献质量 5，前沿利用 8，可行 8，验证聚焦 7，投稿就绪 5
- 审稿人对投稿层级的判断：若开发集只是小幅正结果，更可能是强 workshop 或实证/分析论文；方法型 ICML 正刊需要留出集达到锚点，并且对照揭示一个有分量、非显然的原则，外加效率优势。

<details>
<summary>原文</summary>

CCR is now substantially more concrete and better controlled. Most specification objections from round 2 have been addressed. The remaining concern is whether the resulting method contributes enough beyond proposition grouping and history retrieval. It is not a question of whether the method needs more modules.

My assessment is **REVISE**. The proposal is specified well enough for a controlled pilot once a few protocol corrections are made. If the development results are only modestly positive, a strong workshop paper or an empirical/analysis paper is more likely than a method-first ICML main-track paper.

| Dimension | Round 2 | Round 3 | Assessment |
|---|---:|---:|---|
| Problem Fidelity | 8 | **9/10** | The original anchor is preserved verbatim, and all four success conditions remain binding. |
| Method Specificity | 6 | **8/10** | Proposition identity, candidate exposure, ordering, and conditional chain admission can now be implemented. |
| Contribution Quality | 4 | **5/10** | A focused minimal interface, but its operations remain close to established version-history retrieval. |
| Frontier Leverage | 8 | **8/10** | LLM-based canonicalization and question-conditioned reading are appropriate. Deterministic lookup avoids unnecessary inference calls. |
| Feasibility | 7 | **8/10** | A practical prototype using existing components. Slot consistency and chain admission are measurable risks, not unspecified subsystems. |
| Validation Focus | 6 | **7/10** | The shared ledger, frozen seeds, matched-record presentation controls, and a history-retaining annotation control improve identification. |
| Venue Readiness | 4 | **5/10** | Ready to produce useful development evidence. Novelty and main-track significance remain unestablished. |

**OVERALL SCORE: 7.20/10**, up from 6.10.

Weighted calculation: 0.15(9)+0.25(8)+0.25(5)+0.15(8)+0.10(8)+0.05(7)+0.05(5)=7.20.

These scores assess the revised proposal. No CCR performance results were supplied, so hypothetical positive outcomes receive no empirical credit.

**Problem Anchor: preserved.**

A direct comparison confirms that the original anchor is unchanged. Removing event threads from slot semantics does not itself constitute drift. A method may target one failure mechanism while retaining the complete task and evaluation requirements.

The distinction that matters is between "event ordering is not directly targeted by this representation" and "event ordering no longer matters." The proposal makes the former choice and explicitly retains all original success conditions. That is acceptable. Reporting the gap to 74.4 is also necessary. Falling short would still mean the original BEAM-100K success condition was not achieved.

**Dominant contribution: sharper.**

The proposal now has one coherent intervention. It identifies assertion families without adjudicating truth, retrieves their recorded alternatives together, and lets the fixed reader interpret them for the question. The A–F conditions are controls, not parallel contributions.

**Complexity: appropriately small.**

The architecture is no longer overbuilt or dependent on a hidden second reasoning stage. The top-20-plus-recent-10 candidate policy makes the write interface concrete. Semantic canonicalization is still substantial work, but its risks can now be audited directly.

**Frontier leverage: appropriate.**

Foundation models handle the two tasks that need semantic judgment: recognizing proposition identity and interpreting assertions for a question. Code handles lookup, ordering, and packing. I would not add training, RL, agents, a graph database, or another compilation pass.

**1. Contribution Quality — 5/10. Priority: CRITICAL.**

**Specific weakness:** The proposal has improved its scientific framing more than its algorithmic novelty.

A proposition key, append-only assertions, same-key expansion, chronological serialization, and whole-group admission form a useful design. They are also recognizable forms of keyed history retrieval. "One field" describes the schema syntax. It does not establish minimal semantic complexity or a new retrieval principle.

The literature distinctions are now more accurate:

- Graphiti retains historical facts with temporal metadata, so historical retention alone is not the distinguishing feature. [Zep paper](https://arxiv.org/html/2501.13956v1)
- LeanMem already presents event evidence chronologically within a context budget, so ordering by time alone is insufficient novelty. [LeanMem, §3.4](https://arxiv.org/html/2608.03463v1)
- TRACE distinguishes updates from unresolved contradictions, expands evidence along evolution relations, and constructs structured reader context. Avoiding its richer validity machinery is a meaningful simplification, but the general evidence-reconstruction objective is shared. [TRACE, §§II-D–II-F](https://arxiv.org/html/2607.00339v1)
- TOKI preserves losing assertions in audit records, so avoiding adjudication must be distinguished from preserving adjudicated history. [TOKI](https://arxiv.org/abs/2606.06240)

The proposal correctly notes that TRACE disables update detection in its LongMemEval-S configuration. That is a relevant implementation limitation and an opportunity for a better controlled study. It does not make proposition-history retrieval new. [TRACE, implementation details](https://arxiv.org/html/2607.00339v1)

**Concrete method-level fix:** Keep the small architecture, but make its claim operational:

> Under a fixed reader budget, proposition identity and co-selection of recorded alternatives provide sufficient structure for the targeted conversational tasks, without persistent validity judgments.

Then let the controls determine which part deserves contribution status:

| Observed result | Defensible interpretation |
|---|---|
| B improves over A, while C approximately matches B | Proposition-based expansion supplies the benefit. Chain presentation is not independently supported. |
| C reliably improves over B with identical records | Grouped chronological presentation improves this reader's use of the evidence. |
| C beats D primarily when D assigns incorrect validity labels | Evidence against unreliable persistent annotations. It is not a general demonstration that write-time reasoning is inferior. |
| C matches F with substantially fewer inference tokens or calls | A useful efficiency result for the minimal structured interface. |

Do not require every component to contribute independently. If expansion explains the gain, simplify the contribution around expansion. Conversely, do not advertise chain presentation as the main mechanism if C–B provides little evidence for it.

The remaining potentially distinctive result is the **empirical sufficiency of a restricted interface**, supported by clear improvements and meaningful efficiency. That can be a contribution. The current document has not yet established it, and ordinary history lookup remains a serious novelty challenge.

**2. Venue Readiness — 5/10. Priority: IMPORTANT.**

**Specific weakness:** Passing the proposed development gate would establish a promising intervention. It would not by itself establish a main-track method contribution.

A three-point improvement over a flat baseline on a small, repeatedly examined development set is useful evidence for continuing. Non-significant differences from D or F would not establish equivalence. A contradiction-category gain alone could remain tied to BEAM's response requirements.

**Concrete method-level fix:** Define the claim attached to each comparison before examining results. In particular:

- Treat C–A as overall utility, B–A as expansion, and C–B as presentation.
- Specify a practical noninferiority margin for claims that C is "not worse" than D or F. Use paired uncertainty against that margin. Absence of significance is not evidence of equivalence.
- Require the original anchor outcomes for declaring the original research goal achieved.
- Tie improvements to recovered or better-used assertions through the existing diagnostics, rather than relying on category averages alone.

My honest venue judgment is:

**With modest positive development results, this is more likely a strong workshop paper or a focused empirical/analysis paper.** The method would be useful, economical, and explainable, but the underlying operations are familiar.

**A method-first ICML main-track paper remains possible.** That requires the frozen method to meet the original anchor on held-out data. It also requires the controls to reveal a consequential, non-obvious principle about what structure conversational memory needs. The strongest case would combine broad task performance, a clear mechanism, and an efficiency advantage over relevant alternatives.

Matching a same-evidence annotation control does not establish parity with TRACE or Graphiti end to end. Those systems also change extraction, linking, candidate discovery, and selection. Either keep the claim specifically about validity annotations, or support any broader system-level claim with a faithful comparator. There is no need to reproduce every named memory system.

I would not add architectural complexity to manufacture a larger contribution. If the results support only a useful baseline or annotation-policy finding, report that contribution honestly.

**Remaining mechanism and protocol corrections**

These are bounded handoff corrections, not a request for another architectural redesign.

**A. Make the validity-label control capable of being correct. Priority: CRITICAL.**

The revised D condition is much better than latest-only storage. Its exact semantics still matter.

The wording that each older assertion is marked "superseded" must not become an implementation that automatically invalidates every earlier value. D should permit unresolved contradiction, coexistence, or insufficient evidence. TRACE explicitly distinguishes contradiction from update rather than invalidating both indiscriminately. [TRACE, §II-D](https://arxiv.org/html/2607.00339v1)

The [technical-gap argument](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-2-refinement.md:103) still says that write time cannot distinguish a correction from a contradiction. That is too strong. Explicit correction language, event times, and scope can supply distinguishing evidence before any future question exists.

The defensible hypothesis is that write-time judgments are sometimes unreliable or unnecessary for answering. It is not that the distinction is inherently unavailable.

Specify D's inputs and chronology. If it is intended to represent online write-time judgment, it must operate on the available prefix without the future question. If it labels completed histories afterward, describe it as an offline annotation control.

Inspect D's classification errors using the planned audit. Otherwise, a C–D advantage can reduce to the expected observation that erroneous authoritative-looking metadata harms a reader.

**B. Preserve the budget when holding record IDs fixed. Priority: IMPORTANT.**

B and C sharing record IDs is an important improvement. D adds annotations to those records, however, and serialization lengths differ across all three conditions.

If C already fills 8K characters, adding D's labels can exceed the limit. Truncating D afterward would destroy the intended same-evidence comparison.

Select a common record set that fits the relevant renderings, accounting for all metadata, or reserve annotation space consistently. Specify the ordering of closure-added records that have no original seed rank. These are deterministic implementation choices, not new method components.

**C. Clarify the relationship between full chains and filler records. Priority: IMPORTANT.**

The [packing rule](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-2-refinement.md:57) now states a valid conditional guarantee: every admitted slot has all its ledger records present.

However, filling leftover space with individual records can reintroduce one side of a chain that was skipped for being too large. A record assigned to two slots can similarly expose part of an unadmitted second slot.

This does **not** invalidate the narrower guarantee. It means the entire final context is not necessarily composed of complete histories.

Identify complete groups separately from partial filler evidence, and report their coverage separately. The simplest initial implementation can omit filler from skipped multi-assertion slots. Define deduplication for records belonging to two admitted slots so that B/C/D comparisons do not accidentally change repetition.

Do not introduce recursive closure or a graph to solve this. Accurate scope and consistent serialization are sufficient.

**D. Keep proposition identity independent of the changing answer. Priority: IMPORTANT.**

The new slot definition is usable, but the canonical statement should identify the question, including entity, attribute, and event instance where relevant, without embedding its current answer.

Two separate visits to Foot Locker must not merge simply because their attributes have the same name. Two revisions of one planned visit must not split because the revised date differs. If a record contains more independent claims than its two-slot limit supports, specify whether extraction splits it or leaves some content unkeyed.

The planned split/merge audit is the right place to examine this. The fifty-largest-slot sample measures a high-risk subset. Report it as such rather than as an unbiased population merge rate.

**E. Freeze one continuation rule. Priority: IMPORTANT.**

The proposal contains incompatible gates:

- [Line 88](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-2-refinement.md:88) uses three generations and an OR condition.
- [Line 139](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-2-refinement.md:139) uses one generation and an AND condition, with a different category requirement.

Choose one before observing results. Given the reported generation noise, the first pass is better treated as an implementation check than a definitive three-point futility decision. Use the already planned repeated comparison for the scientific continuation decision. A development continuation gate should remain distinct from the final success criteria.

**Simplification Opportunities**

1. Keep E as an optional supporting lossy ablation. A/B/C, followed by the necessary D/F comparisons, answer the central questions.
2. Remove filler from skipped multi-assertion chains in the first implementation, avoiding a mixed completeness contract.
3. Let the A/B/C results determine the final method claim. If grouping adds little beyond expansion, remove that contribution claim rather than adding another mechanism.

**Modernization Opportunities**

**NONE requiring a new model family or training procedure.** The existing LLMs, embeddings, structured IDs, and deterministic operations are sufficient for this proposal. Its weakness is not technological datedness.

**Drift Warning**

**NONE in the current problem formulation.** The anchor is preserved, and restricting slot semantics to mutable propositions is a legitimate mechanism choice while the broader task remains intact.

Dropping the overall benchmark criteria after observing gains only on contradictions would constitute drift. Choosing a workshop venue would not itself change the problem. Redefining success around a favorable subset would.

**Remaining Action Items**

1. **CRITICAL:** Define a non-forced, temporally scoped D annotation policy, and restrict conclusions to what that control actually tests.
2. **IMPORTANT:** Freeze shared-record serialization within budget, including annotation overhead, deduplication, and filler completeness.
3. **IMPORTANT:** Resolve the two continuation rules and define what "not worse" means statistically.
4. **IMPORTANT:** Use the planned pilot to determine whether expansion, presentation, or annotation avoidance carries the contribution.
5. **IMPORTANT:** Preserve all original success conditions, and assess main-track potential from held-out mechanism and task evidence, not the development gate alone.

**Verdict: REVISE.**

The proposal no longer needs a larger system or another conceptual pivot before a pilot. It needs a few precise protocol decisions and empirical evidence. Its strongest plausible outcome at present is a useful minimal method with a focused empirical finding. A method-first ICML main-track claim remains possible but requires substantially stronger evidence than modest positive development results.

No files were edited and no experiments were run.

Thread ID: `01a142f5-dd49-7931-bb2f-fa83887f262a`

</details>
