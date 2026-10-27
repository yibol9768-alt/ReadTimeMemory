# Round 2 Review（Codex，同一线程；2026-10-16）

- 总分 6.10/10，结论 REVISE
- 分项：问题忠实 8，方法具体 6，贡献质量 4，前沿利用 8，可行 7，验证聚焦 6，投稿就绪 4
- 新指出的重叠：TRACE（arXiv 2607.00339）、TECQA（ACL 2026）、TOKI（arXiv 2606.06240）

<details>
<summary>原文</summary>

CCR is more focused, more concrete, and operationally simpler than RTC. Demoting generic compilation to a control baseline was the right revision. The remaining blockers are the definition of a consistent slot, an overflow rule that contradicts the claimed preservation guarantee, and a novelty argument that understates existing history-preserving temporal memory.

| Dimension | Round 1 | Round 2 | Assessment |
|---|---:|---:|---|
| Problem Fidelity | 8 | **8/10** | The original anchor is preserved verbatim. The broader capabilities remain evaluation requirements. |
| Method Specificity | 5 | **6/10** | The write/read interface is clearer, but slot identity and bounded chain packing remain unresolved. |
| Contribution Quality | 3 | **4/10** | A sharper conversational-memory mechanism, but substantial overlap with versioned facts and temporal evidence reconstruction remains. |
| Frontier Leverage | 8 | **8/10** | LLM-based semantic labeling, deterministic expansion, and reader-side reasoning are appropriate. |
| Feasibility | 7 | **7/10** | A prototype is practical. Reliable slot reuse over 1M-token histories is the main implementation risk. |
| Validation Focus | 5 | **6/10** | Frozen planning, clustered comparisons, and stronger baselines improve the design. Important evidence and presentation confounds remain. |
| Venue Readiness | 3 | **4/10** | Suitable for a focused development-stage investigation; the central guarantee and novelty comparison need correction first. |

**OVERALL SCORE: 6.10/10**, up from 5.50.

Weighted calculation: 0.15(8)+0.25(6)+0.25(4)+0.15(8)+0.10(7)+0.05(6)+0.05(4)=6.10.

**Problem Anchor: preserved.** A direct text comparison confirms that the original anchor is unchanged. The revised coverage-versus-usability interpretation appropriately qualifies its overstrong causal wording. Targeting updates and contradictions is compatible with that anchor, provided the overall success conditions (including summarization and exceeding full context on BEAM-100K) remain binding.

**Dominant contribution: sharper.** Preserving assertions, expanding their associated history, and presenting that history together form one coherent mechanism. The write-time/read-time comparison should remain supporting analysis.

**Complexity: simpler, not overbuilt.** Removing query-time compiler calls is a substantial simplification. However, semantic canonicalization is doing more work than "one string field" suggests. The architecture is small; its essential interface is underspecified.

**Frontier leverage: appropriate.** Using an LLM to identify semantic correspondence and another invocation of the fixed reader to interpret evidence is reasonable. Exact lookup and sorting should remain ordinary code. No training, RL, additional agents, or other model family is needed.

**1. Method Specificity — 6/10. Priority: CRITICAL.**

**Specific weakness: slot consistency is assumed rather than specified.**

The [write procedure](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-1-refinement.md:139) exposes at most N existing slot names and asks the encoder to reuse a name for the same matter. This leaves several decisions that determine whether closure works:

- Which slots remain visible when the registry exceeds N?
- How does a distant topic reuse its original identity after disappearing from that visible set?
- How are aliases, paraphrases, and ambiguous subjects handled?
- Does one slot identify a subject-attribute family, an individual event, or a broad event thread?
- What happens when one record contains claims belonging to several families?

The examples mix relatively narrow attributes with broad experiences. Those groupings have different semantics. A broad shopping-experience thread can support event ordering but grow large; an atomic attribute slot can remain compact while failing to connect an event sequence.

Moreover, N controls registry visibility, not semantic granularity. Varying N is not a clean slot-granularity ablation.

**Concrete method-level fix:** Define the slot's semantic unit, use stable IDs, and specify a deterministic policy for exposing candidate IDs to the encoder. Keep entity/coreference disambiguation distinct from deciding which assertion is true. Define reuse, creation, ambiguity, and multi-claim handling explicitly. Existing retrieval can supply candidate slots without introducing a new learned retriever.

Audit both directions of error: required claims split across slots, and unrelated claims merged into one slot. Checking only whether contradiction pairs share a key misses the budget damage caused by over-grouping.

**Specific weakness: the overflow exception invalidates the main guarantee.**

The proposal repeatedly claims that corrections cannot separate from their originals and that chains are never internally truncated. Yet [the algorithm](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-1-refinement.md:145) retains only the first, last, and query-relevant intermediate records when a chain exceeds B_chain.

An original statement and its correction can both occur in the middle. Relevance selection can retain the original while dropping the correction. An omission marker discloses missing material but does not preserve the dependency.

There is also a fundamental boundary: an arbitrarily long chain cannot simultaneously remain complete and fit inside 8K characters.

**Concrete method-level fix:** Choose an honest bounded contract. The simplest version admits only complete chains that fit the remaining budget, skips chains that do not fit, and reports excluded chains. Its guarantee is conditional: all recorded assertions for each admitted slot are present.

If oversized chains are common, this version may fail empirically. Then a partial-history selection mechanism must explicitly preserve the dependencies it claims to preserve; retaining endpoints is insufficient. Do not describe that partial history as full closure.

Specify whether packing skips a non-fitting chain or stops entirely. Count headers, dates, IDs, and completeness metadata inside the final limit. Preserve utterance time separately from event time, and use source-turn order to break same-date ties.

Finally, the reader instruction must describe completeness accurately. "All statements" cannot mean the complete history when extraction may omit assertions, slots may split, and chains may be excluded.

**2. Contribution Quality — 4/10. Priority: CRITICAL.**

**Specific weakness: the central contrast conflates write-time adjudication, historical retention, and query-time exposure.**

These are separate decisions. A system can assign validity at write time, retain both assertions, and expose their history at read time. Consequently, "write-time resolution loses the old statement; read-time resolution preserves it" is not an adequate characterization of the literature.

My assessment of the requested comparisons is:

| Related work | Relevant existing capability | Implication for CCR |
|---|---|---|
| **Zep/Graphiti** | Maintains historical relationships and bitemporal metadata. Invalidation sets temporal boundaries; it does not simply delete the earlier fact. Its context constructor includes validity ranges. | The latest-only control is not a faithful stand-in for Graphiti. CCR must distinguish avoiding adjudication from merely retrieving retained history. [Zep paper](https://arxiv.org/html/2501.13956v1) |
| **Mem0** | Uses semantic ADD/UPDATE/DELETE/NOOP decisions. UPDATE can augment complementary information. Its graph variant marks obsolete relationships invalid rather than physically removing them. | "Keep the latest per slot" is a synthetic lossy policy, not an implementation of Mem0's update mechanism. [Mem0 paper](https://arxiv.org/html/2504.19413v1) |
| **LeanMem** | Preserves immutable source-grounded records, groups related events during evolution, and chronologically orders event evidence during query-time composition. | Temporal grouping and chronological presentation are already present. CCR's possible distinction is complete assertion-family expansion without state consolidation. [LeanMem, §§3.3-3.4](https://arxiv.org/html/2608.03463v1) |
| **A-Mem** | Stores timestamped notes, establishes semantic links, and evolves contextual descriptions, keywords, and tags. | Memory evolution is not necessarily equivalent to collapsing competing assertions into one current fact. CCR offers a simpler, more constrained association rule, whose adequacy remains to be established. [A-Mem paper](https://arxiv.org/html/2502.12110v1) |
| **Hindsight** | Maintains narrative facts with entity and temporal links; observations are summaries above underlying facts. Graph retrieval can surface related evidence. | Observations do not imply that the underlying history has disappeared. CCR needs to establish the benefit of its particular grouping and preservation policy. [Hindsight paper](https://arxiv.org/html/2512.12818) |

Two additional papers are especially relevant.

**TRACE** addresses evolving conversational state using update and contradiction relations. It expands retrieval seeds along evolution edges, renders structured support paths under a context budget, and keeps items atomic during truncation. It also distinguishes updates from unresolved contradictions. This directly overlaps CCR's motivation and several proposed read-time operations. Its richer graph and write-time validity annotations provide concrete differences, but "recover related versions and present structured evidence" is already established. [TRACE, §§II-D-II-F](https://arxiv.org/html/2607.00339v1)

**TECQA** constructs temporal evidence chains through structure-guided retrieval, temporal-neighbor pruning, and serialization for an LLM reader. Its setting assumes temporal knowledge graphs rather than extracted conversational records, so it is not identical to CCR. Nevertheless, preserving connected temporal evidence for reader reasoning is not itself new. [TECQA](https://aclanthology.org/2026.acl-long.1361/)

Related work also explicitly separates adjudication from retaining the losing assertion: **TOKI** studies write-time resolution operators with provenance-preserving audit rows. It is not a substitute for CCR's QA evaluation, but further weakens the equation of adjudication with historical erasure. [TOKI](https://arxiv.org/abs/2606.06240)

The proposal also incorrectly groups CompAct and Chain-of-Agents with independent fragment processing. Their sequential information exchange was already identified in round 1. RTC is the independent control proposed here; that limitation should not be attributed to every related method. [CompAct](https://arxiv.org/html/2407.09014v2), [Chain-of-Agents](https://arxiv.org/abs/2406.02818)

**Is the composition new enough? Not yet established.** Stable keys, append-only assertions, keyed history lookup, chronological rendering, and grouped packing are recognizable operations. I found no basis to claim that this exact composition duplicates every detail of a particular paper, but the present argument does not establish a sufficiently distinct method either.

**Concrete method-level fix:** Center the contribution on a precise, bounded assertion-preservation contract and show what the minimal slot interface achieves beyond ordinary version-history retrieval. Correct history-preserving comparators are essential.

A plausible paper would show that this restricted interface supplies the necessary evidence more reliably, at lower construction or query cost, than more elaborate history-aware mechanisms under the anchored reader budget. That could support a focused minimal-method contribution. Merely outperforming flat retrieval or latest-only storage would not establish it.

Do not add a graph simply because these papers use graphs. The opportunity is to demonstrate that a smaller interface is sufficient, not to rename established history operations.

**3. Validation Focus — 6/10. Priority: CRITICAL.**

**Specific weakness: "same recall" now refers only to the initial seeds.**

CCR expands those seeds with ledger records that ordinary retrieval or RTC may never receive. That is a legitimate mechanism, but it changes available evidence. Freezing the initial ranking does not make the final candidate sets identical.

**Concrete method-level fix:** Separate expansion from presentation using the planned conditions:

- Compare seed-only retrieval with closure-based retrieval to measure evidence recovery.
- Compare CCR's packing with flat presentation and generic compression over the same expanded candidate pool.
- For the presentation-only comparison, hold selected record IDs fixed and change grouping/order. Otherwise selection changes can masquerade as a chaining effect.

This clarifies the existing ablations rather than requiring a large experiment suite. Generic compression must receive the slot, date, and source information available to CCR when the purpose is to test whether its processing can exploit those relationships.

**Specific weakness: re-encoding changes the evidence before the read mechanism is evaluated.**

The revised encoder both adds slots and removes the previous tendency to resolve competing assertions. Its record text and coverage may therefore differ from the original ledger. Improvements over the historical 69.0 baseline cannot automatically be assigned to closure or presentation.

**Concrete method-level fix:** Run the read-method comparisons on one frozen CCR-compatible ledger, with the same text index and initial ranking. Include plain record retrieval over that ledger. Compare the original pipeline separately as an end-to-end baseline. This distinguishes better assertion retention during writing from better evidence use during reading.

**Specific weakness: Claim 2 currently tests a deliberately impoverished view.**

Keeping only the latest entry from a broad event thread destroys earlier events even when they are complementary, not contradictory. Historical and contradiction questions then predictably suffer. This does not isolate the location of reasoning or simulate the named systems faithfully.

**Concrete method-level fix:** Label latest-only as a lossy ablation. Either narrow the claim to the effect of suppressing prior assertions, or compare against write-time validity annotations with historical assertions still retained and accessible. The latter is necessary if the paper keeps a general claim about write-time versus read-time adjudication.

**Specific weakness: the reader prompt and success criteria need tighter control.**

CCR adds an instruction about possible updates and contradictions. On BEAM, where contradiction handling includes response behavior, that instruction could explain some improvement independently of grouping.

**Concrete method-level fix:** Use common task instructions across conditions where possible. Treat any claim about chain completeness as metadata that must be truthful. Do not let one condition alone receive the instruction needed to satisfy the contradiction rubric.

Keep all original anchor thresholds explicit. Beating record retrieval and RTC on BEAM-100K does not establish the required improvement over full context. On BEAM-1M, exceeding 63.3 strengthens the earlier target but should not replace comparison with Hindsight and raw retrieval.

Three generations and conversation-clustered comparisons are improvements. With only two BEAM-1M development histories, however, clustered uncertainty remains weakly estimated. Repeated generations do not supply independent histories; broader generalization must ultimately come from the existing untouched test split.

**4. Venue Readiness — 4/10. Priority: IMPORTANT.**

**Specific weakness: the new diagnostic is suggestive, but its causal interpretation remains unverified.**

The reported 10 missing-side rubric items, 10 reader-resolution items, and 12 retrieval items identify a useful target. They are not equivalent to recoverable overall-score points. Several rubric items can depend on the same question and evidence.

A missing assertion also does not, by itself, establish that the encoder's disambiguation state caused its omission. Likewise, a reader choosing one side does not establish that spatial separation caused the error; instructions or temporal interpretation may explain it.

**Concrete method-level fix:** Attach source-level examples and denominators to this diagnostic, and connect each mechanism to its corresponding comparison: assertion retention, closure recovery, and presentation with identical evidence. The planned ablations can provide this evidence.

The proposed integration point also needs correction. CCR can be added to an extractor only if the underlying assertions survive and receive usable keys. A read-time wrapper cannot recover records that a destructive writer never retained. Describe the required write contract explicitly.

Reader errors after receiving a chain must remain end-to-end method failures. They cannot be dismissed as outside memory when the anchored objective is to make this fixed reader use historical information correctly.

The next readiness milestone is a frozen slot policy, a truthful bounded packing rule, and development evidence against a history-aware comparator. No premature test-set access is needed.

**Simplification Opportunities**

1. Remove the endpoint-based overflow fallback from the first implementation. Admit complete chains and report exclusions; introduce partial-history selection only if observed failures require it.
2. Keep latest-only as a supporting ablation, rather than a second headline contribution or a proxy for Graphiti/Mem0.
3. Keep the existing encoder, retrieval engine, and reader. Resolve the slot interface before adding structure, routing, or additional model passes.

**Modernization Opportunities**

**NONE requiring a different model or training paradigm.** The current allocation of semantic labeling to an LLM and exact expansion to code is appropriate. Stable structured IDs and validated output schemas improve the existing interface; they do not require a new model component.

**Drift Warning**

**NONE in the revised objective.** The original anchor is preserved, and the proposed revisions remain within its scope.

There is a scope risk if evaluation shifts to only contradiction, ordering, and update questions. Those categories may explain the mechanism, but they cannot replace the overall benchmark and the original summarization, extraction, temporal, and budget requirements. Keeping the old pipeline for other abilities is acceptable only if the shared packing policy does not silently displace their evidence.

**Remaining Action Items**

1. **CRITICAL:** Specify stable slot identity, bounded registry exposure, record membership, and ambiguous cases.
2. **CRITICAL:** Resolve the incompatibility between complete chains and the overflow fallback; state and implement the actual conditional guarantee.
3. **CRITICAL:** Correct the literature comparison and distinguish CCR from history-preserving temporal retrieval, especially TRACE and chronological event composition in LeanMem.
4. **CRITICAL:** Separate write retention, closure, and presentation using shared ledgers and appropriately shared evidence; relabel latest-only.
5. **IMPORTANT:** Preserve every anchor success condition, count registry input tokens as well as calls, and validate the new diagnostic with source-level evidence.

**Verdict: REVISE.**

CCR is now a focused candidate worth investigating. Its architecture does not need expansion. Its preservation guarantee, semantic-key interface, and comparison with existing temporal memory need substantial revision before it supports a distinct method-first paper.

No files were edited and no experiments were run.

Thread ID: `01a142f5-dd49-7931-bb2f-fa83887f262a`

</details>
