# Round 1 Review（Codex，gpt-6.1-sol，Pro；2026-10-16）

- 总分 5.50/10，结论 RETHINK
- 分项：问题忠实 8，方法具体 5，贡献质量 3，前沿利用 8，可行 7，验证聚焦 5，投稿就绪 3
- Codex 会话 id：01a142f5-dd49-7931-bb2f-fa83887f262a

<details>
<summary>原文</summary>

The proposal identifies a useful engineering intervention, but the current RTC mechanism is insufficiently distinct from existing query-conditioned context compression to support a method-first top-venue paper. The main revision should establish one conversational-memory-specific mechanism, while preserving the proposal's restrained architecture.

I reviewed the Problem Anchor first, followed by the full proposal, the supporting progress entries, the method tree, the relevant survey entries, and primary sources for the closest related work. The reported experimental numbers below are development results from the supplied documents; I did not independently reproduce them.

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 8/10 | Preserves the fixed reader, long conversational histories, short final context, and API-only constraints. The causal motivation is stronger than the evidence supports. |
| Method Specificity | 5/10 | The pipeline is implementable as a prototype, but its evidence representation, cross-block behavior, and budget enforcement remain underspecified. |
| Contribution Quality | 3/10 | The dominant mechanism is presently generic query-conditioned map-reduce compression applied to conversational memory. |
| Frontier Leverage | 8/10 | Using the same foundation model as an evidence compiler is appropriate. Training, RL, VLMs, and diffusion are unnecessary here. |
| Feasibility | 7/10 | Existing infrastructure makes a pilot practical. Reliable merging and actual API cost need resolution before the larger claims are credible. |
| Validation Focus | 5/10 | The proposed comparisons address useful questions but omit the decisive generic-compression control and do not adequately separate mechanism from computation. |
| Venue Readiness | 3/10 | The research question is worthwhile; the current novelty argument and claim-to-evidence alignment are not submission-ready. |

**OVERALL SCORE: 5.50/10**

Weighted calculation: 0.15(8)+0.25(5)+0.25(3)+0.15(8)+0.10(7)+0.05(5)+0.05(3)=5.50.

**1. Method Specificity — 5/10. Priority: CRITICAL.**

**Specific weakness:** The unresolved part is precisely where the claimed memory capabilities should reside: converting independent local extractions into a globally useful, bounded evidence sheet.

The [proposed algorithm](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine/round-0-initial-proposal.md:80) sorts candidates by retrieval rank, divides them into independent blocks, extracts locally relevant facts, and merges by date. This does not specify how to preserve facts whose relevance becomes apparent only across blocks.

For example, one block identifies Mira as the person the user met in Kyoto; another records Mira's preferred hobby. For a question about "the hobby of the person I met in Kyoto," the second compiler lacks the binding that establishes relevance. A later merge cannot recover a fact already discarded as NONE. Short inputs reduce exposure to long-context difficulties; they do not establish that independent extraction preserves all necessary reasoning evidence.

Likewise, "retain both sides of updates or contradictions" is only a local prompt instruction. If the two statements occur in different blocks, no compiler sees the pair. Both might survive independently, but the algorithm provides no preservation rule ensuring this.

The overflow rule is especially problematic. Truncating by block rank can retain an older statement while removing its correction. Once earlier blocks fill the final budget, increasing retrieval width may contribute nothing to the reader. This directly undermines the proposed computation-scaling mechanism.

Additional ambiguities affect implementation:

- Does "date" mean utterance time, event time, or a generated normalization? These are different for temporal questions.
- What counts as a duplicate when a record paraphrases a raw utterance? Collapsing similar assertions can erase differences in time, speaker, negation, or certainty.
- Is the 8K-character limit a hard serialization limit including dates, IDs, and metadata?
- How are turns, records, and source references preserved at block boundaries?
- An existing source ID does not establish that the compiler's generated assertion is supported by that source.
- Local NONE means no evidence was selected from a block; it does not establish that the complete history lacks an answer.

**Concrete method-level fix:** Specify one evidence representation and one complete packing algorithm. Evidence units should retain speaker, source identity, an exact supporting span, utterance time, and explicitly supported event time when available. Missing temporal information should remain unknown. Validate source references and copied spans mechanically.

Replace rank-prefix truncation with a specified selection rule that preserves required evidence groups, including corrections and their antecedents. Define how the compiler conservatively retains possible cross-block dependencies. If a bounded global pass is necessary, make it an explicit part of this same compiler and include its calls and tokens in the cost.

This replaces the underspecified merge; it does not require a separate memory graph, learned router, or agent loop. It also does not automatically establish novelty.

**2. Contribution Quality — 3/10. Priority: CRITICAL.**

**Specific weakness:** The asserted technical gap is not a gap in the broader literature. Query-dependent processing of retrieved material before a reader consumes it is already established.

My novelty assessment from the primary sources is:

| Closest work | Established mechanism | Consequence for RTC |
|---|---|---|
| [Chain-of-Agents](https://arxiv.org/abs/2406.02818) | Segmented reading with sequential worker communication and a final manager. | RTC is not identical: its workers are independent and parallel. However, distributing long-context processing across short calls is already established. |
| [ReadAgent](https://arxiv.org/abs/2402.09727) | Gist memories associated with original passages, followed by task-dependent lookup. | Retaining source material and accessing it according to a question are not distinguishing contributions. |
| [RECOMP](https://arxiv.org/abs/2310.04408) | Compresses retrieved documents before reader integration, including selective empty output for irrelevant material. | A short query-relevant representation and NONE behavior substantially overlap with prior compression objectives. |
| [CompAct](https://arxiv.org/html/2407.09014v2) | Compresses retrieved segments into compact query-relevant context, integrates information across segments, and studies benefits from larger retrieval sets. | Separating retrieval breadth from reader burden, including wider-retrieval performance curves, is already a closely related objective. |
| [LongRefiner](https://arxiv.org/abs/2505.10413) | Query-aware refinement of long retrieved documents using hierarchical structure and a trained foundation model. | RTC's training-free implementation is a practical distinction, but query-aware refinement itself is not new. |

The differences matter: RECOMP and LongRefiner use trained components, CompAct maintains sequential compressed context, and RTC proposes independent parallel extraction. I would not describe RTC as a literal reproduction of any one of them. As written, however, it is a straightforward member of the existing compression/map-reduce family.

The conversational setting, immutable inputs, timestamps, controlled comparisons, and width curves do not by themselves introduce a new algorithm. [LeanMem](https://arxiv.org/abs/2608.03463) already combines immutable source-grounded records with evidence assembly according to query-specific demands, further weakening these features as sufficient differentiators.

The "no drift" argument also overreaches. Rebuilding from immutable sources avoids recursively overwriting previous generated memory. It does not eliminate initial extraction errors, changing retrieval results, or query-time compilation errors. Preserving raw episodes as first-class evidence is already advocated by [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978).

**Concrete method-level fix:** Make the dominant contribution a precise conversational-memory operation that ordinary compression lacks. A plausible candidate is preserving correction and evidence dependencies across blocks under a fixed reader budget. Specify what must survive, how dependencies are represented, and how packing handles them.

Then compare that operation against ordinary query-conditioned map-reduce with the same sources, model, block sizes, final budget, and computation. The distinguishing result must be that the memory-specific operation fixes a failure the generic compiler retains.

This is a candidate research direction, not a certification of novelty. Timestamps, source IDs, and a "retain both" prompt alone remain insufficient. If no distinct operation emerges, RTC should remain a strong baseline rather than be presented as a new memory method.

**3. Validation Focus — 5/10. Priority: CRITICAL.**

**Specific weakness:** The validation currently establishes whether extra query-time processing helps more directly than it establishes why RTC is a new and effective mechanism.

The same-recall comparison is valuable and should remain. However, comparing one direct reader call with several compiler calls plus a reader jointly changes decomposition, evidence selection, context length, and computation. It cannot isolate the proposed contribution.

**Concrete method-level fix:** Center validation on three comparisons using frozen candidate evidence:

1. **Direct reading versus generic compilation:** Establish whether broad evidence benefits from a short compiled representation.
2. **Generic compilation versus the proposed memory-specific compiler:** Establish the contribution under the same final reader budget and comparable computation.
3. **The resulting method versus the strongest existing memory baselines:** Establish that the mechanism improves the anchored task.

These comparisons can share conditions; they do not require a large new benchmark suite.

Several protocol details are essential:

**Freeze evidence for the mechanism comparison.** Reuse the same planning output, candidate set, and block partition across relevant conditions. Otherwise retrieval changes can masquerade as compiler improvements. Measure complete-pipeline resampling separately where appropriate.

**Match computation meaningfully.** Equal call counts do not imply equal costs: repeatedly reading 40K characters differs substantially from five calls each reading 8K characters. Self-consistency is therefore an incomplete control. The generic compiler with the same topology is the more decisive control.

**Report actual input and output tokens.** The proposal defines W, S, and the final sheet in characters but compares them with a full-context budget in tokens. The supporting progress table reports about 661K characters for the BEAM-100K full-context condition. The claimed "roughly half the input" does not follow from the supplied quantities. Include compiler outputs, repeated questions and instructions, planning, final-reader usage, and any global merge call. Report latency under a stated concurrency limit.

**Keep the strongest existing record baseline on BEAM-1M.** The reported scores are 63.3 for Lychee-lite and 58.6 for Hindsight. A target of Hindsight +5 gives 63.6, effectively the existing record baseline's level. Clearing that target alone would provide little evidence of RTC's value over the current pipeline.

**Treat width as an empirical tradeoff.** Wider retrieval is neither guaranteed to improve accuracy nor guaranteed to preserve it under a fixed output budget. Use nested candidate sets where possible, and relate gains or regressions to evidence retained in the final sheet. Do not make monotonic improvement a presumed property.

**Keep uncertainty tied to independent histories.** Three generations help estimate generation variability, but do not create new conversational histories. The development sets contain six BEAM-100K histories and two BEAM-1M histories. Use paired comparisons and acknowledge clustering; establish generalization on the existing untouched test split after freezing the method.

Finally, the write-time/read-time comparison changes query availability as well as timing. That is a legitimate design tradeoff, but not proof that write-time compilation inherently drifts. Either narrow this claim to query-conditioned versus query-independent compression, with costs amortized over queries, or remove it from the central validation story.

**4. Venue Readiness — 3/10. Priority: IMPORTANT.**

**Specific weakness:** The proposal's motivating observations are useful, but several interpretations are presented more strongly than their designs justify.

The [supporting error analysis](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/PROGRESS.md:776) explicitly states that all full-context losses are classified as reader losses by definition. Consequently, the 25.6-point number does not independently demonstrate long-context overload. It can include reasoning failures, temporal ambiguity, abstention behavior, and grading disagreement.

Similarly, H33 changes context composition under a fixed budget. Adding source passages and neighboring turns can displace records and change evidence density. The −9.7 result supports rejection of that particular source-expansion design; it does not isolate the causal effect of giving the reader more information.

For record retrieval, the supplied attribution assigns 15.7 points to write loss and 9.6 to reader loss. Thus, "the main problem is use rather than storage" is too categorical across the very baselines motivating the proposal. The more defensible motivation is a tradeoff between evidence coverage and reader usability.

**Concrete method-level fix:** Align the proposal around one testable statement: a specified compiler preserves more answer-relevant conversational evidence within the same final budget than generic compression does. Audit the path from source evidence to selected evidence to answer on paired failures. Separate missing evidence from incorrect interpretation, and treat abstention cases separately from positive factual evidence.

Remove "no drift," guaranteed short-context reliability, and monotonic scaling as established properties. Replace language claiming the mechanism "proves" superiority with proposed hypotheses until those comparisons exist.

For this early-stage proposal, I am not requesting premature test-set evaluation. The next readiness milestone is a frozen algorithm and a development result against the generic compiler that supports the claimed distinction.

**Simplification Opportunities**

1. Keep one compiler contribution. Treat loss attribution as supporting analysis and remove write-time superiority and drift prevention as separate headline claims.
2. Replace the current merge/truncation rule with one specified evidence-preservation rule. Do not restore the earlier collection of structure layers, routers, or self-check loops.
3. Make the same-model generic map-reduce comparison decisive. Defer broad wrapper demonstrations across multiple memory systems until the core mechanism survives that comparison.

**Modernization Opportunities**

**NONE requiring a new model family or training paradigm.** The proposal already uses foundation-model capabilities appropriately. Structured evidence output and mechanically checked source spans would strengthen the existing compiler. RL, VLMs, diffusion, additional agents, and model training would add complexity without resolving the central novelty question.

**Drift Warning**

**NONE for the current task formulation or the recommended revisions.** The fixed reader, API-only implementation, short final evidence sheet, and full range of anchored conversational capabilities can remain intact.

A revision that evaluates only factual extraction or knowledge updates while dropping summarization, temporal reasoning, or ordering would narrow the Problem Anchor and must be identified as problem drift. The proposed evidence-preservation mechanism must therefore support those broader capabilities rather than obtain a cleaner result by excluding them.

**Verdict: RETHINK.**

The anchored problem deserves continued work, and RTC is worth implementing as a baseline. The framing of generic query-conditioned map-reduce compression as a new conversational-memory mechanism requires rethinking. A focused memory-specific preservation operation could justify a revised proposal; additional modules or a larger benchmark checklist would not resolve the current issue.

No files were edited.

Codex thread/session ID: `01a142f5-dd49-7931-bb2f-fa83887f262a`

</details>
