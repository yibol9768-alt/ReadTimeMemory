# Round 2 Review（Codex，gpt-6.1-sol，Pro；2026-10-16 21:32 ET）

- 总分 6.75/10，结论 REVISE
- 分项：问题忠实 8，方法具体 6，贡献质量 6，前沿利用 8，可行 8，验证聚焦 6，投稿就绪 5
- 线程 id：01a14645-3421-7291-b36e-a3e79ff72525（续用）

<details>
<summary>原文</summary>

Thread id: 01a14645-3421-7291-b36e-a3e79ff72525. It was resumed, not started fresh. The run used model gpt-6.1-sol at xhigh effort and was read-only.

I did not set up or check how the call was authenticated (Pro direct via ~/.codex, no proxy on 18319). The call went through the Codex companion script, and I didn't verify the 18319 point.

Codex's full final review text:

**Verdict: REVISE. Overall: 6.75/10** (previously 6.35).

Dropping the unsupported CTR mechanism claim is the right response to the controls. The revised direction is more credible scientifically, but its strongest claims still exceed the evidence. The promising contribution is **how answer prompts and retrieval budgets interact with memory pipelines**, rather than a finding that storage representation does not matter.

I read the prior review, revised proposal, relevant progress entries, and selected implementation details. No files were modified.

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 8/10 | Application preserved; contribution and success criteria materially reframed. |
| Method Specificity | 6/10 | Reading strategy is concrete; decomposition does not yet cleanly isolate its named factors. |
| Contribution Quality | 6/10 | Useful findings, but substantial overlap with MemDelta and insufficiently established interaction claims. |
| Frontier Leverage | 8/10 | Appropriate use of API models and existing representations; no additional model machinery needed. |
| Feasibility | 8/10 | Most required controls can reuse frozen memories and answers. |
| Validation Focus | 6/10 | Better comparisons, but equivalence claims, limited independent histories, and single-generation findings remain problems. |
| Venue Readiness | 5/10 | A plausible empirical-paper direction, currently supported by exploratory evidence. |

Weighted overall: \(0.15(8)+0.25(6)+0.25(6)+0.15(8)+0.10(8)+0.05(6)+0.05(5)=6.75\).

**Problem Anchor and drift warning**

The deployment problem is preserved: answering point and broad questions over long conversational histories with a fixed reader and limited context.

However, this is a **declared research reframing**, not merely a narrower version of the original claim. A new resolution-selection mechanism and significant superiority over Hindsight have been replaced by controlled analysis and a proposed accuracy–context tradeoff. Those original success conditions remain unmet or have been retired.

Also remove the anchor's repeated premise that existing systems use only one resolution. Round 1 already established that this premise is inaccurate. The new evidence motivates reader–retrieval interaction as the question to investigate.

**Method Specificity — 6/10**

**Weakness:** Equal context length does not isolate *storage representation*. Hindsight and Lychee differ in extraction, candidate selection, ranking, provenance, and formatting. The implementation also includes source excerpts and raw-dialogue matches in the Lychee views. These are comparisons between retrieved-context pipelines, not pure narrative-versus-atomic encoding interventions.

**Concrete fix — CRITICAL:** Either rename this factor "memory pipeline under matched reader budgets," or hold source evidence, retrieval, ordering, and provenance constant while changing its representation. The first option is the smaller adequate revision.

The prompt intervention likewise changes a package of factors: instructions, system/user placement, and question position relative to memory. It establishes prompt-package sensitivity; it does not isolate the instruction to scan and cross-reference memories. The relevant branches are visible in [beam.py](/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/mem/beam.py:646).

**Concrete fix — IMPORTANT:** Publish the exact templates and describe the intervention accurately. Add a placement-normalized control only if explaining the instructional mechanism is central.

The notes comparison also needs consistent conditions: **72.7 is notes-only**, whereas **73.8 is notes plus original context under the alternative prompt**. The log contains matched conditions, so this can be repaired without inventing another component.

**Contribution Quality — 6/10**

**Weakness:** "Controlled comparisons reveal confounds in memory-system scores" is already the central contribution of **MemDelta**. It studies model-dependent ranking reversals, embedding effects, and write-path cost on LongMemEval-S across three reader families. It explicitly distinguishes controlled probes from an additive causal decomposition. [MemDelta paper](https://arxiv.org/html/2606.29914v1)

Moving that protocol to BEAM is useful, but insufficient differentiation by itself.

**Concrete fix — CRITICAL:** Make the primary question:

> Under what prompt and retrieval-budget conditions do comparisons between conversational-memory pipelines change, and do those changes persist across readers?

Your potential distinction is an explicit **pipeline × prompt × budget interaction study at 100K–1M scale**, supported by evidence about which answers change and why. This goes beyond another collection of isolated component deltas.

The claim that public score differences "mainly" come from reader-side factors is currently too broad. Your intervention demonstrates a large effect in your harness. It cannot attribute differences between published results that used other readers, judges, or system versions.

Similarly, limited stacking of notes and prompts does not establish that they solve the same underlying failure. A matched interaction contrast and a small examination of changed answers would make this explanation credible.

**Is decomposition a credible ICML main-track primary contribution?**

**Potentially yes; the current version is not yet strong enough.** An empirical paper can stand on a consequential, reproducible finding without a novel memory architecture. Here, the distinguishing result must be more precise than "prompts matter" or "simple retrieval can compete."

A defensible claim would be:

> Memory-pipeline rankings depend on interactions between reader instructions and retrieval budget; one shared answer prompt is therefore insufficient to establish a stable architectural advantage.

That claim must survive the targeted checks below. Adding a stronger method is not a necessary rescue.

**Validation Focus — 6/10**

**Weakness:** Several conclusions confuse failure to detect a difference with evidence of equivalence.

| Observation | Supported interpretation |
|---|---|
| Coverage allocation: 68.9 versus 68.8 | This pilot found no benefit supporting the CTR allocation claim. |
| Four broad-question conditions lack significant differences | Differences remain unresolved on this sample. |
| 1M method difference: +1.4, interval [−4.0, +5.8] | Both a meaningful loss and a meaningful gain remain compatible with the evidence. |
| Notes add little under the alternative prompt | No reliable incremental benefit has yet been established. |

**Concrete fix — CRITICAL:** Predefine a meaningful noninferiority or equivalence margin. The current 1M interval does **not** establish noninferiority at the previously discussed −2-point margin. Until confirmed, say "similar point estimates with substantially less reader context," rather than "equal accuracy."

**Weakness:** The proposal describes three-generation measurement for every factor, but the central prompt and notes findings are still single-generation pilots. Five independent 1M conversations remain a small basis for generalization.

**Concrete fix — IMPORTANT:** Separate exploratory results from confirmed comparisons. Confirm the primary interactions on untouched conversations, average repetitions within questions, and retain conversation-level paired uncertainty. More generations cannot replace more independent histories.

**Venue Readiness — 5/10**

**Weakness:** The paper currently combines an overgeneralized decomposition with an unconfirmed noninferiority claim and incomplete cost accounting.

**Concrete fix — IMPORTANT:** Freeze a narrower analysis protocol, confirm its central finding, and report actual reader tokens alongside total online input/output tokens, calls, and amortized write cost. The approximately **1/5 reader-context ratio** is a useful measured quantity; it is not yet a fivefold total-cost claim.

**Should the method half remain?**

**Keep the strategy, but merge it into the analysis as a practical baseline and consequence.** Point/area routing, deeper retrieval, and conflict checking do not presently justify a separate algorithmic contribution.

Its essential controls are small:

- A uniform budget at approximately the same average reader-token cost.
- The same point/area budget policy applied to Hindsight.

If these also work, the result supports a general reading-policy finding. If routing contributes no reliable advantage, drop it and retain the simpler budget baseline. The paper should not need another named method.

**Minimal additional evidence / remaining action items**

1. **CRITICAL — Confirm the interaction.** Freeze a small crossed comparison of two pipelines, two prompts, and two matched token budgets on held-out BEAM conversations. Repeat the central contrasts with a genuinely different reader family while keeping compiled memories fixed.
2. **IMPORTANT — Check judging.** Rescore the same frozen answers with an independent judge family. Manually inspect a small, balanced sample of disagreements and prompt-induced gains/losses, particularly abstention and temporal answers.
3. **IMPORTANT — Establish the claimed tradeoff.** Test a predefined noninferiority margin against a budget-tuned Hindsight baseline, including the two routing controls above. If uncertainty remains wide, retain descriptive accuracy–context curves.
4. **IMPORTANT — Check transfer selectively.** LongMemEval-S is appropriate for the central prompt/budget interaction. The 62-question development subset is a pilot; confirm on its untouched portion before claiming generality beyond BEAM. Replicating every intervention there is unnecessary.
5. **IMPORTANT — Complete the audit.** Distinguish one- and three-generation results, publish exact prompts and intervention boundaries, and count write and online costs consistently.

**Simplification Opportunities**

1. Use one dominant contribution: prompt–budget–pipeline interactions.
2. Merge the reading strategy into the analysis; retire CTR/STT mechanism claims.
3. Keep notes and conflict checking as supporting probes unless their interactions become central confirmed findings.

**Modernization Opportunities: NONE.** The missing work is sharper experimental identification and confirmation, not additional foundation-model components.

**Drift Warning:** The conversational-memory task is preserved. The paper's contribution and success criterion have materially changed; acknowledge that reframing explicitly. **Verdict: REVISE.**

Relevant files:
- /mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/docs/refine2/codex_round_2_review_bundle.md
- /mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ReadTimeMemory/mem/beam.py

</details>
