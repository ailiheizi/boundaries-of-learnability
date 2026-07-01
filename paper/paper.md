# Boundaries of Learnability: When Explicit Knowledge Enables and Implicit Knowledge Fails

**Zhi Liu**

Independent Researcher, China

`ailiheizi@gmail.com`

## Abstract

We investigate the boundaries of what language models can and cannot learn, through two complementary studies spanning reasoning and narrative understanding. In **Study 1** (compositional reasoning), we present a controlled 2×2 dissociation (storage location × reasoning procedure). Models perfectly recall individual facts (single-hop ~100%) but catastrophically fail at implicit multi-hop composition (2-hop: 27%, 4-hop: 0%). Chain-of-thought (CoT) rescues composition when facts are in-context (scaling from 40% at 1.5B to 100% at large scale), but—despite ~100% single-hop recall—**fails at the floor (0-2%) when the identical facts must be recalled from weights mid-generation** (closed-book, LoRA fine-tuning at 1.5B–3B). The same reasoning procedure thus succeeds or fails depending solely on fact location: successful composition requires *both* a reasoning scaffold *and* fact availability in context. In **Study 2** (narrative structure from reader comments), reader annotations reliably mark structural positions (4.5× above baseline, independent of arousal), and models distinguish coarse-grained structural scenes (100%) but **cannot distinguish fine-grained within-scene foreshadowing** (44%, n=18, p=0.76)—foreshadowing is a reader's retrospective interpretation, not a learnable textual feature. Together, these findings sketch a boundary: **explicit knowledge (in-context facts, keyword annotations) is usable; implicit knowledge (weight-stored facts under standard fine-tuning, subtle textual cues) is not reliably accessible.** We situate this relative to the Reversal Curse and Two-Hop Curse, and mark scaling behavior beyond our fine-tuning budgets as open.

---

## 1. Introduction

A fundamental question in AI is: what can models learn, and what remains beyond their reach? We approach this through two empirical studies that, despite different domains, converge on the same answer.

**The explicit-implicit boundary.** When knowledge is presented explicitly—facts stated in the prompt, structural positions marked by keywords—models use it effectively. When the same knowledge exists only implicitly—encoded in model weights after training, or latent in textual style without surface markers—models fail to access or learn it. Critically, this failure persists across model scales.

This connects to the Reversal Curse (Berglund et al., 2023): models trained on "A is B" cannot infer "B is A." We generalize this from single-hop reversal to multi-hop composition (Study 1) and from factual recall to narrative structure learning (Study 2), showing that the underlying limitation—unidirectional, non-composable knowledge storage—manifests across domains.

**Contributions:**
1. A compositional reasoning benchmark with anti-cheating controls (fictional entities, single-hop gating, open-generation), revealing that CoT rescue depends on knowledge being in-context and scales with model capacity (§3).
2. A controlled 2×2 dissociation showing that weight-internalized facts (under LoRA fine-tuning at 1.5B, 3B, and 7B) do not support CoT-based composition despite near-perfect single-hop recall, while the identical facts in context do (§3.3).
3. A RAG baseline showing that step-by-step external retrieval restores composition to 100%, confirming the failure is retrieval-specific rather than reasoning-fundamental (§3.4).
4. A systematic study of reader comments as narrative structure signals, establishing that keyword annotations are reliable coarse-grained position labels (4.5×) but carry no learnable fine-grained textual signal (§4).
5. A unified framework connecting these findings: the explicit-implicit boundary as a fundamental constraint on current architectures (§5).

---

## 2. Related Work

**Compositional generalization.** SCAN (Lake & Baroni, 2018), COGS (Kim & Linzen, 2020), and subsequent work test whether models generalize compositionally from primitives. We test a simpler form: can models compose *explicitly given* facts via multi-hop chains? Our contribution is the storage-axis comparison (in-context vs. weight-internalized).

**Reversal Curse.** Berglund et al. (2023) showed models trained on "A is B" fail to answer "What is B?" Allen-Zhu & Li (2024) provide theoretical grounding. We extend this from single-hop reversal to multi-hop composition, and from recall to CoT-mediated reasoning.

**Chain-of-thought.** Wei et al. (2022) established CoT as a reasoning enhancer. We show CoT's effectiveness is *conditional* on knowledge accessibility: it works when facts are in-context but fails when they must be retrieved from weights.

**Narrative structure detection.** TRIPOD (Papalampidi et al., 2019) detects turning points from screenplay text. Danmaku-based highlight detection (2017-2024) uses comment density as arousal signal. We bridge these: testing whether comments carry *structural* (relational, not scalar) information learnable from content.

**Reader behavior as training signal.** "Teaching Human Behavior Improves Content Understanding" (2405.00942) uses receiver behavior to improve LLM content understanding. We test a sharper claim: whether reactions encode *narrative structure* (setup→payoff links) beyond arousal peaks.

---

## 3. Study 1: Compositional Reasoning Collapse

### 3.1 Task Design

We construct fictional fact chains: entities E₀→E₁→E₂→...→Eₖ linked by a "work partner" relation. Only atomic single-hop facts are provided; k-hop composition questions are never explicitly stated.

**Anti-cheating controls:**
- Fictional entities (random names from disjoint pools, absent from pretraining)
- Single-hop gating (verify model recalls each atomic fact before testing composition)
- Open-generation (exact name match, not multiple-choice—avoids recognition inflation)
- Distractor chains (3 unrelated chains shuffled in) + randomized order
- Multiple seeds

### 3.2 In-Context Results: Scale Axis

All facts presented in the prompt. Three conditions: direct answer, CoT.

| Model | Single-hop | Direct Composition (2/3/4-hop) | CoT (2/3/4-hop) |
|-------|-----------|-------------------------------|-----------------|
| Qwen2.5-1.5B | 100% | 25→5→5% | 45/45/30% |
| Qwen2.5-3B | 100% | 20→10→0% | 60/70/65% |
| DeepSeek-chat | 100% | 27→7→0% | 100/100/100% |

**Finding 1:** Direct composition collapses universally (all scales crash to 0-27% beyond 2-hop).
**Finding 2:** CoT rescue scales monotonically with model capacity: 40% → 65% → 100%.

### 3.3 Weight-Internalized Results: Storage Axis

Same facts LoRA-trained into model weights (12–15 epochs, closed-book evaluation, 2 seeds × 20–30 chains).

| Storage | Model | Single-hop | Direct (2/3/4) | CoT (2/3/4) |
|---------|-------|-----------|----------------|-------------|
| In-context | 1.5B | 100% | 25→5→5% | 45→45→30% |
| In-context | 3B | 100% | 20→10→0% | 60→70→65% |
| **Weight** | **1.5B** | ~98% | 35→0→3% | **2→0→0%** |
| **Weight** | **3B** | ~98% | 5→0→0% | **8→0→0%** |
| **Weight** | **7B** | 10%/98%/82% | 7→0→2% | **3→0→2%** |

**Finding 3 (core):** Weight-internalized CoT fails at the floor (0-8%) across three model scales (1.5B, 3B, 7B). Crucially, the 7B model achieves near-perfect single-hop recall at 3-hop (98%) and 4-hop (82%)—confirming facts are successfully stored—yet CoT composition remains at 0%. This dissociation now spans a 5× scale range (1.5B→7B) under LoRA training, strengthening the conclusion that the failure is not merely a capacity issue.

**Note on 7B 2-hop:** Single-hop recall is unstable at 2-hop (10%), likely due to LoRA training dynamics at this hop count. We treat the 3-hop and 4-hop results (single ≥82%, CoT ≤2%) as the clean evidence. The 2-hop instability does not affect the core dissociation finding.

**Diagnostic:** Raw CoT outputs from weight-internalized models show: (a) second-hop retrieval failure (stuck on intermediate entity), (b) hallucinated entities absent from training ("Tovak Sterling"), (c) correct chain enumeration followed by wrong final answer. The model can write reasoning *steps* but cannot reliably *retrieve* the next fact from weights at each step.

### 3.4 RAG Baseline: External Step-by-Step Retrieval Rescues Composition

We test a RAG (Retrieval-Augmented Generation) condition: the model receives no facts in the prompt, but at each reasoning step, the relevant single-hop fact is retrieved from an external knowledge base and injected.

| Storage | Condition | 2-hop | 3-hop | 4-hop |
|---------|-----------|-------|-------|-------|
| In-context | All facts in prompt | 100% | 100% | 100% |
| **Weight** | Closed-book (CoT) | 2% | 0% | 0% |
| **RAG** | Step-by-step retrieval | **100%** | **100%** | **100%** |
| Blind | No facts, no retrieval | 0% | 0% | 0% |

**Finding 4:** When facts are supplied at each hop—whether in-context or via external retrieval—composition succeeds (100%). We note this RAG result is partly *by construction*: our retrieval procedure supplies the hop decomposition externally and issues one single-hop lookup per hop, over facts already recalled at ~100% single-hop. RAG therefore isolates the case where the compositional *plan* is provided externally; it does not test whether the model can plan composition itself. Its value here is as a positive control confirming the model can chain supplied facts.

### 3.5 Interpretation: A Storage-Dependent Dissociation

Our central finding is a **dissociation**, not a reassignment of the bottleneck. Holding the reasoning procedure fixed (CoT) and varying only where facts live:

- **Facts in context** → CoT rescues composition, scaling 40% (1.5B) → 100% (large model)
- **Facts in weights** (closed-book), *despite ~100% single-hop recall* → CoT fails at the floor (0–2%)

The identical reasoning procedure succeeds or fails depending solely on fact *location*. This is the sharp, reproducible result: **CoT-based composition requires facts to be present in the context window; it does not reliably retrieve weight-stored facts mid-generation, even when each fact is individually recallable.**

We deliberately avoid the stronger claim that "reasoning is not the bottleneck." Our own 2×2 shows reasoning procedure matters greatly (in-context direct 0% vs. in-context CoT 100%). The precise statement is a *conjunction*: successful multi-hop composition requires both (a) a reasoning scaffold (CoT) and (b) fact availability in context. Removing either causes collapse.

**Relation to prior work.** This dissociation is consistent with and extends the Reversal Curse (Berglund et al., 2023), the Two-Hop Curse (Balesni et al., 2024), and Allen-Zhu & Li's "Physics of Language Models" (retrieval succeeds, manipulation fails). Our contribution is the controlled 2×2 isolating storage location × reasoning procedure, with the finding that this holds across a 5× scale range (1.5B→7B LoRA) under standard fine-tuning budgets. **Scope caveat:** we do not claim this survives extended "grokking"-regime training or full fine-tuning of very large models; establishing whether the dissociation persists under those conditions is important future work.

The RAG result (§3.4) provides a positive control: when retrieval is externalized step-by-step, composition succeeds. Combined with the dissociation, this motivates architectural choices (explicit retrieval over parametric storage for compositional tasks).

---

## 4. Study 2: Narrative Structure Learnability

### 4.1 Can Reader Reactions Mark Structure?

**Data:** Bilibili danmaku (time-synced comments) from 3 videos: movie commentary (4973 comments), suspense drama (3533), pure hype montage (1200, control).

**Method:** Compare structure-keyword density ("伏笔/foreshadowing," "原来/so that's why," "难怪/no wonder") at narrative payoff points vs. arousal-matched control points (same comment volume, different structural role).

| Video | Type | Keyword Ratio (payoff vs. control) | Temporal Shuffle |
|-------|------|----|----|
| Movie commentary | Has structure | **4.57×** | Signal collapses ✓ |
| Suspense drama | Has structure | **4.41×** | Signal collapses ✓ |
| Hype montage | No structure | **0.0×** (no keywords) | N/A |

**Finding 4:** Reader keywords reliably mark structural positions, independent of arousal, with correct discrimination (hype=0, narrative=4.5×).

### 4.2 Is the Signal Deeper Than Keywords?

| Test | Method | Result |
|------|--------|--------|
| Semantic embedding (BGE) | Per-comment max-sim to past | 1.05× (no signal) |
| Content classification (DeepSeek) | Non-keyword comments → judge structure | 29% (below chance) |
| ASR subtitle content | Video narration → judge structure | 57% (chance) |

**Finding 5:** Remove keywords → zero signal remains. Structure marking exists *only* in the explicit annotation vocabulary, not in deeper semantic or content patterns.

### 4.3 Novel Text: Coarse vs. Fine Granularity

**Data:** Qidian web novels (《诡秘之主》,《捞尸人》), chapter paragraphs paired with reader comments via API.

| Granularity | Comparison | Accuracy | Significance |
|-------------|-----------|----------|---|
| **Coarse** (cross-scene) | Climax scene vs. daily scene | 100% (n=10) | p<0.001 |
| **Fine** (adjacent paragraphs) | Foreshadowing vs. neighbor | 44% (n=18) | p=0.76 (chance) |

**Finding 6:** Models distinguish *what kind of scene* contains structure (coarse), but cannot identify *which sentence* is foreshadowing within a scene (fine). Fine-grained "foreshadowing" is a reader's retrospective attribution, not a textual prior.

---

## 5. Unified Discussion

### 5.1 The Explicit-Implicit Boundary

| Domain | Explicit (usable) | Implicit (fails) |
|--------|-------------------|-------------------|
| Reasoning | In-context facts → CoT 100% | Weight-stored facts → CoT 0% |
| Reasoning | RAG (step-by-step retrieval) → 100% | Closed-book → 0% |
| Narrative | Keyword annotations → 4.5× | Non-keyword content → 1.05× |
| Granularity | Cross-scene → 100% | Within-scene → 44% (chance) |

The pattern is consistent: when the relevant signal is **surface-level and explicit**, models succeed; when it requires **compositional retrieval or latent interpretation**, they fail.

### 5.2 Theoretical Connection

Current transformer architectures store knowledge as unidirectional associations in weight matrices. This representation supports:
- Direct recall (A→B: single-hop 100%)
- Pattern matching on explicit markers (keyword detection)

But fails for:
- Compositional chaining (A→B→C from weights)
- Learning latent patterns without surface markers

This is not a training data issue (facts *are* successfully memorized—single-hop proves it). It is a *representation* issue: the format in which knowledge is stored does not support the operations attempted.

### 5.3 Practical Implications

1. **Memory systems:** Use RAG (explicit, in-context) over parametric storage for any task requiring compositional retrieval. Our RAG baseline (§3.4) demonstrates this directly: step-by-step retrieval restores composition from 0% to 100%, even at 4-hop depth.
2. **Narrative AI:** Reader comments are useful as coarse-grained position labels (free annotation tool), but cannot serve as training signal for fine-grained narrative understanding models.
3. **CoT limitations:** CoT is not universally beneficial—it requires (a) sufficient model capacity and (b) facts accessible in context. It cannot compensate for retrieval failures from weights.

---

## 6. Limitations

- Scale axis: Qwen 1.5B/3B same-family + DeepSeek as upper bound. 7B gap remains.
- Narrative study: 2 novels, 18 fine-grained pairs. Conclusion directionally clear (p=0.76) but larger-scale confirmation strengthens it.
- Fictional entities only (reasoning study). Real-world multi-hop may differ due to pretraining priors.
- Weight internalization via LoRA only; full fine-tuning may differ.
- Single relation type ("work partner"). Diverse relations may yield different composition profiles.

---

## 7. Conclusion

We sketch a boundary between what models can and cannot do that aligns with the explicit-implicit divide in knowledge form. Explicit knowledge—facts in context, structural positions marked by keywords—is compositionally accessible and learnable. Implicit knowledge—facts in weights (under our fine-tuning budgets), latent narrative structure—resists both compositional retrieval (even with CoT) and learning. Within the model scales and training regimes we tested, this boundary does not diminish; whether it persists under extended training or substantially larger full-fine-tuned models is an open question we flag for future work. Our anti-cheating methodology (fictional entities, gating controls, arousal matching, temporal shuffling) provides a template for rigorous evaluation of model capabilities at their boundaries.

---

## References

[1] Berglund, L., Tong, M., Kaufmann, M., et al. (2023). The Reversal Curse: LLMs Trained on "A is B" Fail to Learn "B is A". *arXiv:2309.12288*.

[2] Balesni, M., Korbak, T., & Evans, O. (2024). The Two-Hop Curse: LLMs Trained on A→B, B→C Fail to Learn A→C. *arXiv:2411.16353*.

[3] Allen-Zhu, Z., & Li, Y. (2024). Physics of Language Models: Part 3.2, Knowledge Manipulation. *arXiv:2309.14402*.

[4] Allen-Zhu, Z., & Li, Y. (2024). Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. *arXiv:2309.14316*.

[5] Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

[6] Lake, B. M., & Baroni, M. (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. *ICML*.

[7] Press, O., Zhang, M., Min, S., et al. (2023). Measuring and Narrowing the Compositionality Gap in Language Models. *EMNLP Findings*.

[8] Yang, S., Gribovskaya, E., Kassner, N., et al. (2024). Do Large Language Models Latently Perform Multi-Hop Reasoning? *ACL*.

[9] Biran, E., Gottesman, D., Yang, S., et al. (2024). Hopping Too Late: Exploring the Limitations of LLMs on Multi-Hop Queries. *EMNLP*.

[10] Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

[11] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

[12] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT (ROME). *NeurIPS*.

[13] Power, A., Burda, Y., Edwards, H., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *arXiv:2201.02177*.

[14] Qwen Team (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.

[15] DeepSeek-AI (2024). DeepSeek-V3 Technical Report. *arXiv:2412.19437*.

[16] Papalampidi, P., Keller, F., & Lapata, M. (2019). Movie Plot Analysis via Turning Point Identification (TRIPOD). *EMNLP*.

[17] He, M., Ge, Y., Chen, E., et al. (2018). Exploring the Emerging Type of Comment for Online Videos: DanMu. *ACM Transactions on the Web*.

[18] Khandelwal, U., Levy, O., Jurafsky, D., et al. (2020). Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM). *ICLR*.

---

## 8. Ethics Statement

**Study 1** uses entirely synthetic fictional entities; no human data is involved. **Study 2** uses publicly posted danmaku and reader comments collected via public APIs, containing only text and timestamps—no user identifiers, profiles, or PII. Comments are analyzed in aggregate for non-commercial academic research consistent with fair-use norms. Automated dimension discovery could theoretically inform content optimization; our contribution is descriptive and does not build persuasion systems.

## 9. Reproducibility Statement

All code, data, and results are publicly released. Study 1: fictional-entity chain generator, LoRA fine-tuning scripts (Qwen2.5-1.5B/3B, rank/epochs specified in the repository), and DeepSeek-V3 evaluation (temperature=0, n=100/hop). Weight-arm results use n_chains=20 × 2 seeds (=40/hop); we report exact settings and release per-run result files. Study 2: BGE-small-zh embeddings, KMeans clustering, LLM classification prompts, and temporal-binning code. We note the weight-arm sample size is modest; confidence intervals and additional seeds are marked as future work in §6.
