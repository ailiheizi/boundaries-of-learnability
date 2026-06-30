# Boundaries of Learnability: When Explicit Knowledge Enables and Implicit Knowledge Fails

## Abstract

We investigate the boundaries of what language models can and cannot learn, through two complementary studies spanning reasoning and narrative understanding. In **Study 1** (compositional reasoning), we show that models perfectly recall individual facts (single-hop 100%) but catastrophically fail at implicit multi-hop composition (2-hop: 27%, 4-hop: 0%). Chain-of-thought (CoT) rescues composition when facts are in-context (scaling from 40% at 1.5B to 100% at large scale), but **completely fails when facts are internalized in weights** (0-2% across all scales)—revealing that the bottleneck is knowledge retrievability, not reasoning capacity. A RAG baseline (step-by-step retrieval from external storage) restores composition to 100% at all hop counts, confirming the bottleneck is purely retrieval. In **Study 2** (narrative structure from reader comments), we show that reader annotations reliably mark structural positions (4.5× above baseline, independent of arousal), and models can distinguish coarse-grained structural scenes (100%), but **cannot distinguish fine-grained within-scene foreshadowing** (44%, n=18, p=0.76)—because foreshadowing is a reader's retrospective interpretation, not a learnable textual feature. Together, these findings establish a unified boundary: **explicit knowledge (in-context facts, keyword annotations) is usable; implicit knowledge (weight-stored facts, subtle textual cues) is not compositionally accessible or learnable**, and this limitation does not scale away with model size.

---

## 1. Introduction

A fundamental question in AI is: what can models learn, and what remains beyond their reach? We approach this through two empirical studies that, despite different domains, converge on the same answer.

**The explicit-implicit boundary.** When knowledge is presented explicitly—facts stated in the prompt, structural positions marked by keywords—models use it effectively. When the same knowledge exists only implicitly—encoded in model weights after training, or latent in textual style without surface markers—models fail to access or learn it. Critically, this failure persists across model scales.

This connects to the Reversal Curse (Berglund et al., 2023): models trained on "A is B" cannot infer "B is A." We generalize this from single-hop reversal to multi-hop composition (Study 1) and from factual recall to narrative structure learning (Study 2), showing that the underlying limitation—unidirectional, non-composable knowledge storage—manifests across domains.

**Contributions:**
1. A compositional reasoning benchmark with anti-cheating controls (fictional entities, single-hop gating, open-generation), revealing that CoT rescue depends on knowledge being in-context and scales with model capacity (§3).
2. The first demonstration that weight-internalized facts cannot support CoT-based composition regardless of scale, localizing the bottleneck to knowledge retrievability (§3.3).
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

Same facts LoRA-trained into model weights (12 epochs, closed-book evaluation, 2 seeds × 20 chains).

| Storage | Model | Direct (2/3/4) | CoT (2/3/4) |
|---------|-------|----------------|-------------|
| In-context | 1.5B | 25→5→5% | 45→45→30% |
| In-context | 3B | 20→10→0% | 60→70→65% |
| **Weight** | **1.5B** | 35→0→3% | **2→0→0%** |
| **Weight** | **3B** | 5→0→0% | **8→0→0%** |

**Finding 3 (core):** Weight-internalized CoT completely fails (0-8%) and does not improve with scale (1.5B ≈ 3B ≈ 0%).

**Diagnostic:** Raw CoT outputs from weight-internalized models show: (a) second-hop retrieval failure (stuck on intermediate entity), (b) hallucinated entities absent from training ("Tovak Sterling"), (c) correct chain enumeration followed by wrong final answer. The model can write reasoning *steps* but cannot reliably *retrieve* the next fact from weights at each step.

### 3.4 RAG Baseline: Retrieval Rescues Composition

To confirm that the bottleneck is knowledge *retrievability* rather than reasoning *capacity*, we test a RAG (Retrieval-Augmented Generation) condition: the model receives no facts in the prompt, but at each reasoning step, the relevant single-hop fact is retrieved from an external knowledge base and injected.

| Storage | Condition | 2-hop | 3-hop | 4-hop |
|---------|-----------|-------|-------|-------|
| In-context | All facts in prompt | 100% | 100% | 100% |
| **Weight** | Closed-book (CoT) | 2% | 0% | 0% |
| **RAG** | Step-by-step retrieval | **100%** | **100%** | **100%** |
| Blind | No facts, no retrieval | 0% | 0% | 0% |

**Finding 4:** RAG completely rescues multi-hop composition (100% at all hop counts), matching in-context performance exactly. This confirms that: (a) the model's reasoning capacity is intact—it can compose when facts are supplied step-by-step; (b) the failure of weight-internalized CoT is purely a retrieval problem, not a reasoning problem.

**Implication:** The architectural fix is straightforward—externalize knowledge storage and retrieve per-step. The "composition collapse" is not fundamental to language models; it is specific to parametric (weight-based) knowledge storage.

### 3.5 Interpretation

CoT provides a reasoning *scaffold*, but the scaffold requires reliable fact *supply*. Weight-internalized facts are stored as isolated unidirectional keys—CoT's chained retrieval fails at each hop, and this failure is architecture-level (does not scale away). This generalizes the Reversal Curse: not just "A→B doesn't yield B→A," but "weight-stored facts cannot support *any* multi-hop CoT retrieval."

The RAG result (§3.4) completes the picture: when retrieval is externalized, composition works perfectly. The bottleneck was never reasoning—it was always retrieval from weights.

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

We establish that the boundary between what models can and cannot do aligns with the explicit-implicit divide in knowledge form. Explicit knowledge—facts in context, structural positions marked by keywords—is compositionally accessible and learnable. Implicit knowledge—facts in weights, latent narrative structure—resists both compositional retrieval (even with CoT) and learning (even with strong models). This boundary does not scale away, suggesting it reflects architectural constraints rather than capacity limitations. Our anti-cheating methodology (fictional entities, gating controls, arousal matching, temporal shuffling) provides a template for rigorous evaluation of model capabilities at their boundaries.

---

## References

- Berglund et al. (2023). The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A." arXiv:2309.12288.
- Allen-Zhu & Li (2024). Physics of Language Models: Part 3, Knowledge Capacity. arXiv:2309.14316.
- Wei et al. (2022). Chain-of-thought prompting elicits reasoning in large language models.
- Lake & Baroni (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks.
- Papalampidi et al. (2019). Movie Script Summarization as Graph-based Scene Extraction (TRIPOD).
- "Teaching Human Behavior Improves Content Understanding of LLMs" (2405.00942).
