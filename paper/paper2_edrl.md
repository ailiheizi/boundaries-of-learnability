# Evaluation-Driven Representation Discovery: When Audience Reactions Reveal Content Structure Without Labels

## Abstract

We show that natural audience evaluations (e.g., real-time video comments/danmaku) contain sufficient semantic structure to automatically discover meaningful content dimensions—without any human-defined labels or feature engineering. By clustering the semantic embeddings of evaluative comments and aligning them with temporal content, we demonstrate that: (1) 9 interpretable evaluation dimensions emerge automatically (plot, acting, visuals, music, pacing, etc.), stable across videos; (2) these dimensions temporally align with specific content segments (entropy 0.01–0.46 vs. 1.0 expected for random); (3) the temporal distribution pattern of dimensions serves as a content-type fingerprint (suspense dramas show dispersed plot discussion; film analyses show concentrated plot peaks). Our findings establish a new paradigm—**Evaluation-Driven Representation Learning**—where natural human evaluations, rather than engineered labels, drive the discovery of content structure.

---

## 1. Introduction

Understanding what makes content effective—what aspects audiences notice, appreciate, or criticize—is central to content analysis, recommendation systems, and creative AI. Traditional approaches require either:
- Expert-defined feature taxonomies (labor-intensive, domain-specific)
- Large-scale human annotation (expensive, non-scalable)
- Self-supervised objectives (powerful but not aligned with human perception)

We propose a different path: **using naturally-occurring audience evaluations as the signal source for representation discovery**. Our key insight is that audience reactions are not random noise—they contain structured semantic information about *which dimensions of content* triggered the response.

### Contribution

1. We demonstrate that semantic clustering of natural evaluations (video danmaku) automatically discovers 9 interpretable content dimensions, without any predefined taxonomy.
2. We show these dimensions temporally align with specific content segments, with concentration metrics (normalized entropy 0.01–0.55) far below random expectation.
3. We establish that dimension distributions differ systematically by content type, serving as content fingerprints.
4. We validate cross-video stability: core dimensions (plot, acting, visuals) recur across different videos and genres.
5. We propose the EDRL (Evaluation-Driven Representation Learning) paradigm as a general framework for leveraging natural evaluations.

### Relation to Prior Work

This work extends our earlier finding (Author et al., 2026) that keyword-based audience annotations mark narrative structure reliably (4.5× above baseline), while semantic content alone fails (1.05×). The present work resolves this apparent contradiction: the semantic signal *is* present, but requires **dimension-aware clustering** rather than direct density comparison to unlock.

---

## 2. Related Work

### 2.1 Audience Response Analysis
- Danmaku/live comment analysis (Chen et al., 2020; Wu et al., 2021)
- Sentiment analysis of viewer reactions (Zhou et al., 2022)
- *Gap: No work uses semantic clustering of reactions to discover content dimensions*

### 2.2 Video Content Understanding
- Video summarization via engagement signals (Song et al., 2023)
- Highlight detection from comments (Xu et al., 2021)
- *Gap: These use comments as scalar signals (density/sentiment), not as multi-dimensional semantic feedback*

### 2.3 Representation Learning from Human Feedback
- RLHF (Ouyang et al., 2022) — binary preference
- Reward modeling — learned scalar
- *Gap: RLHF uses curated labels; we use natural, multi-dimensional, unsolicited evaluations*

### 2.4 Unsupervised Dimension Discovery
- Topic modeling (LDA, BERTopic)
- Aspect-based sentiment analysis
- *Gap: These don't link discovered dimensions to temporal content features*

---

## 3. Method: Evaluation-Driven Representation Discovery

### 3.1 Data Collection
- Platform: Bilibili (Chinese video platform with time-synced danmaku)
- Videos: N videos spanning K genres (film analysis, suspense drama, entertainment, etc.)
- Raw data: timestamped comments (弹幕), 1000–5000 per video

### 3.2 Filtering Pipeline
1. **Rule-based noise removal**: deduplication, length filter, spam removal (removes ~40% noise)
2. **Evaluativeness classification**: LLM judges whether each comment evaluates content quality (vs. social interaction, memes, participation) → ~40% are evaluative in content-heavy videos

### 3.3 Dimension Discovery
1. Encode evaluative comments with BGE-small-zh (512-dim sentence embeddings)
2. Cluster via KMeans (k=6–10) or HDBSCAN
3. Automatic naming via LLM: each cluster → 2-4 character dimension label
4. **No predefined taxonomy** — dimensions emerge from data

### 3.4 Temporal Alignment
1. Bin timeline into 30-second segments
2. For each segment: compute dimension distribution
3. Metrics:
   - **Peak ratio**: max density / mean density (how concentrated is the dimension?)
   - **Normalized entropy**: H(temporal distribution) / log2(n_bins) (0=perfectly concentrated, 1=uniform)
   - **Dominant dimension per segment**: which dimension dominates at each timepoint

### 3.5 Cross-Video Validation
- Run pipeline on multiple videos of different genres
- Compare: which dimensions recur? How do distributions differ?

---

## 4. Results

### 4.1 Dimension Discovery (Study 1)

**Setting:** Film analysis video (惊天魔盗团, 41min, 4973 comments)

| Discovered Dimension | Count | Interpretation |
|---------------------|-------|----------------|
| Plot (剧情) | 44 (37%) | Story/logic evaluation |
| Overall (整体) | 20 (17%) | Holistic quality judgment |
| Details (细节) | 18 (15%) | Technical detail observation |
| Acting (演技) | 14 (12%) | Performance evaluation |
| Visuals (画面) | 9 (8%) | Visual quality |
| Emotional resonance (情感共鸣) | 5 (4%) | Emotional impact |
| Narrative technique (叙事手法) | 5 (4%) | Craft evaluation |
| Music (音乐) | 3 (2%) | Score/soundtrack |
| Pacing (节奏) | 2 (2%) | Rhythm evaluation |

**Finding 1:** 9 interpretable dimensions emerge without predefined taxonomy. These align with established film criticism categories, validating that audience evaluations implicitly encode professional analysis dimensions.

### 4.2 Temporal Alignment (Study 2)

| Dimension | Peak Time | Peak/Mean Ratio | Normalized Entropy | Concentrated? |
|-----------|-----------|-----------------|-------------------|---------------|
| Music | 4:00 | 79.2× | 0.01 | ★★★ |
| Acting | 37:00 | 36.8× | 0.24 | ★★ |
| Emotional resonance | 24:30 | 21.2× | 0.39 | ★★ |
| Plot | 7:00 | 16.6× | 0.46 | ★ |
| Details | 4:00 | 9.9× | 0.55 | ★ |
| Overall | 6:00 | 3.0× | 0.79 | (dispersed) |

**Finding 2:** Content-specific dimensions (music, acting) show extreme temporal concentration (entropy 0.01–0.24), while content-general dimensions (overall) are naturally dispersed (0.79). This confirms that dimension peaks correspond to specific content events.

**Finding 3 (built-in control):** "Overall" evaluation is the only dimension with high entropy (0.79), serving as an internal negative control—holistic judgments are position-independent, as expected.

### 4.3 Cross-Video Stability (Study 3)

**Setting:** 3 videos — film analysis, suspense drama, entertainment montage

| Metric | Film Analysis | Suspense Drama | Entertainment |
|--------|--------------|----------------|---------------|
| Evaluative rate | 40% | 44% | 10% |
| Dominant dimension | Plot (37%) | Plot (56%) | Visuals (30%) |
| Polarity | Balanced (+22%/-23%) | Negative-heavy (+16%/-57%) | Negative (50%) |
| Plot entropy | 0.46 (concentrated) | 0.90 (dispersed) | — |

**Finding 4:** Core dimensions (plot, acting, visuals, overall) recur across videos and genres.

**Finding 5 (genre fingerprint):** The same dimension ("plot") shows opposite temporal patterns by genre:
- Film analysis: concentrated (entropy 0.46) — plot discussion triggered by specific explanation segments
- Suspense drama: dispersed (entropy 0.90) — plot logic debated throughout

This demonstrates that **dimension distribution patterns serve as content-type fingerprints**, discoverable without any metadata.

### 4.4 Evaluative Rate as Content Signal

| Video Type | Evaluative Rate | Interpretation |
|------------|----------------|----------------|
| Content-heavy (film analysis, drama) | 40-44% | Rich evaluative signal |
| Entertainment-only (montage) | 10% | Minimal evaluative content |

**Finding 6:** Evaluative rate itself distinguishes content types — a zero-parameter content classifier.

---

## 5. Discussion

### 5.1 The EDRL Paradigm

We propose **Evaluation-Driven Representation Learning (EDRL)** as a general framework:

```
Traditional:  Human defines features → Model learns mapping
RLHF:        Human labels preferences → Model optimizes reward
EDRL:        Natural evaluations → Automatic dimension discovery → Content representation
```

Key differences from RLHF:
- **Source**: Natural (free, abundant) vs. curated (expensive)
- **Dimensionality**: Multi-dimensional semantic vs. scalar preference
- **Discovery**: Dimensions emerge vs. predefined
- **Interpretability**: Named dimensions vs. black-box reward

### 5.2 Why Clustering Succeeds Where Direct Comparison Failed

Our earlier work (Paper 1) found that semantic similarity between evaluative and non-evaluative comments was only 1.05× — seemingly no signal. The resolution: **the signal is in the clustering structure, not in pairwise similarity**. Different evaluation dimensions occupy different regions of embedding space, but within each region, evaluative and non-evaluative comments are intermixed. Clustering separates dimensions; temporal alignment reveals the content signal.

### 5.3 Limitations and Future Work

1. **Scale**: 3 videos is proof-of-concept; 50+ needed for robust claims
2. **Downstream tasks**: Need to show dimensions improve actual tasks (summarization, recommendation, highlight detection)
3. **Generalizability**: Currently Bilibili-specific; extend to YouTube comments, app reviews, product reviews
4. **Representation learning**: Currently descriptive; next step is using dimensions as supervision for training content encoders

### 5.4 Broader Applicability

The EDRL paradigm extends beyond video:
- **Food/recipes**: Taste reviews → discover flavor dimensions → optimize recipes
- **Products**: Customer reviews → discover quality dimensions → improve design  
- **Games**: Player feedback → discover experience dimensions → balance gameplay
- **Code**: Code review comments → discover quality dimensions → train better code models

Any domain where humans naturally produce evaluative text can use this framework.

---

## 6. Conclusion

We demonstrate that natural audience evaluations contain structured semantic information sufficient to automatically discover content dimensions, without labels or feature engineering. The discovered dimensions are interpretable, temporally aligned with content, stable across videos, and discriminative of content types. We propose Evaluation-Driven Representation Learning (EDRL) as a general paradigm for leveraging the vast, untapped resource of natural human evaluations.

---

## References

[To be filled]

---

## Appendix

### A. Polarity Analysis

Negative evaluations ("criticism") provide qualitatively different information from positive:
- Positive: "这段配乐绝了" → identifies music as a strength
- Negative: "节奏太拖了" → identifies pacing as a weakness
- Neutral: "这里是伏笔" → identifies narrative technique without judgment

All three contribute to dimension discovery but carry different signals for representation learning.

### B. Implementation Details

- Embedding model: BAAI/bge-small-zh-v1.5 (512-dim)
- Clustering: KMeans (k=6-10) with silhouette score selection
- LLM classification: DeepSeek-V3, temperature=0, batch=30
- LLM naming: DeepSeek-V3, temperature=0, 2-4 character labels
- Temporal binning: 30-second windows
