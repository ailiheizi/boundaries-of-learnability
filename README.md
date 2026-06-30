# Boundaries of Learnability

[![DOI](https://zenodo.org/badge/1284425541.svg)](https://doi.org/10.5281/zenodo.21073254)

**When Explicit Knowledge Enables and Implicit Knowledge Fails**

Two empirical studies revealing that language models can use explicit knowledge (in-context facts, keyword annotations) but fail with implicit knowledge (weight-stored facts, subtle textual features) — and this boundary does not scale away.

## Key Findings

### Study 1: Compositional Reasoning Collapse
- Models recall facts perfectly (single-hop 100%) but fail multi-hop composition (2-hop→4-hop: 27%→0%)
- CoT rescues composition **only when facts are in-context** (scales 40%→65%→100% with model size)
- **Weight-internalized facts: CoT completely fails (0-2%) regardless of scale**
- **RAG baseline: step-by-step retrieval restores composition to 100% at all hop counts**
- Bottleneck is *knowledge retrievability*, not *reasoning capacity*

### Study 2: Narrative Structure from Reader Comments  
- Reader keywords ("foreshadowing"/"I see now") reliably mark structure (4.5× above arousal-matched controls)
- Coarse-grained (cross-scene): 100% distinguishable
- **Fine-grained (adjacent paragraphs): 44%, p=0.76 = chance**
- Foreshadowing is reader's retrospective interpretation, not a learnable textual feature

### Unified Conclusion
| | Explicit | Implicit |
|---|---|---|
| Reasoning | In-context CoT → 100% | Weight-stored → 0% |
| Reasoning | RAG (retrieval per step) → 100% | Closed-book → 0% |
| Narrative | Keyword annotations → 4.5× | Non-keyword content → 1.05× |

## Repository Structure

```
├── paper/paper.md                    # Full paper manuscript
├── src/
│   ├── compositional/               # Study 1 experiment code
│   │   ├── pilot_incontext.py       # In-context composition + CoT (data gen + eval)
│   │   ├── pilot_server.py          # Generalized server version (local/DeepSeek)
│   │   ├── idea4_weight.py          # Weight-internalization via LoRA
│   │   ├── local_client.py          # llama-server OpenAI-compatible client
│   │   ├── debug_cot.py             # CoT raw output diagnostic
│   │   └── pilot_smallmodel.py      # Local transformers version
│   └── narrative/                    # Study 2 experiment code
│       ├── fetch_danmaku.py          # Bilibili danmaku scraper
│       ├── pilot_a.py               # Keyword structure signal test
│       ├── pilot_a_semantic_v2.py    # BGE semantic backreference test
│       ├── stepb_classify.py         # DeepSeek content classification (danmaku)
│       └── stepb_asr_classify.py     # ASR subtitle classification
├── data/
│   └── narrative/                    # Bilibili danmaku + Qidian novel excerpts
├── results/
│   ├── compositional/               # All JSON results (in-context, weight, scaling)
│   └── narrative/                    # All JSON results (pilot A, semantic, fine-grained)
```

## Reproducing Results

### Study 1 (Compositional Reasoning)

```bash
# In-context (DeepSeek API)
export DEEPSEEK_API_KEY=your_key
cd src/compositional
python pilot_incontext.py

# Local small models (requires llama.cpp Vulkan + GGUF models)
# Start server: llama-server -m model.gguf -ngl 99 --port 8080
python pilot_server.py --backend local --tag qwen1.5b --n 20

# Weight internalization (requires GPU + peft)
python idea4_weight.py --model Qwen/Qwen2.5-1.5B --mode full
```

### Study 2 (Narrative Structure)

```bash
# Fetch Bilibili danmaku
cd src/narrative
python fetch_danmaku.py BV1LSoyYqEuU

# Pilot A: keyword structure signal
python pilot_a.py ../data/narrative/BV1LSoyYqEuU.json

# Fine-grained classification (requires DeepSeek API)
export DEEPSEEK_API_KEY=your_key
python stepb_classify.py
```

## Anti-Cheating Methodology

All experiments employ rigorous controls:
- **Fictional entities** (compositional): random names absent from pretraining
- **Single-hop gating**: verify fact recall before testing composition
- **Open-generation EM**: exact match, not multiple-choice (avoids recognition inflation)
- **Arousal matching** (narrative): compare structure-high vs. same-volume control points
- **Temporal shuffle**: randomize timestamps, signal must collapse
- **Multi-seed**: all key results replicated across seeds
- **Strong-model falsification**: DeepSeek used to verify small-model artifacts

## Environment

- Local GPU inference: AMD RX 5700 XT via llama.cpp Vulkan (205 tok/s on 1.5B-Q4)
- LoRA training: AutoDL RTX 4080 SUPER 32GB
- APIs: DeepSeek-chat, Bilibili danmaku XML, Qidian chapter review
- Dependencies: torch, transformers, peft, sentence-transformers (for BGE)

## Citation

If you use this work, please cite:

```
@misc{boundaries-of-learnability-2026,
  title={Boundaries of Learnability: When Explicit Knowledge Enables and Implicit Knowledge Fails},
  author={ailiheizi},
  year={2026},
  url={https://github.com/ailiheizi/boundaries-of-learnability}
}
```

## License

MIT
