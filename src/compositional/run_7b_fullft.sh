#!/bin/bash
# Boundaries Study 1 — 7B + Full Fine-Tuning experiment for Findings/main-conf push
# ============================================================================
# Reviewer demand: isolate STORAGE LOCATION as the only variable.
# The current weak point: LoRA 1.5B/3B vs in-context 671B confounds
#   scale × training-regime × storage. This script adds the critical arms:
#
#   1. Qwen2.5-7B, FULL fine-tuning (not LoRA), closed-book CoT
#      -> if still 0%, the dissociation survives at 7B + full-FT (STRONG)
#   2. Extended training (grokking regime: many epochs) at matched single-hop
#      -> tests whether more training unlocks composition
#   3. Matched single-hop recall gate: only test composition when single-hop ~100%
#
# Run on AutoDL (needs ~40GB+ VRAM for 7B full-FT with bf16 + grad checkpointing)
#
# Usage:
#   bash run_7b_fullft.sh
# ============================================================================

set -e
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Arm 1: Qwen2.5-3B FULL fine-tuning (baseline for full-FT) ==="
python idea4_weight.py \
    --model Qwen/Qwen2.5-3B \
    --full-ft \
    --n-chains 30 \
    --seeds 42,43,44 \
    --epochs 15 \
    --hops 2,3,4 \
    --out ../../results/compositional/weight_3b_fullft.json

echo ""
echo "=== Arm 2: Qwen2.5-7B FULL fine-tuning (THE critical arm) ==="
python idea4_weight.py \
    --model Qwen/Qwen2.5-7B \
    --full-ft \
    --n-chains 30 \
    --seeds 42,43,44 \
    --epochs 15 \
    --hops 2,3,4 \
    --out ../../results/compositional/weight_7b_fullft.json

echo ""
echo "=== Arm 3: Qwen2.5-7B extended training (grokking regime, 40 epochs) ==="
python idea4_weight.py \
    --model Qwen/Qwen2.5-7B \
    --full-ft \
    --n-chains 30 \
    --seeds 42,43,44 \
    --epochs 40 \
    --hops 2,3,4 \
    --out ../../results/compositional/weight_7b_grokking.json

echo ""
echo "=== Done! Results: ==="
echo "  weight_3b_fullft.json   (3B full-FT)"
echo "  weight_7b_fullft.json   (7B full-FT — key result)"
echo "  weight_7b_grokking.json (7B extended training)"
echo ""
echo "Interpretation:"
echo "  If 7B full-FT CoT still ~0% -> dissociation holds at scale (upgrade claim)"
echo "  If 7B full-FT CoT >> 0%     -> it was a budget/regime effect (revise claim)"
