#!/bin/bash
# 扩 n=100 验证: weight-internalized CoT = 0%
# 在 AutoDL 上执行 (RTX 4080 SUPER 32GB)
# 预计耗时: 1.5B ~2h, 3B ~4h

set -e
cd "$(dirname "$0")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

echo "=== 1.5B n=100 (50 chains × 2 seeds) ==="
python idea4_weight.py \
    --model Qwen/Qwen2.5-1.5B \
    --n-chains 50 \
    --seeds 42,43 \
    --epochs 12 \
    --hops 2,3,4 \
    --out ../results/compositional/weight_1.5b_n100.json

echo "=== 3B n=100 (50 chains × 2 seeds) ==="
python idea4_weight.py \
    --model Qwen/Qwen2.5-3B \
    --n-chains 50 \
    --seeds 42,43 \
    --epochs 12 \
    --hops 2,3,4 \
    --out ../results/compositional/weight_3b_n100.json

echo "=== Done! ==="
echo "Results at:"
echo "  ../results/compositional/weight_1.5b_n100.json"
echo "  ../results/compositional/weight_3b_n100.json"
