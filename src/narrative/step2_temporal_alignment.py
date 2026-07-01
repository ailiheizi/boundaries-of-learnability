"""评价驱动表征学习 - Step 2: 评价维度 × 时间轴对齐
================================================================
目标: 验证评价维度是否和视频内容的时间位置有结构性关联

方法:
1. 对全部弹幕做 LLM 评价性分类 (扩展 step1b 的 300 样本到全量)
   - 但成本太高 → 替代方案: 用 step1b 训出的模式做规则+embedding 近邻分类
2. 把评价性弹幕按时间分 bin (30s/bin)
3. 统计每个 bin 的维度分布
4. 可视化: 时间轴上各维度的密度变化

验证:
  如果某些时间段集中触发"剧情"评价, 另一些触发"画面"评价 →
  说明评价维度和视频内容特征是对齐的 (不同内容触发不同维度的评价)

用法:
  export DEEPSEEK_API_KEY=xxx
  python step2_temporal_alignment.py
"""
import os, sys, json, time, re, argparse
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))


def load_danmaku(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['danmaku'], data.get('title', ''), data.get('duration', 0)


def rule_filter(danmaku):
    """复用 step1b 的规则过滤"""
    from collections import Counter
    text_counts = Counter(d['text'].strip() for d in danmaku)
    filtered = []
    seen_texts = set()
    for d in danmaku:
        text = d['text'].strip()
        if len(text) < 4 or len(text) > 80:
            continue
        if text_counts[text] > 3:
            continue
        if text in seen_texts:
            continue
        chinese_count = sum(1 for c in text if '一' <= c <= '鿿')
        alpha_count = sum(1 for c in text if c.isalpha())
        if chinese_count + alpha_count < 3:
            continue
        if text.count('哈') > 3 or text.count('草') > 2:
            continue
        seen_texts.add(text)
        filtered.append({'t': d['t'], 'text': text})
    return filtered


def classify_by_embedding_similarity(texts, reference_data):
    """用 embedding 近邻法批量分类 (避免对每条都调 LLM)
    reference_data: step1b 的已分类样本作为锚点
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import KNeighborsClassifier

    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    # 构建参考集
    ref_texts = [r['text'] for r in reference_data]
    ref_dims = [r['dimension'] for r in reference_data]
    ref_pols = [r['polarity'] for r in reference_data]

    print(f"  编码参考集 ({len(ref_texts)} 条)...")
    ref_embeddings = model.encode(ref_texts, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False)

    print(f"  编码目标文本 ({len(texts)} 条)...")
    target_embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=False)

    # KNN 分类
    print(f"  KNN 分类 (k=5)...")
    knn_dim = KNeighborsClassifier(n_neighbors=min(5, len(ref_texts)), metric='cosine')
    knn_dim.fit(ref_embeddings, ref_dims)
    pred_dims = knn_dim.predict(target_embeddings)

    knn_pol = KNeighborsClassifier(n_neighbors=min(5, len(ref_texts)), metric='cosine')
    knn_pol.fit(ref_embeddings, ref_pols)
    pred_pols = knn_pol.predict(target_embeddings)

    # 计算置信度 (距离)
    distances, _ = knn_dim.kneighbors(target_embeddings)
    confidences = 1 - distances.mean(axis=1)  # 越近越confident

    return pred_dims, pred_pols, confidences


def temporal_analysis(danmaku_classified, duration, bin_size=30):
    """按时间 bin 统计维度分布"""
    n_bins = int(np.ceil(duration / bin_size))
    bins = [{'dims': {}, 'pols': {}, 'count': 0, 'start': i*bin_size, 'end': (i+1)*bin_size}
            for i in range(n_bins)]

    for d in danmaku_classified:
        bin_idx = min(int(d['t'] / bin_size), n_bins - 1)
        bins[bin_idx]['count'] += 1
        dim = d['dimension']
        pol = d['polarity']
        bins[bin_idx]['dims'][dim] = bins[bin_idx]['dims'].get(dim, 0) + 1
        bins[bin_idx]['pols'][pol] = bins[bin_idx]['pols'].get(pol, 0) + 1

    return bins


def find_dimension_peaks(bins, min_count=3):
    """找出每个维度的峰值时间段"""
    from collections import defaultdict
    dim_timeseries = defaultdict(list)

    for b in bins:
        total = max(b['count'], 1)
        for dim, count in b['dims'].items():
            dim_timeseries[dim].append(count / total)
        # 补零
        for dim in dim_timeseries:
            if dim not in b['dims']:
                dim_timeseries[dim].append(0)

    peaks = {}
    for dim, series in dim_timeseries.items():
        if len(series) == 0:
            continue
        arr = np.array(series)
        if arr.max() > 0:
            peak_idx = int(np.argmax(arr))
            peaks[dim] = {
                'peak_bin': peak_idx,
                'peak_time': peak_idx * 30,
                'peak_density': float(arr.max()),
                'mean_density': float(arr.mean()),
                'peak_ratio': float(arr.max() / max(arr.mean(), 0.01)),
            }

    return peaks, dim_timeseries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_here, '..', '..', 'data', 'narrative', 'BV1LSoyYqEuU.json'))
    ap.add_argument("--reference", default=os.path.join(_here, '..', '..', 'results', 'narrative', 'eval_filtered_dimensions.json'))
    ap.add_argument("--bin-size", type=int, default=30)
    ap.add_argument("--confidence-threshold", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("  Step 2: 评价维度 × 时间轴对齐")
    print("=" * 60)

    # 1. 加载弹幕
    danmaku, title, duration = load_danmaku(args.data)
    print(f"\n  视频: {title} ({duration}s)")

    # 2. 规则过滤
    filtered = rule_filter(danmaku)
    print(f"  过滤后: {len(filtered)} 条")

    # 3. 加载参考数据 (step1b 的结果)
    with open(args.reference, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    reference = ref_data['evaluative_samples']
    print(f"  参考集: {len(reference)} 条已分类评价弹幕")

    if len(reference) < 10:
        print("  ERROR: 参考集太小，需要先跑 step1b")
        sys.exit(1)

    # 4. 用 embedding 近邻法分类全部弹幕
    texts = [d['text'] for d in filtered]
    pred_dims, pred_pols, confidences = classify_by_embedding_similarity(texts, reference)

    # 5. 过滤低置信度
    classified = []
    for i, (d, dim, pol, conf) in enumerate(zip(filtered, pred_dims, pred_pols, confidences)):
        if conf >= args.confidence_threshold:
            classified.append({
                't': d['t'],
                'text': d['text'],
                'dimension': dim,
                'polarity': pol,
                'confidence': float(conf),
            })

    print(f"  高置信度分类: {len(classified)}/{len(filtered)} = {len(classified)/len(filtered):.0%}")

    # 6. 时间轴分析
    bins = temporal_analysis(classified, duration, args.bin_size)

    # 7. 找维度峰值
    peaks, dim_timeseries = find_dimension_peaks(bins)

    print(f"\n{'=' * 60}")
    print(f"  维度峰值分析 (bin={args.bin_size}s)")
    print(f"{'=' * 60}")
    print(f"  {'维度':8s} {'峰值时间':>10s} {'峰值密度':>10s} {'均值':>8s} {'峰/均比':>8s}")
    for dim, p in sorted(peaks.items(), key=lambda x: -x[1]['peak_ratio']):
        t_min = p['peak_time'] // 60
        t_sec = p['peak_time'] % 60
        print(f"  {dim:8s} {t_min:>3d}:{t_sec:02d} {p['peak_density']:>9.2f} "
              f"{p['mean_density']:>7.2f} {p['peak_ratio']:>7.1f}x")

    # 8. 时间轴概览 (每5分钟的主导维度)
    print(f"\n  时间轴概览 (主导维度 per 5min):")
    chunk_size = 300  # 5分钟
    for start in range(0, duration, chunk_size):
        end = min(start + chunk_size, duration)
        chunk_bins = [b for b in bins if b['start'] >= start and b['start'] < end]
        if not chunk_bins:
            continue
        # 合并该段的维度统计
        from collections import Counter
        chunk_dims = Counter()
        chunk_count = 0
        for b in chunk_bins:
            for dim, cnt in b['dims'].items():
                chunk_dims[dim] += cnt
            chunk_count += b['count']
        if chunk_count == 0:
            continue
        top_dims = chunk_dims.most_common(3)
        top_str = ", ".join(f"{d}({c})" for d, c in top_dims)
        t_min = start // 60
        print(f"    {t_min:>2d}:00-{t_min+5:>2d}:00 (n={chunk_count:>3d}): {top_str}")

    # 9. 验证: 维度时间分布的熵 (低熵=集中=和内容对齐)
    print(f"\n  维度集中度验证 (熵越低=时间分布越集中=和特定内容对齐):")
    for dim, series in sorted(dim_timeseries.items()):
        arr = np.array(series)
        arr = arr / max(arr.sum(), 1e-10)  # normalize
        entropy = -np.sum(arr[arr > 0] * np.log2(arr[arr > 0]))
        max_entropy = np.log2(len(arr))
        norm_entropy = entropy / max(max_entropy, 1)
        concentrated = "★集中" if norm_entropy < 0.7 else "分散"
        print(f"    {dim:8s}: H={entropy:.2f}/{max_entropy:.2f} = {norm_entropy:.2f} {concentrated}")

    # 10. 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'narrative', 'temporal_alignment.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        "video": title,
        "duration": duration,
        "bin_size": args.bin_size,
        "total_classified": len(classified),
        "confidence_threshold": args.confidence_threshold,
        "peaks": peaks,
        "timeline_summary": [
            {"start": b['start'], "end": b['end'], "count": b['count'],
             "top_dim": max(b['dims'], key=b['dims'].get) if b['dims'] else None}
            for b in bins if b['count'] > 0
        ],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")


if __name__ == "__main__":
    main()
