"""评价驱动表征学习 (Evaluation-Driven Representation Learning) - Step 1
=========================================================================
从弹幕评价中自动发现语义维度（不预设类别）

方法:
1. 用 BGE embedding 对所有弹幕做向量化
2. 聚类 (HDBSCAN/KMeans) 自动发现评价维度
3. 用 LLM 对每个聚类命名（给出语义标签）
4. 验证: 这些聚类是否对应人类可理解的评价维度？

核心假说:
  弹幕文本的语义空间里自然存在评价维度的聚类结构，
  且这些维度能被自动发现而无需人工定义。

用法:
  python step1_discover_dimensions.py [--data ../data/narrative/BV1LSoyYqEuU.json]
"""
import os, sys, json, time, argparse
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))


def load_danmaku(path):
    """加载弹幕, 过滤无效内容"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    danmaku = data['danmaku']
    # 过滤: 太短(<3字), 纯符号, 太长(>50字)
    valid = []
    for d in danmaku:
        text = d['text'].strip()
        if len(text) < 3 or len(text) > 50:
            continue
        # 过滤纯符号/乱码
        chinese_ratio = sum(1 for c in text if '一' <= c <= '鿿') / max(len(text), 1)
        if chinese_ratio < 0.3 and not any(c.isalpha() for c in text):
            continue
        valid.append({'t': d['t'], 'text': text})
    return valid, data.get('title', ''), data.get('duration', 0)


def embed_texts(texts, batch_size=64):
    """用 BGE 对文本做 embedding (sentence-transformers)"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("需要安装: pip install sentence-transformers")
        sys.exit(1)

    print(f"  加载 BGE 模型...")
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    print(f"  编码 {len(texts)} 条弹幕...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings


def cluster_embeddings(embeddings, method='kmeans', n_clusters=10):
    """对 embedding 聚类"""
    from sklearn.cluster import KMeans, DBSCAN
    try:
        from sklearn.cluster import HDBSCAN as HDBSCANCluster
        has_hdbscan = True
    except ImportError:
        has_hdbscan = False

    if method == 'hdbscan' and has_hdbscan:
        print(f"  HDBSCAN 聚类...")
        clusterer = HDBSCANCluster(min_cluster_size=20, min_samples=5)
        labels = clusterer.fit_predict(embeddings)
    else:
        print(f"  KMeans 聚类 (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

    return labels


def analyze_clusters(texts, labels, timestamps, top_n=15):
    """分析每个聚类的内容"""
    clusters = {}
    for text, label, t in zip(texts, labels, timestamps):
        if label == -1:  # noise (HDBSCAN)
            continue
        if label not in clusters:
            clusters[label] = {'texts': [], 'timestamps': []}
        clusters[label]['texts'].append(text)
        clusters[label]['timestamps'].append(t)

    results = []
    for label in sorted(clusters.keys()):
        c = clusters[label]
        n = len(c['texts'])
        # 取代表性样本 (随机抽)
        import random
        random.seed(42)
        samples = random.sample(c['texts'], min(top_n, n))
        avg_t = np.mean(c['timestamps'])
        std_t = np.std(c['timestamps'])
        results.append({
            'cluster_id': int(label),
            'size': n,
            'samples': samples,
            'avg_timestamp': float(avg_t),
            'std_timestamp': float(std_t),
            'temporal_spread': 'concentrated' if std_t < 60 else 'dispersed',
        })

    return results


def name_clusters_with_llm(clusters, client=None):
    """用 LLM 对每个聚类自动命名"""
    if client is None:
        # 尝试导入 DeepSeek
        try:
            sys.path.insert(0, os.path.join(_here, '..', '..', '..', '..', 'memory', 'memory-engine', 'memory_engine'))
            from deepseek_client import DeepSeekClient
            client = DeepSeekClient()
        except:
            print("  [无 LLM] 跳过自动命名")
            return clusters

    print(f"  用 LLM 命名 {len(clusters)} 个聚类...")
    for c in clusters:
        samples_text = "\n".join(f"  - {s}" for s in c['samples'][:10])
        prompt = (f"以下是一组视频弹幕评论的聚类（{c['size']}条），请用2-4个字概括这个聚类的评价维度/主题：\n\n"
                  f"{samples_text}\n\n"
                  f"只输出维度名称（2-4个字），如：'剧情评价'、'视觉效果'、'节奏快慢'、'情感共鸣'等。不要解释。")
        try:
            r = client.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=20)
            c['dimension_name'] = r['content'].strip().strip("'\"。")
        except:
            c['dimension_name'] = f"cluster_{c['cluster_id']}"
        time.sleep(0.3)

    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_here, '..', '..', 'data', 'narrative', 'BV1LSoyYqEuU.json'))
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "hdbscan"])
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("  Step 1: 从弹幕中自动发现评价维度")
    print("=" * 60)

    # 1. 加载弹幕
    danmaku, title, duration = load_danmaku(args.data)
    print(f"\n  视频: {title}")
    print(f"  时长: {duration}s, 有效弹幕: {len(danmaku)}条")

    texts = [d['text'] for d in danmaku]
    timestamps = [d['t'] for d in danmaku]

    # 2. Embedding
    embeddings = embed_texts(texts)
    print(f"  Embedding shape: {embeddings.shape}")

    # 3. 聚类
    labels = cluster_embeddings(embeddings, method=args.method, n_clusters=args.n_clusters)
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  发现 {n_clusters_found} 个聚类")

    # 4. 分析
    clusters = analyze_clusters(texts, labels, timestamps)

    # 5. LLM 命名
    if not args.no_llm and os.environ.get("DEEPSEEK_API_KEY"):
        clusters = name_clusters_with_llm(clusters)

    # 6. 输出
    print(f"\n{'=' * 60}")
    print(f"  发现的评价维度:")
    print(f"{'=' * 60}")
    for c in sorted(clusters, key=lambda x: -x['size']):
        name = c.get('dimension_name', f"cluster_{c['cluster_id']}")
        spread = c['temporal_spread']
        print(f"\n  [{name}] (n={c['size']}, {spread})")
        for s in c['samples'][:5]:
            print(f"    - {s}")

    # 7. 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'narrative', 'eval_dimensions_discovered.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        "video": title,
        "total_danmaku": len(danmaku),
        "n_clusters": n_clusters_found,
        "method": args.method,
        "dimensions": clusters,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

    # 8. 初步验证: 维度是否有时间分布差异
    print(f"\n{'=' * 60}")
    print(f"  时间分布验证 (维度是否和视频位置相关)")
    print(f"{'=' * 60}")
    for c in sorted(clusters, key=lambda x: x['avg_timestamp']):
        name = c.get('dimension_name', f"cluster_{c['cluster_id']}")
        pct = c['avg_timestamp'] / max(duration, 1) * 100
        print(f"  {name:12s}: avg={c['avg_timestamp']:.0f}s ({pct:.0f}%), spread={c['std_timestamp']:.0f}s")


if __name__ == "__main__":
    main()
