"""评价驱动表征学习 - Step 1b: 弹幕过滤 + 重新聚类
================================================================
过滤噪音:
1. 去重复/刷屏 (同一文本出现>5次的)
2. 去纯互动 (太短<4字, 纯标点/符号)
3. 评价性判断: 用规则+LLM 筛选"对内容有评价"的弹幕
   评价性 = 对视频内容(画面/剧情/音乐/节奏/演技等)有正面或负面评判

保留的弹幕应该是:
  ✓ "这段配乐好燃" (正面评价-音乐)
  ✓ "节奏太拖了" (负面批评-节奏)
  ✓ "这个伏笔绝了" (正面评价-叙事)
  ✓ "演技尴尬" (负面批评-表演)
  ✓ "这里逻辑有bug" (批评-逻辑)
  ✗ "不愧是我" (纯社交)
  ✗ "哈哈哈哈" (纯情绪无评价)
  ✗ "方块7" (互动参与)
  ✗ "疯狂星期四" (无关梗)

用法:
  export DEEPSEEK_API_KEY=xxx
  python step1b_filter_and_recluster.py
"""
import os, sys, json, time, re
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))


def load_danmaku(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['danmaku'], data.get('title', ''), data.get('duration', 0)


def rule_filter(danmaku):
    """规则过滤: 去重复/刷屏/太短/纯符号"""
    # 统计频次
    from collections import Counter
    text_counts = Counter(d['text'].strip() for d in danmaku)

    filtered = []
    seen_texts = set()  # 同一文本只保留一次

    for d in danmaku:
        text = d['text'].strip()

        # 过滤条件
        if len(text) < 4:
            continue
        if len(text) > 80:
            continue
        # 去刷屏 (出现>3次的)
        if text_counts[text] > 3:
            continue
        # 去重复
        if text in seen_texts:
            continue
        # 纯符号/乱码
        chinese_count = sum(1 for c in text if '一' <= c <= '鿿')
        alpha_count = sum(1 for c in text if c.isalpha())
        if chinese_count + alpha_count < 3:
            continue
        # 去纯表情/颜文字
        if text.count('哈') > 3 or text.count('草') > 2:
            continue

        seen_texts.add(text)
        filtered.append({'t': d['t'], 'text': text})

    return filtered


def llm_classify_evaluative(texts, client, batch_size=30):
    """用 LLM 批量判断弹幕是否有评价性
    返回每条的分类: evaluative(有评价) / non-evaluative(无评价)
    同时提取评价维度和情感极性
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))

        prompt = f"""以下是一批视频弹幕。请判断每条是否对视频内容有评价性（对画面/剧情/音乐/节奏/演技/导演手法等有正面或负面评判）。

弹幕列表:
{numbered}

对每条输出JSON格式（一行一条）:
{{"id": 序号, "eval": true/false, "dim": "维度", "polarity": "+"/"-"/"0"}}

维度选项: 剧情/画面/音乐/节奏/演技/叙事手法/细节/整体/情感共鸣/其他
- eval=true: 有评价（对内容某方面有判断）
- eval=false: 无评价（纯互动/梗/无意义）
- polarity: "+"正面, "-"负面, "0"中性陈述

只输出JSON行，不要其他文字:"""

        try:
            r = client.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=1500)
            # 解析输出
            for line in r['content'].strip().split('\n'):
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    obj = json.loads(line)
                    results.append(obj)
                except:
                    continue
        except Exception as e:
            print(f"    LLM batch error: {e}")
            # fallback: 全标 unknown
            for j in range(len(batch)):
                results.append({"id": j+1, "eval": None, "dim": "unknown", "polarity": "0"})

        time.sleep(0.5)
        if i % 90 == 0:
            print(f"    [{i}/{len(texts)}] 已分类...", flush=True)

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_here, '..', '..', 'data', 'narrative', 'BV1LSoyYqEuU.json'))
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--max-classify", type=int, default=300, help="最多送LLM分类多少条(控制成本)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    sys.path.insert(0, os.path.join(_here, '..', '..', '..', '..', 'memory', 'memory-engine', 'memory_engine'))
    from deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    print("=" * 60)
    print("  Step 1b: 过滤噪音 + 评价性分类 + 重新聚类")
    print("=" * 60)

    # 1. 加载
    danmaku, title, duration = load_danmaku(args.data)
    print(f"\n  视频: {title}")
    print(f"  原始弹幕: {len(danmaku)}条")

    # 2. 规则过滤
    filtered = rule_filter(danmaku)
    print(f"  规则过滤后: {len(filtered)}条 (去掉 {len(danmaku)-len(filtered)} 条噪音)")

    # 3. LLM 评价性分类 (抽样)
    import random
    random.seed(42)
    sample_size = min(args.max_classify, len(filtered))
    sample_indices = sorted(random.sample(range(len(filtered)), sample_size))
    sample_texts = [filtered[i]['text'] for i in sample_indices]

    print(f"\n  LLM 分类 {sample_size} 条弹幕的评价性...")
    classifications = llm_classify_evaluative(sample_texts, client)

    # 统计
    evaluative = []
    non_evaluative = []
    for idx, cls in zip(sample_indices, classifications):
        if cls.get('eval', False):
            evaluative.append({
                **filtered[idx],
                'dimension': cls.get('dim', 'unknown'),
                'polarity': cls.get('polarity', '0'),
            })
        else:
            non_evaluative.append(filtered[idx])

    eval_rate = len(evaluative) / max(len(classifications), 1)
    print(f"\n  评价性弹幕: {len(evaluative)}/{len(classifications)} = {eval_rate:.0%}")
    print(f"  非评价性: {len(non_evaluative)}")

    # 4. 维度分布
    from collections import Counter
    dim_counts = Counter(e['dimension'] for e in evaluative)
    pol_counts = Counter(e['polarity'] for e in evaluative)

    print(f"\n  维度分布:")
    for dim, cnt in dim_counts.most_common():
        print(f"    {dim:10s}: {cnt} ({cnt/len(evaluative):.0%})")

    print(f"\n  极性分布:")
    for pol, cnt in pol_counts.most_common():
        label = {'+': '正面', '-': '负面', '0': '中性'}.get(pol, pol)
        print(f"    {label}: {cnt} ({cnt/len(evaluative):.0%})")

    # 5. 对评价性弹幕做 embedding + 聚类
    if len(evaluative) >= 20:
        print(f"\n  对 {len(evaluative)} 条评价性弹幕做 embedding + 聚类...")
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        eval_texts = [e['text'] for e in evaluative]
        embeddings = model.encode(eval_texts, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False)

        k = min(args.n_clusters, len(evaluative) // 5)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        print(f"\n{'=' * 60}")
        print(f"  评价性弹幕聚类 (k={k})")
        print(f"{'=' * 60}")

        for cluster_id in range(k):
            members = [(evaluative[i], i) for i in range(len(evaluative)) if labels[i] == cluster_id]
            if not members:
                continue
            # 用该聚类的主要维度命名
            dims = Counter(m[0]['dimension'] for m in members)
            pols = Counter(m[0]['polarity'] for m in members)
            main_dim = dims.most_common(1)[0][0]
            main_pol = pols.most_common(1)[0][0]
            pol_label = {'+': '👍', '-': '👎', '0': '→'}.get(main_pol, '?')

            print(f"\n  [{main_dim} {pol_label}] (n={len(members)})")
            for m, _ in members[:5]:
                print(f"    {m['polarity']} {m['text']}")

    # 6. 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'narrative', 'eval_filtered_dimensions.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        "video": title,
        "total_raw": len(danmaku),
        "after_rule_filter": len(filtered),
        "sample_classified": len(classifications),
        "evaluative_count": len(evaluative),
        "evaluative_rate": eval_rate,
        "dimension_distribution": dict(dim_counts),
        "polarity_distribution": dict(pol_counts),
        "evaluative_samples": evaluative[:50],  # 保存样本
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")


if __name__ == "__main__":
    main()
