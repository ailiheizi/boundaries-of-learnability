"""Pilot A-语义版v2: 单条弹幕级别的语义回指检测
解决v1的问题: bin平均嵌入被无关弹幕淹没。
改用: 每条弹幕 → 和过去区间每条弹幕的 max cosine sim

方法:
1. 所有弹幕做BGE嵌入
2. 对每个bin的每条弹幕, 计算它与"过去"(2-15分钟前)所有弹幕的max语义相似度
3. 取bin内所有弹幕的max-sim平均值 = 该bin的"语义回指强度"
4. 对比: 结构高点(关键词标的) vs 唤醒匹配控制点 的语义回指强度
5. 另外: 单独看含结构关键词的弹幕 vs 不含的, 它们的回指sim有没有差

如果结构高点的(去掉含关键词弹幕后的)其他弹幕也高回指 → 深层语义回指真实存在
"""
import json, sys, os, random
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

KW_STRUCT=['伏笔','原来','前面','呼应','回收','二刷','细节','暗示','铺垫','果然','难怪','怪不得']

def load_data(path):
    with open(path,encoding='utf-8') as f: return json.load(f)

def main():
    path = sys.argv[1] if len(sys.argv)>1 else os.path.join(
        os.path.dirname(__file__),'..','data','BV1LSoyYqEuU.json')
    d=load_data(path)
    dms=d['danmaku']; dur=d['duration']
    print(f"视频: {d['title']}")
    print(f"弹幕: {len(dms)}条, {dur//60}分")

    from sentence_transformers import SentenceTransformer
    print("加载BGE模型...")
    model=SentenceTransformer('BAAI/bge-small-zh-v1.5')

    # 编码所有弹幕
    texts=[dm['text'] for dm in dms]
    times=np.array([dm['t'] for dm in dms])
    print(f"编码 {len(texts)} 条弹幕...")
    embs=model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    embs=np.array(embs)  # (N, dim)
    print("  OK")

    # 按30s bin分组
    bin_size=30
    bin_ids=(times//bin_size).astype(int)

    # 对每条弹幕计算: 它与"过去"(60s-600s前)弹幕的max sim
    lookback_min, lookback_max = 60, 600  # 回看1-10分钟
    print(f"计算每条弹幕的语义回指强度 (lookback {lookback_min}-{lookback_max}s)...")
    max_sims = np.zeros(len(dms))
    for i in range(len(dms)):
        t_i = times[i]
        # 找过去的弹幕索引
        mask = (times < t_i - lookback_min) & (times > t_i - lookback_max)
        if mask.sum() == 0:
            max_sims[i] = 0
            continue
        past_embs = embs[mask]
        sims = past_embs @ embs[i]  # (n_past,) cosine sims
        max_sims[i] = sims.max()
    print("  OK")

    # 按bin聚合
    from collections import defaultdict
    bin_scores = defaultdict(list)
    bin_scores_nokw = defaultdict(list)  # 去掉含关键词的弹幕后
    for i, dm in enumerate(dms):
        b = bin_ids[i]
        bin_scores[b].append(max_sims[i])
        if not any(k in dm['text'] for k in KW_STRUCT):
            bin_scores_nokw[b].append(max_sims[i])

    # 关键词标的结构高点 vs 低点
    struct_ratios = {}
    for b in set(bin_ids):
        b_texts = [dms[i]['text'] for i in range(len(dms)) if bin_ids[i]==b]
        if len(b_texts) >= 5:
            sc = sum(1 for t in b_texts if any(k in t for k in KW_STRUCT))
            struct_ratios[b] = sc / len(b_texts)

    valid = [(b, struct_ratios[b]) for b in struct_ratios if b in bin_scores and len(bin_scores[b])>=5]
    valid.sort(key=lambda x: x[1], reverse=True)
    n = len(valid)
    top_bins = [b for b,_ in valid[:n//4]]
    bot_bins = [b for b,_ in valid[n//2:]]

    import statistics
    # 全弹幕版
    top_all = [np.mean(bin_scores[b]) for b in top_bins if bin_scores[b]]
    bot_all = [np.mean(bin_scores[b]) for b in bot_bins if bin_scores[b]]
    # 去关键词版 (真正的"词汇盲"测试)
    top_nokw = [np.mean(bin_scores_nokw[b]) for b in top_bins if bin_scores_nokw.get(b)]
    bot_nokw = [np.mean(bin_scores_nokw[b]) for b in bot_bins if bin_scores_nokw.get(b)]

    print(f"\n{'='*60}")
    print(f"  语义回指 (单条弹幕 max-sim to past)")
    print(f"{'='*60}")
    if top_all and bot_all:
        ta, ba = statistics.mean(top_all), statistics.mean(bot_all)
        print(f"  [全弹幕] 结构高点: {ta:.4f}  控制点: {ba:.4f}  倍数: {ta/ba:.2f}x")
    if top_nokw and bot_nokw:
        tn, bn = statistics.mean(top_nokw), statistics.mean(bot_nokw)
        print(f"  [去关键词] 结构高点: {tn:.4f}  控制点: {bn:.4f}  倍数: {tn/bn:.2f}x")
        print(f"\n  判定: ", end="")
        if tn > bn * 1.1:
            print(f"✅ 去掉关键词弹幕后, 结构高点其他弹幕仍有更高语义回指 → 深层信号")
        else:
            print(f"⚠️ 去关键词后无差异 → 信号就是那些关键词本身, 非深层语义")
    else:
        print("  数据不足")

if __name__=="__main__": main()
