"""Step B 真正版: 从ASR字幕内容预测结构位置
用弹幕关键词标的位置作为ground truth, 看DeepSeek从字幕内容能否判断结构点。
这才是"内容→结构"的真实验(之前是弹幕→结构, 现在是视频解说内容→结构)。
"""
import json, sys, os, random, time
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","..","memory","memory-engine","memory_engine"))
from deepseek_client import DeepSeekClient
from collections import defaultdict

KW_STRUCT=['伏笔','原来','前面','呼应','回收','二刷','细节','暗示','铺垫','果然','难怪','怪不得']

def main():
    # 加载弹幕(标ground truth) + ASR字幕(做input)
    base=os.path.join(os.path.dirname(__file__),'..')
    dm_data=json.load(open(os.path.join(base,'data','BV1LSoyYqEuU.json'),encoding='utf-8'))
    asr_data=json.load(open(os.path.join(base,'data','BV1LSoyYqEuU_asr.json'),encoding='utf-8'))
    bin_texts=asr_data['bin_texts']  # {bin_idx_str: text}
    dms=dm_data['danmaku']

    # 用弹幕关键词标结构位置
    bin_size=30
    bins_dm=defaultdict(list)
    for dm in dms: bins_dm[int(dm['t']//bin_size)].append(dm['text'])
    struct_ratio={}
    for b,txts in bins_dm.items():
        if len(txts)>=5:
            struct_ratio[b]=sum(1 for t in txts if any(k in t for k in KW_STRUCT))/len(txts)

    # 找结构高点 + 控制点(弹幕量匹配)
    ranked=sorted(struct_ratio.items(),key=lambda x:x[1],reverse=True)
    top_bins=[(b,r) for b,r in ranked[:10] if r>0.08 and str(b) in bin_texts]
    bot_bins=[(b,r) for b,r in ranked if r==0 and str(b) in bin_texts and len(bins_dm[b])>=5]

    # 配对(弹幕量±50%)
    random.seed(42)
    pairs=[]
    used=set()
    for tb,tr in top_bins:
        target=len(bins_dm[tb])
        for cb,cr in random.sample(bot_bins,min(len(bot_bins),30)):
            if cb in used: continue
            if abs(len(bins_dm[cb])-target)/max(target,1)<0.5:
                pairs.append((tb,cb)); used.add(cb); break
    print(f"配对数: {len(pairs)} (结构高点 vs 控制点)")
    print(f"输入: ASR字幕内容(视频解说文本, 非弹幕)\n")

    c=DeepSeekClient()
    correct=0; total=0
    for i,(tb,cb) in enumerate(pairs):
        s_text=bin_texts[str(tb)]
        c_text=bin_texts[str(cb)]
        # 随机AB顺序
        if random.random()>0.5:
            a,b_=s_text,c_text; answer="A"
        else:
            a,b_=c_text,s_text; answer="B"
        prompt=f"""以下是一段电影解说视频中两个不同时间段的解说内容(ASR转录):

A段内容:
{a[:300]}

B段内容:
{b_[:300]}

其中一段是在揭晓一个之前埋下的伏笔/反转/关键剧情(叙事结构节点),
另一段是普通的剧情描述。
请根据内容判断哪段更像是在揭晓/反转。只回答A或B:"""
        try:
            r=c.chat([{"role":"user","content":prompt}],temperature=0,max_tokens=5)
            pred=r['content'].strip().upper()[:1]
            hit=pred==answer
            correct+=int(hit); total+=1
            t_min=f"{tb*30//60}:{tb*30%60:02d}"
            c_min=f"{cb*30//60}:{cb*30%60:02d}"
            print(f"  [{i}] struct@{t_min} vs ctrl@{c_min} | 正确={answer} 预测={pred} {'✓' if hit else '✗'}")
        except Exception as e:
            print(f"  [{i}] err: {e}")
        time.sleep(0.3)

    acc=correct/total if total else 0
    print(f"\n{'='*60}")
    print(f"  Step B: 从ASR字幕内容预测结构位置 (DeepSeek)")
    print(f"  准确率: {correct}/{total} = {acc:.0%} (chance=50%)")
    print(f"\n  判定: ", end="")
    if acc>=0.75:
        print(f"✅ 内容信号存在! 字幕(视频解说)能区分结构节点 → 模型能从内容学叙事结构")
    elif acc>=0.6:
        print(f"⚠️ 弱信号 — 有一定区分但需更多数据确认")
    else:
        print(f"❌ 字幕内容无法区分 → 结构信号不在解说文本中(可能在视觉/音调)")
    out=os.path.join(base,'results','stepb_asr_classify.json')
    json.dump({"accuracy":acc,"correct":correct,"total":total},
              open(out,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(f"  保存 {out}")

if __name__=="__main__": main()
