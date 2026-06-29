"""Step B pilot: DeepSeek从去关键词弹幕内容判断结构位置
测: 去掉"伏笔/原来如此/难怪"这些标注词后, 剩余弹幕的讨论内容
    是否仍然能区分"叙事揭晓点" vs "普通时间点"?
方法: 给DeepSeek两组弹幕(A=结构高点去关键词, B=控制点), 盲猜哪个是揭晓点。
如果准确率>>50% → 内容信号存在(语义回指检测方法不行, 但信号在)
如果≈50% → 内容真的没有额外信号, 就是那些关键词
"""
import json, sys, os, random
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","..","memory","memory-engine","memory_engine"))
from deepseek_client import DeepSeekClient
import time

KW_STRUCT=['伏笔','原来','前面','呼应','回收','二刷','细节','暗示','铺垫','果然','难怪','怪不得']

def load_data(path):
    with open(path,encoding='utf-8') as f: return json.load(f)

def main():
    path = sys.argv[1] if len(sys.argv)>1 else os.path.join(
        os.path.dirname(__file__),'..','data','BV1LSoyYqEuU.json')
    d=load_data(path)
    dms=d['danmaku']; dur=d['duration']
    print(f"视频: {d['title']}")
    print(f"弹幕: {len(dms)}条")

    # 按30s bin分组, 去掉含关键词的弹幕
    bin_size=30
    from collections import defaultdict
    bins=defaultdict(list)
    for dm in dms:
        if not any(k in dm['text'] for k in KW_STRUCT):
            bins[int(dm['t']//bin_size)].append(dm['text'])

    # 标结构高点(用关键词版: 含关键词弹幕占比top)
    bins_all=defaultdict(list)
    for dm in dms:
        bins_all[int(dm['t']//bin_size)].append(dm['text'])
    struct_ratio={}
    for b,txts in bins_all.items():
        if len(txts)>=5:
            struct_ratio[b]=sum(1 for t in txts if any(k in t for k in KW_STRUCT))/len(txts)
    ranked=sorted(struct_ratio.items(),key=lambda x:x[1],reverse=True)
    top_bins=[b for b,r in ranked[:8] if r>0.08]       # 结构高点(回指比>8%)
    bot_bins=[b for b,r in ranked[len(ranked)//2:] if r==0 and len(bins[b])>=5]  # 控制点(零关键词)

    # 配对: 每对 = (结构高点去关键词弹幕, 控制点弹幕), 让DeepSeek盲猜
    random.seed(42)
    pairs=[]
    used=set()
    for tb in top_bins:
        if len(bins[tb])<3: continue
        # 找弹幕量±50%的控制点
        target=len(bins[tb])
        for cb in random.sample(bot_bins, min(len(bot_bins),20)):
            if cb in used: continue
            if abs(len(bins[cb])-target)/target < 0.5:
                pairs.append((tb, cb))
                used.add(cb); break
    print(f"配对数: {len(pairs)} (结构高点 vs 控制点, 去关键词后)")

    c=DeepSeekClient()
    correct=0; total=0
    for i,(tb,cb) in enumerate(pairs):
        struct_texts=bins[tb][:8]  # 取前8条(去关键词)
        ctrl_texts=bins[cb][:8]
        # 随机决定A/B顺序(防位置偏见)
        if random.random()>0.5:
            a_texts, b_texts = struct_texts, ctrl_texts
            answer="A"
        else:
            a_texts, b_texts = ctrl_texts, struct_texts
            answer="B"
        prompt = f"""以下是某部电影解说视频两个不同时间段的弹幕(已去掉明显标注词)。
其中一个时间段是叙事的"揭晓/反转"节点(观众刚看到关键剧情), 另一个是普通段落。
请根据弹幕讨论的内容语气判断哪个是揭晓/反转节点。

A组弹幕:
{chr(10).join('- '+t for t in a_texts)}

B组弹幕:
{chr(10).join('- '+t for t in b_texts)}

哪组更像是叙事揭晓/反转节点? 只回答A或B:"""
        try:
            r=c.chat([{"role":"user","content":prompt}],temperature=0,max_tokens=5)
            pred=r['content'].strip().upper()[:1]
            hit = pred==answer
            correct+=int(hit); total+=1
            if i<3:
                print(f"  [{i}] 正确={answer} 预测={pred} {'✓' if hit else '✗'}")
        except Exception as e:
            print(f"  [{i}] err: {e}")
        time.sleep(0.3)

    acc=correct/total if total else 0
    print(f"\n{'='*60}")
    print(f"  DeepSeek 从去关键词弹幕判断结构位置")
    print(f"  准确率: {correct}/{total} = {acc:.0%} (chance=50%)")
    print(f"\n  判定: ", end="")
    if acc >= 0.75:
        print(f"✅ 内容信号存在 — 去掉标注词后, 讨论内容仍可区分结构点")
    elif acc >= 0.6:
        print(f"⚠️ 弱信号 — 有一定区分但不强")
    else:
        print(f"❌ 无信号 — 去关键词后内容无法区分 → 信号确实只在那些关键词")

    out=os.path.join(os.path.dirname(__file__),'..','results','stepb_deepseek_classify.json')
    json.dump({"accuracy":acc,"correct":correct,"total":total,"n_pairs":len(pairs)},
              open(out,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(f"  保存 {out}")

if __name__=="__main__": main()
