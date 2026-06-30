"""RAG基线: 模拟检索增强的组合推理
三条件对比(同一批虚构事实):
1. in-context: 所有事实在prompt (已有=100%)
2. weight: 事实训进权重, 闭卷 (已有=0%)
3. RAG: 事实不在prompt, 但每步CoT时"检索"到相关事实注入

RAG模拟方式: 不是真的向量检索(DeepSeek API没法接外部DB),
而是给模型一个"检索工具"——每步推理时, 把当前实体的单跳事实注入。
这模拟了"每步查一次数据库拿到下一跳"的RAG行为。

如果RAG条件下组合成功 → "组合推理需要显式检索, 不需要全量in-context"
= 正面结论, 对memory系统设计有直接指导
"""
import os, sys, json, random, time
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","..","memory","memory-engine","memory_engine"))
from deepseek_client import DeepSeekClient

# 复用 pilot_incontext 的数据生成
sys.path.insert(0, os.path.dirname(__file__))
from pilot_incontext import gen_dataset, normalize

def main():
    c = DeepSeekClient()
    dataset = gen_dataset(n_per_hop=15)
    hops_list = sorted(set(d['hops'] for d in dataset))

    # 构建"知识库": 每条事实 (subject, relation, object) 可按 subject 检索
    # RAG 模式: prompt不给全部事实, 而是给模型一个"逐步检索"的机制
    # 实现: 每个问题附带"你可以查询任何人的搭档"的指令 + 把chain展开为多轮

    print("="*60)
    print("  RAG Baseline: 逐步检索 vs In-context vs 闭卷")
    print("="*60)

    results = {h: {'incontext':0,'rag':0,'blind':0,'n':0} for h in hops_list}

    for i, d in enumerate(dataset):
        h = d['hops']
        facts = d['target_facts'] + [f for f in d['facts'] if f not in d['target_facts']]
        question = d['question']
        answer = d['answer']

        # 构建知识库(dict: subject -> object)
        kb = {}
        for s, r, o in d['facts']:
            kb[s] = o

        # --- 条件1: In-context (已有, 快速确认) ---
        random.shuffle(facts)
        fact_lines = "\n".join(f"- {s}{r}{o}。" for s,r,o in facts)
        prompt_ic = (f"以下是工作搭档关系:\n{fact_lines}\n\n"
                     f"问题: {question}\n请逐步推理,最后用'最终答案: 全名'回答。")
        r_ic = c.chat([{"role":"user","content":prompt_ic}], temperature=0, max_tokens=300)
        hit_ic = normalize(answer) in normalize(r_ic['content'])

        # --- 条件2: 闭卷(无任何事实) ---
        prompt_blind = (f"问题: {question}\n"
                        f"注意: 这些是虚构人物。请尝试回答。如不知道回答'不知道'。只回答全名。")
        r_blind = c.chat([{"role":"user","content":prompt_blind}], temperature=0, max_tokens=30)
        hit_blind = normalize(answer) in normalize(r_blind['content'])

        # --- 条件3: RAG (逐步检索) ---
        # 给模型一个"查询接口描述" + 每步实际提供检索结果
        # 模拟: 告诉模型"你可以查询, 我会给你结果", 然后多轮对话逐步给
        first_entity = d['target_facts'][0][0]  # 链起点

        # 构建逐步检索的prompt: 给模型起点 + 告诉它每步查到了什么
        rag_steps = []
        current = first_entity
        for hop in range(h):
            if current in kb:
                rag_steps.append(f"查询'{current}的工作搭档': 结果是{kb[current]}。")
                current = kb[current]
            else:
                rag_steps.append(f"查询'{current}的工作搭档': 未找到。")
                break

        prompt_rag = (f"你需要回答: {question}\n\n"
                      f"你没有这些人的背景知识,但可以逐步查询数据库。以下是查询结果:\n"
                      + "\n".join(rag_steps) + "\n\n"
                      f"根据以上检索结果,最终答案是谁?只回答全名:")
        r_rag = c.chat([{"role":"user","content":prompt_rag}], temperature=0, max_tokens=30)
        hit_rag = normalize(answer) in normalize(r_rag['content'])

        results[h]['incontext'] += int(hit_ic)
        results[h]['rag'] += int(hit_rag)
        results[h]['blind'] += int(hit_blind)
        results[h]['n'] += 1

        if i % 5 == 0:
            print(f"  [{i}/{len(dataset)}] {h}-hop ic={hit_ic} rag={hit_rag} blind={hit_blind}", flush=True)
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"  RAG vs In-context vs Blind (DeepSeek)")
    print(f"{'='*60}")
    print(f"  {'hop':>4} {'In-context':>12} {'RAG(逐步检索)':>15} {'Blind(闭卷)':>13}")
    for h in hops_list:
        r = results[h]; n = r['n']
        print(f"  {h:>4} {r['incontext']/n:>11.0%} {r['rag']/n:>14.0%} {r['blind']/n:>12.0%}")

    print(f"\n  关键对比: RAG vs Weight-internalized(之前=0%)")
    print(f"  如果RAG高 → '组合推理需要显式检索即可,不需要全在context'")

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'rag_baseline.json')
    json.dump({"results": {str(h): {k:v/results[h]['n'] for k,v in results[h].items() if k!='n'}
               for h in hops_list}},
              open(out,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"  保存 {out}")

if __name__ == "__main__": main()
