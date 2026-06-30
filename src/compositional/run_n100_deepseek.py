"""扩大样本量验证: DeepSeek in-context CoT=100% + RAG=100% + blind=0%
n_per_hop=100 (之前 pilot=15, rag=15)
同时收集三个条件, 一次跑完。
预计: 300样本 × 3条件 × 0.3s = ~5分钟
"""
import os, sys, json, random, time
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None
# deepseek_client 在 research/memory/memory-engine/memory_engine/
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here,"..","..","..","..","memory","memory-engine","memory_engine"))
sys.path.insert(0, _here)
from deepseek_client import DeepSeekClient
from pilot_incontext import gen_dataset, normalize, build_prompt_cot, extract_cot_answer

N_PER_HOP = 100

def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    c = DeepSeekClient()
    dataset = gen_dataset(n_per_hop=N_PER_HOP, hops=(2,3,4), seed=2026)
    hops_list = sorted(set(d['hops'] for d in dataset))

    print("="*60)
    print(f"  扩大验证 n={N_PER_HOP}/hop: In-context CoT + RAG + Blind")
    print("="*60)

    results = {h: {'cot':0, 'rag':0, 'blind':0, 'n':0} for h in hops_list}

    for i, d in enumerate(dataset):
        h = d['hops']
        question = d['question']
        answer = d['answer']

        # --- 条件1: In-context CoT ---
        cot_prompt = build_prompt_cot(list(d['facts']), question)
        try:
            r_cot = c.chat([{"role":"user","content":cot_prompt}], temperature=0, max_tokens=600)
            cot_ans = extract_cot_answer(r_cot['content'])
            hit_cot = normalize(answer) in normalize(cot_ans)
        except Exception as e:
            print(f"  [{i}] cot err: {e}"); hit_cot = False

        # --- 条件2: RAG (逐步检索) ---
        kb = {}
        for s, r, o in d['facts']:
            kb[s] = o
        first_entity = d['target_facts'][0][0]
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
        try:
            r_rag = c.chat([{"role":"user","content":prompt_rag}], temperature=0, max_tokens=30)
            hit_rag = normalize(answer) in normalize(r_rag['content'])
        except Exception as e:
            print(f"  [{i}] rag err: {e}"); hit_rag = False

        # --- 条件3: Blind (闭卷) ---
        prompt_blind = (f"问题: {question}\n"
                        f"注意: 这些是虚构人物。请尝试回答。如不知道回答'不知道'。只回答全名。")
        try:
            r_blind = c.chat([{"role":"user","content":prompt_blind}], temperature=0, max_tokens=30)
            hit_blind = normalize(answer) in normalize(r_blind['content'])
        except Exception as e:
            print(f"  [{i}] blind err: {e}"); hit_blind = False

        results[h]['cot'] += int(hit_cot)
        results[h]['rag'] += int(hit_rag)
        results[h]['blind'] += int(hit_blind)
        results[h]['n'] += 1

        if i % 10 == 0:
            n_done = results[h]['n']
            print(f"  [{i}/{len(dataset)}] {h}-hop "
                  f"cot={results[h]['cot']}/{n_done} "
                  f"rag={results[h]['rag']}/{n_done} "
                  f"blind={results[h]['blind']}/{n_done}", flush=True)
        time.sleep(0.3)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  大样本验证 n={N_PER_HOP}/hop (DeepSeek, seed=2026)")
    print(f"{'='*60}")
    print(f"  {'hop':>4} {'CoT(in-ctx)':>12} {'RAG':>8} {'Blind':>8} {'n':>5}")
    for h in hops_list:
        r = results[h]; n = r['n']
        print(f"  {h:>4} {r['cot']/n:>11.0%} {r['rag']/n:>7.0%} {r['blind']/n:>7.0%} {n:>5}")

    # 保存
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'compositional', 'n100_deepseek.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_data = {
        "description": f"DeepSeek in-context CoT + RAG + Blind, n={N_PER_HOP}/hop, seed=2026",
        "n_per_hop": N_PER_HOP,
        "results": {}
    }
    for h in hops_list:
        r = results[h]; n = r['n']
        out_data["results"][str(h)] = {
            "cot_accuracy": r['cot']/n,
            "rag_accuracy": r['rag']/n,
            "blind_accuracy": r['blind']/n,
            "cot_correct": r['cot'],
            "rag_correct": r['rag'],
            "blind_correct": r['blind'],
            "total": n
        }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")

if __name__ == "__main__":
    main()
