"""想法4 Pilot: 组合推理崩溃边界测试 (compositional reasoning boundary)

核心问题: 模型能否把分散给出的事实链 A->B, B->C 组合出 A->C?
(这个组合从不在prompt里显式出现)

反作弊设计:
1. 虚构实体 (随机名+随机属性) — 预训练里不存在, 杜绝靠世界知识
2. 单跳对照 — 先确认模型"记住"了每个中间事实, 否则组合失败不算
3. 组合从不显式出现 — prompt只给分散事实
4. 开放生成 (要求精确答出实体名), 非选择题 — 避免recognition虚高
5. 事实顺序打乱 — 防止靠"相邻"作弊

用法:
  export DEEPSEEK_API_KEY=xxx   # 或 set DEEPSEEK_API_KEY=xxx (Windows)
  python idea4_pilot.py
没有key时会先跑 --dry 验证数据生成逻辑。
"""
from __future__ import annotations
import os, sys, json, random, time

# ---- 复用 memory-engine 的 stdlib DeepSeek 客户端 ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "memory", "memory-engine", "memory_engine"))

# 虚构词库 (随机组合, 确保预训练里不存在这些具体绑定)
FIRST = ["Zelph","Quorin","Vandel","Mírabel","Tovak","Yelena","Karsh","Brünn","Oslo","Veyra",
         "Drennan","Sythe","Marlow","Pendric","Casimir","Thaddeus","Wren","Galen","Ophira","Lucan"]
SURNAME = ["Voss","Karr","Thorne","Belmont","Ashby","Cregan","Dunmore","Fenwick","Halloran","Ives",
           "Quill","Ravenscroft","Sterling","Underwood","Vance","Whitlock","Yarrow","Zane","Brackett","Coyle"]

def rand_entity(used):
    while True:
        name = f"{random.choice(FIRST)} {random.choice(SURNAME)}"
        if name not in used:
            used.add(name); return name

def make_chain(n_hops, used):
    """生成一条 n_hops 长的事实链: E0 --rel--> E1 --rel--> ... --> En
    返回 (facts[list of (subj, rel, obj)], entities[list], chain_question, answer)
    关系用'搭档'(partner)构成链: E0的搭档是E1, E1的搭档是E2...
    n_hops跳问题: E0的搭档的搭档...(n次)是谁? -> En
    """
    ents = [rand_entity(used) for _ in range(n_hops+1)]
    facts = []
    for i in range(n_hops):
        facts.append((ents[i], "的工作搭档是", ents[i+1]))
    # 组合问题
    rel_chain = "的工作搭档" * n_hops
    question = f"{ents[0]}{rel_chain}是谁?"
    answer = ents[n_hops]
    return facts, ents, question, answer

def build_prompt(all_facts, question):
    random.shuffle(all_facts)  # 打乱事实顺序 (反作弊)
    fact_lines = "\n".join(f"- {s}{r}{o}。" for s,r,o in all_facts)
    return (
        "以下是一些关于虚构人物工作关系的事实:\n\n"
        f"{fact_lines}\n\n"
        f"问题: {question}\n"
        "只回答人物的全名(名 + 姓), 不要解释。如果无法确定, 回答'不知道'。"
    )

def build_prompt_cot(all_facts, question):
    """CoT版: 允许逐步推理, 最后给出答案"""
    random.shuffle(all_facts)
    fact_lines = "\n".join(f"- {s}{r}{o}。" for s,r,o in all_facts)
    return (
        "以下是一些关于虚构人物工作关系的事实:\n\n"
        f"{fact_lines}\n\n"
        f"问题: {question}\n"
        "请一步一步追踪这条工作搭档链, 逐步推理每一跳。\n"
        "完成推理后, 必须在最后另起一行, 严格用这个格式输出: '最终答案: <名> <姓>'\n"
        "注意: 最终答案必须是链条终点那个人, 不要输出中间步骤的人名。"
    )

def extract_cot_answer(text):
    """从CoT输出里抽取最终答案。
    1) 优先找显式标记: 最终答案/答案/所以/因此 后面的内容
    2) fallback: 取最后一个'X的工作搭档是Y'里的Y(推理链终点)
    3) 再fallback: 最后一行
    """
    import re
    # 1) 显式答案标记 (取最后一个匹配, 因为可能多次出现)
    ms = list(re.finditer(r"(?:最终答案|答案|答)[:：]?\s*(?:全名[:：]?)?\s*([A-Za-zÀ-ÿ]+\s+[A-Za-zÀ-ÿ]+)", text))
    if ms:
        return ms[-1].group(1)
    # 2) 结论标记 所以/因此 后的全名
    ms2 = list(re.finditer(r"(?:所以|因此|故)[，,]?.*?是\s*([A-Za-zÀ-ÿ]+\s+[A-Za-zÀ-ÿ]+)", text))
    if ms2:
        return ms2[-1].group(1)
    # 3) 最后一个 'X的工作搭档是Y' 的 Y
    ms3 = list(re.finditer(r"是\s*([A-Za-zÀ-ÿ]+\s+[A-Za-zÀ-ÿ]+)", text))
    if ms3:
        return ms3[-1].group(1)
    lines = [l for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text

def build_singlehop_check(fact):
    """单跳对照: 直接问一个给定事实, 确认模型读进去了"""
    s,r,o = fact
    return f"已知事实: {s}{r}{o}。\n问题: {s}{r}是谁? 只回答全名。"

def gen_dataset(n_per_hop=15, hops=(2,3,4), n_distractor_chains=3, seed=42):
    """每个hop数生成 n_per_hop 个测试样本。
    每个样本: 1条目标链 + 若干干扰链 (增加难度+防止只有一条链可选)"""
    random.seed(seed)
    dataset = []
    for h in hops:
        for _ in range(n_per_hop):
            used = set()
            tgt_facts, tgt_ents, q, ans = make_chain(h, used)
            # 干扰链 (同样长度, 无关实体)
            distractor_facts = []
            for _ in range(n_distractor_chains):
                df, _, _, _ = make_chain(h, used)
                distractor_facts += df
            all_facts = tgt_facts + distractor_facts
            dataset.append({
                "hops": h,
                "facts": all_facts,
                "target_facts": tgt_facts,
                "question": q,
                "answer": ans,
                "first_entity": tgt_ents[0],
            })
    return dataset

def normalize(s):
    return s.strip().strip("。.,，\"'").lower()

def run(dry=False):
    dataset = gen_dataset()
    print(f"生成 {len(dataset)} 个样本 (hops: {sorted(set(d['hops'] for d in dataset))})")

    if dry:
        # 只展示样例, 验证数据逻辑
        for h in (2,3,4):
            ex = next(d for d in dataset if d['hops']==h)
            print(f"\n{'='*50}\n[{h}-hop 样例]")
            print(build_prompt(list(ex['facts']), ex['question']))
            print(f"  >>> 正确答案: {ex['answer']}")
        return

    from deepseek_client import DeepSeekClient
    client = DeepSeekClient()

    results = {h: {"comp_correct":0, "comp_total":0, "single_correct":0, "single_total":0,
                   "cot_correct":0, "cot_total":0}
               for h in sorted(set(d['hops'] for d in dataset))}

    for i, d in enumerate(dataset):
        h = d['hops']
        # 1) 单跳对照: 抽查目标链的第一跳, 确认模型能读
        sf = d['target_facts'][0]
        single_prompt = build_singlehop_check(sf)
        try:
            sr = client.chat([{"role":"user","content":single_prompt}], temperature=0, max_tokens=30)
            single_ok = normalize(sf[2]) in normalize(sr['content'])
        except Exception as e:
            print(f"  [{i}] single-hop API err: {e}"); single_ok=False
        results[h]["single_total"] += 1
        results[h]["single_correct"] += int(single_ok)

        # 2) 组合推理 (直接回答)
        comp_prompt = build_prompt(list(d['facts']), d['question'])
        try:
            cr = client.chat([{"role":"user","content":comp_prompt}], temperature=0, max_tokens=30)
            comp_ok = normalize(d['answer']) in normalize(cr['content'])
        except Exception as e:
            print(f"  [{i}] comp API err: {e}"); comp_ok=False; cr={'content':'ERR'}
        results[h]["comp_total"] += 1
        results[h]["comp_correct"] += int(comp_ok)

        # 3) 组合推理 (允许 CoT)
        cot_prompt = build_prompt_cot(list(d['facts']), d['question'])
        try:
            ctr = client.chat([{"role":"user","content":cot_prompt}], temperature=0, max_tokens=600)
            cot_ans = extract_cot_answer(ctr['content'])
            cot_ok = normalize(d['answer']) in normalize(cot_ans)
        except Exception as e:
            print(f"  [{i}] cot API err: {e}"); cot_ok=False
        results[h]["cot_total"] += 1
        results[h]["cot_correct"] += int(cot_ok)

        if i % 10 == 0:
            print(f"  [{i}/{len(dataset)}] {h}-hop | single={single_ok} comp={comp_ok} cot={cot_ok} | "
                  f"ans='{d['answer']}'")
        time.sleep(0.3)  # 限速

    print(f"\n{'='*60}\n  组合推理崩溃边界 (DeepSeek)\n{'='*60}")
    print(f"  {'hops':>5} {'单跳':>8} {'组合(直接)':>12} {'组合(CoT)':>12} {'CoT救回':>10}")
    for h in sorted(results):
        r = results[h]
        sa = r['single_correct']/max(r['single_total'],1)
        ca = r['comp_correct']/max(r['comp_total'],1)
        cota = r['cot_correct']/max(r['cot_total'],1)
        rescue = f"+{(cota-ca)*100:.0f}pt"
        print(f"  {h:>5} {sa:>7.0%} {ca:>11.0%} {cota:>11.0%} {rescue:>10}")
    print(f"\n  解读: 直接组合崩溃 + CoT是否救回 = 推理能否延展边界")
    print(f"  保存结果到 idea4_results.json")
    with open(os.path.join(os.path.dirname(__file__),"idea4_results.json"),"w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    dry = "--dry" in sys.argv or not os.environ.get("DEEPSEEK_API_KEY")
    if dry and not os.environ.get("DEEPSEEK_API_KEY"):
        print("[未检测到 DEEPSEEK_API_KEY, 先跑 --dry 验证数据逻辑]\n")
    run(dry=dry)
