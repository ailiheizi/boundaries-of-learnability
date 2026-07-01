"""Proposer-Solver 对抗 MVP
=========================
Proposer: 生成越来越难的多跳组合推理任务
Solver:   4 种策略同时尝试, 记录各自的崩溃边界
  1. 纯 CoT (in-context, 所有事实在 prompt)
  2. 纯 RAG (逐步检索, 事实不在 prompt)
  3. RAG+CoT 混合 (先规划再检索)
  4. 多层级 (CoT 先试, 失败回退 RAG)

对抗逻辑:
  Round N → Proposer 增加难度 → 4个 Solver 各做一遍 → 记录成功率
  难度维度: 链长 / 干扰链数 / 关系多样性 / 实体混淆度

用法:
  export DEEPSEEK_API_KEY=xxx
  python proposer_solver_mvp.py [--rounds 5] [--samples 10]
"""
import os, sys, json, random, time, argparse
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here,"..","..","..","..","memory","memory-engine","memory_engine"))
sys.path.insert(0, _here)
from deepseek_client import DeepSeekClient
from pilot_incontext import normalize

# ============ PROPOSER ============

FIRST = ["Zelph","Quorin","Vandel","Tovak","Yelena","Karsh","Oslo","Veyra","Drennan","Sythe",
         "Marlow","Pendric","Casimir","Wren","Galen","Ophira","Lucan","Brenna","Caldus","Mirae",
         "Soren","Elric","Thalia","Riven","Kael","Nyra","Dorian","Isolde","Fenris","Lyra"]
SUR = ["Voss","Karr","Thorne","Belmont","Ashby","Cregan","Dunmore","Fenwick","Halloran","Ives",
       "Quill","Sterling","Underwood","Vance","Whitlock","Yarrow","Zane","Brackett","Coyle","Dane",
       "Mercer","Holt","Graves","Sinclair","Aldridge","Blackwood","Crane","Farrow","Knox","Loring"]

RELATIONS = [
    ("的工作搭档是", "的工作搭档"),
    ("的导师是", "的导师"),
    ("的邻居是", "的邻居"),
    ("的合作者是", "的合作者"),
]

def rand_entity(used, rng, confuse_level=0):
    """confuse_level: 0=正常随机, 1=名字首字母相似, 2=共享姓"""
    attempts = 0
    while attempts < 200:
        if confuse_level >= 2 and used:
            # 复用已有姓
            existing = list(used)
            base_sur = rng.choice(existing).split()[-1] if rng.random() < 0.5 else rng.choice(SUR)
            name = f"{rng.choice(FIRST)} {base_sur}"
        elif confuse_level >= 1 and used:
            # 首字母相近
            first = rng.choice(FIRST)
            name = f"{first} {rng.choice(SUR)}"
        else:
            name = f"{rng.choice(FIRST)} {rng.choice(SUR)}"
        if name not in used:
            used.add(name)
            return name
        attempts += 1
    # fallback
    name = f"Entity{len(used)}"
    used.add(name)
    return name


class Proposer:
    """根据难度级别生成推理任务"""

    DIFFICULTY_CONFIG = [
        # (hops, n_distractor, n_relations, confuse_level, description)
        (2, 1, 1, 0, "2-hop, 1干扰链, 单关系"),
        (3, 2, 1, 0, "3-hop, 2干扰链, 单关系"),
        (4, 3, 1, 0, "4-hop, 3干扰链, 单关系"),
        (5, 4, 1, 0, "5-hop, 4干扰链, 单关系"),
        (4, 5, 2, 0, "4-hop, 5干扰链, 2种关系"),
        (5, 6, 2, 1, "5-hop, 6干扰链, 2种关系, 名字相似"),
        (6, 8, 3, 1, "6-hop, 8干扰链, 3种关系, 名字相似"),
        (7, 10, 4, 2, "7-hop, 10干扰链, 4种关系, 共享姓"),
        (8, 12, 4, 2, "8-hop, 12干扰链, 4种关系, 共享姓"),
    ]

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def generate(self, difficulty_level, n_samples=10):
        """生成指定难度的 n_samples 个任务"""
        level = min(difficulty_level, len(self.DIFFICULTY_CONFIG)-1)
        hops, n_dist, n_rels, confuse, desc = self.DIFFICULTY_CONFIG[level]
        rels = RELATIONS[:n_rels]

        tasks = []
        for _ in range(n_samples):
            used = set()
            # 目标链
            target_rel = self.rng.choice(rels)
            entities = [rand_entity(used, self.rng, confuse) for _ in range(hops+1)]
            target_facts = [(entities[i], target_rel[0], entities[i+1]) for i in range(hops)]

            # 干扰链 (可能用不同关系)
            distractor_facts = []
            for _ in range(n_dist):
                d_rel = self.rng.choice(rels)
                d_ents = [rand_entity(used, self.rng, confuse) for _ in range(hops+1)]
                for i in range(hops):
                    distractor_facts.append((d_ents[i], d_rel[0], d_ents[i+1]))

            all_facts = target_facts + distractor_facts
            self.rng.shuffle(all_facts)

            # 问题
            rel_chain = target_rel[1] * hops
            question = f"{entities[0]}{rel_chain}是谁?"
            answer = entities[hops]

            # 知识库 (for RAG): 按 (subject, relation_prefix) 索引
            kb = {}
            for s, r, o in all_facts:
                kb[(s, r.rstrip("是")+"是")] = o
                kb[s] = o  # 简化: subject -> object (假设每人只有一个搭档)

            tasks.append({
                "hops": hops,
                "target_facts": target_facts,
                "all_facts": all_facts,
                "question": question,
                "answer": answer,
                "first_entity": entities[0],
                "relation": target_rel,
                "kb": {s: o for s,r,o in target_facts},  # 只存目标链的 KB
                "kb_full": {s: o for s,r,o in all_facts},
                "difficulty": level,
                "description": desc,
            })
        return tasks


# ============ SOLVER ============

class Solver:
    def __init__(self, client):
        self.c = client

    def solve_cot(self, task):
        """策略1: 纯 CoT (所有事实 in-context)"""
        facts = task['all_facts']
        random.shuffle(facts)
        fact_lines = "\n".join(f"- {s}{r}{o}。" for s,r,o in facts)
        prompt = (f"以下是一些关于虚构人物关系的事实:\n{fact_lines}\n\n"
                  f"问题: {task['question']}\n"
                  f"请一步一步追踪关系链,逐步推理每一跳。\n"
                  f"最后必须另起一行输出: '最终答案: <全名>'")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=800)
            return self._extract(r['content'], task['answer'])
        except:
            return False

    def solve_rag(self, task):
        """策略2: 纯 RAG (逐步检索)"""
        kb = task['kb']
        current = task['first_entity']
        rag_steps = []
        for hop in range(task['hops']):
            if current in kb:
                rag_steps.append(f"查询'{current}{task['relation'][1]}': 结果是{kb[current]}。")
                current = kb[current]
            else:
                rag_steps.append(f"查询'{current}{task['relation'][1]}': 未找到。")
                break
        prompt = (f"你需要回答: {task['question']}\n\n"
                  f"你没有这些人的背景知识,但可以逐步查询数据库。以下是查询结果:\n"
                  + "\n".join(rag_steps) + "\n\n"
                  f"根据以上检索结果,最终答案是谁?只回答全名:")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=30)
            return self._extract_short(r['content'], task['answer'])
        except:
            return False

    def solve_rag_cot(self, task):
        """策略3: RAG+CoT (先规划再检索验证)"""
        kb = task['kb']
        current = task['first_entity']
        rag_steps = []
        for hop in range(task['hops']):
            if current in kb:
                rag_steps.append(f"第{hop+1}步: 查询'{current}'→ 得到'{kb[current]}'")
                current = kb[current]
            else:
                rag_steps.append(f"第{hop+1}步: 查询'{current}'→ 未找到")
                break
        prompt = (f"问题: {task['question']}\n\n"
                  f"推理计划: 需要追踪{task['hops']}跳关系链。\n"
                  f"检索结果:\n" + "\n".join(rag_steps) + "\n\n"
                  f"请验证推理链的每一步是否正确连接,然后给出最终答案。\n"
                  f"最终答案:")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=200)
            return self._extract_short(r['content'], task['answer'])
        except:
            return False

    def solve_multilevel(self, task):
        """策略4: 多层级 (CoT先试, 不确定时回退RAG)"""
        # 先用 CoT
        cot_ok = self.solve_cot(task)
        if cot_ok:
            return True
        # CoT 失败, 回退 RAG
        return self.solve_rag(task)

    def _extract(self, text, answer):
        """检查 CoT 输出是否包含正确答案"""
        import re
        # 找最终答案标记
        ms = list(re.finditer(r"(?:最终答案|答案)[:：]?\s*(.+)", text))
        if ms:
            return normalize(answer) in normalize(ms[-1].group(1))
        return normalize(answer) in normalize(text.split('\n')[-1])

    def _extract_short(self, text, answer):
        return normalize(answer) in normalize(text)


# ============ MAIN LOOP ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5, help="对抗轮数(难度级别0~rounds-1)")
    ap.add_argument("--samples", type=int, default=10, help="每轮每个难度的样本数")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    client = DeepSeekClient()
    proposer = Proposer(seed=args.seed)
    solver = Solver(client)

    strategies = [
        ("CoT(in-ctx)", solver.solve_cot),
        ("RAG(逐步)", solver.solve_rag),
        ("RAG+CoT", solver.solve_rag_cot),
        ("多层级", solver.solve_multilevel),
    ]

    all_results = []

    print("="*70)
    print("  Proposer-Solver 对抗 MVP")
    print("="*70)

    for level in range(min(args.rounds, len(Proposer.DIFFICULTY_CONFIG))):
        tasks = proposer.generate(level, n_samples=args.samples)
        desc = Proposer.DIFFICULTY_CONFIG[level][4]
        print(f"\n--- Round {level}: {desc} (n={args.samples}) ---")

        round_results = {"level": level, "description": desc, "hops": tasks[0]['hops']}

        for strat_name, strat_fn in strategies:
            correct = 0
            for t in tasks:
                ok = strat_fn(t)
                correct += int(ok)
                time.sleep(0.3)
            acc = correct / len(tasks)
            round_results[strat_name] = {"correct": correct, "total": len(tasks), "accuracy": acc}
            print(f"  {strat_name:12s}: {correct}/{len(tasks)} = {acc:.0%}")

        all_results.append(round_results)

    # 汇总
    print(f"\n{'='*70}")
    print(f"  汇总: 各策略随难度的崩溃曲线")
    print(f"{'='*70}")
    header = f"  {'Level':>5} {'描述':20s}"
    for name, _ in strategies:
        header += f" {name:>10}"
    print(header)
    for r in all_results:
        row = f"  {r['level']:>5} {r['description']:20s}"
        for name, _ in strategies:
            row += f" {r[name]['accuracy']:>9.0%}"
        print(row)

    # 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'compositional', 'proposer_solver_mvp.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"seed": args.seed, "samples_per_round": args.samples, "results": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")


if __name__ == "__main__":
    main()
