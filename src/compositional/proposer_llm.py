"""Phase 2: LLM Proposer — 用 DeepSeek 自己攻击 RAG
==================================================
目标: 生成让"逐步检索"策略也失败的组合推理任务

攻击向量 (LLM 自动发现+手工引导):
1. 多值歧义: 同一人有多种关系, 查 "A的搭档" 可能返回错误值
2. 共享实体: A→B→C 和 D→B→E, 查 "B的搭档" 有两个可能
3. 条件依赖: "A在项目X的搭档是B, A在项目Y的搭档是C"
4. 间接引用: 问题用描述而非名字指代 ("那个住在xx的人的搭档是谁")
5. 时间依赖: "A的前搭档是B, A的现搭档是C"

流程:
  Round 1: LLM生成攻击 → Solver(4策略)尝试 → 记录哪些攻击成功
  Round 2: LLM根据上轮成功的攻击模式, 生成更难的变体
  ...

用法:
  export DEEPSEEK_API_KEY=xxx
  python proposer_llm.py [--rounds 3] [--attacks-per-round 5]
"""
import os, sys, json, random, time, argparse
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here,"..","..","..","..","memory","memory-engine","memory_engine"))
sys.path.insert(0, _here)
from deepseek_client import DeepSeekClient
from pilot_incontext import normalize

# ============ LLM PROPOSER ============

ATTACK_VECTORS = [
    "多值歧义: 让同一个人在不同关系下有不同搭档,使得简单查'A的搭档'得到错误值",
    "共享实体: 让目标链和干扰链共享中间实体,使检索走错分支",
    "条件依赖: 关系带条件限定(项目/时间/地点),简单检索忽略条件",
    "间接引用: 问题不直接用名字,而是用描述('住在X的那个人'),需要先解析指代",
    "逆向陷阱: 正向链A→B→C存在,但问题问的是B→?时有歧义(B同时在两条链上)",
]

PROPOSER_SYSTEM = """你是一个对抗出题专家。你的目标是设计组合推理题目,让一个使用"逐步检索"(RAG)策略的AI回答错误。

RAG策略的工作方式:
- AI拿到问题后,从起始实体开始,每步查询知识库获取下一跳
- 知识库是简单的 key-value: 输入实体名 → 返回该实体的搭档/关系人
- AI不会看到全部事实,只能一步步查

你要利用这个策略的弱点来出题。弱点包括:
1. 查询只按实体名匹配,不区分关系类型
2. 如果一个实体在多条链上,查询可能返回错误链的值
3. 查询不理解条件/上下文
"""

def generate_attack_prompt(attack_vector, round_num, prev_successes=None):
    """生成 LLM Proposer 的 prompt"""
    context = ""
    if prev_successes:
        context = f"\n\n上轮成功的攻击模式:\n" + "\n".join(f"- {s}" for s in prev_successes[-3:])
        context += "\n请在这些成功模式基础上生成更难的变体。"

    return f"""{PROPOSER_SYSTEM}

本轮攻击方向: {attack_vector}
{context}

请生成一道题目,要求:
1. 有一组事实(至少6条),包含目标链和干扰/陷阱
2. 有一个组合推理问题(需要2-4跳)
3. 有唯一正确答案
4. 简单的逐步检索会导向错误答案(解释为什么)

严格按以下JSON格式输出(不要加其他文字):
{{
  "facts": ["事实1", "事实2", ...],
  "question": "问题",
  "answer": "正确答案(全名)",
  "trap_answer": "RAG可能给出的错误答案(全名)",
  "attack_type": "攻击类型简述",
  "explanation": "为什么逐步检索会失败"
}}"""


def parse_proposer_output(text):
    """从 LLM 输出中提取 JSON"""
    import re
    # 找 JSON 块
    text = text.strip()
    # 去掉可能的 markdown code block
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except:
        # 尝试找第一个 { 到最后一个 }
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return None


# ============ SOLVERS (adapted for LLM-generated tasks) ============

class LLMSolver:
    def __init__(self, client):
        self.c = client

    def solve_cot(self, facts, question, answer):
        """纯 CoT: 所有事实 in-context"""
        fact_lines = "\n".join(f"- {f}" for f in facts)
        prompt = (f"以下是一些关于人物关系的事实:\n{fact_lines}\n\n"
                  f"问题: {question}\n"
                  f"请一步一步推理,最后另起一行输出: '最终答案: <全名>'")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=800)
            return self._check(r['content'], answer)
        except:
            return False

    def solve_rag_naive(self, facts, question, answer):
        """朴素 RAG: 从事实构建KB, 按实体名查询(不区分关系)"""
        # 构建简单 KB: 从事实里提取 "X的Y是Z" 模式
        import re
        kb = {}  # entity -> list of (relation, target)
        for f in facts:
            m = re.match(r"(.+?)的(.+?)是(.+?)[。.]?$", f)
            if m:
                subj, rel, obj = m.group(1), m.group(2), m.group(3)
                if subj not in kb:
                    kb[subj] = []
                kb[subj].append((rel, obj))

        # 从问题提取起始实体和关系链
        # 简单策略: 找问题里第一个在KB中的实体,然后逐步查
        start_entity = None
        for entity in kb:
            if entity in question:
                if start_entity is None or question.index(entity) < question.index(start_entity):
                    start_entity = entity

        if not start_entity:
            # fallback: 用 CoT
            return self.solve_cot(facts, question, answer)

        # 逐步查 (朴素: 取第一个匹配)
        current = start_entity
        steps = []
        visited = set()
        for _ in range(8):  # 最多8跳
            if current in visited:
                break
            visited.add(current)
            if current in kb and kb[current]:
                # 朴素: 取第一个关系的目标
                rel, target = kb[current][0]
                steps.append(f"查询'{current}' → {rel}: {target}")
                current = target
            else:
                break

        # 让模型从检索结果推断答案
        steps_text = "\n".join(steps)
        prompt = (f"问题: {question}\n\n"
                  f"逐步检索结果:\n{steps_text}\n\n"
                  f"根据检索结果,答案是谁?只回答全名:")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=50)
            return self._check(r['content'], answer)
        except:
            return False

    def solve_rag_smart(self, facts, question, answer):
        """智能 RAG: 从事实构建KB, 查询时考虑关系类型"""
        import re
        kb = {}
        for f in facts:
            m = re.match(r"(.+?)的(.+?)是(.+?)[。.]?$", f)
            if m:
                subj, rel, obj = m.group(1), m.group(2), m.group(3)
                if subj not in kb:
                    kb[subj] = []
                kb[subj].append((rel, obj))

        # 从问题提取关系类型
        # 比如 "A的工作搭档的导师是谁" → 需要先查搭档,再查导师
        # 让模型分析问题结构
        analysis_prompt = (f"分析以下问题的推理链:\n问题: {question}\n\n"
                          f"请列出需要查询的步骤,每步格式: '查询 [实体] 的 [关系]'\n"
                          f"可用的实体和关系:\n" +
                          "\n".join(f"- {e}: {', '.join(r+':'+t for r,t in rels)}" for e, rels in list(kb.items())[:20]))
        try:
            r = self.c.chat([{"role":"user","content":analysis_prompt}], temperature=0, max_tokens=300)
            # 然后用全部信息让模型回答
            all_info = "\n".join(f"- {f}" for f in facts)
            final_prompt = (f"事实:\n{all_info}\n\n"
                           f"推理分析:\n{r['content']}\n\n"
                           f"问题: {question}\n最终答案(只回答全名):")
            r2 = self.c.chat([{"role":"user","content":final_prompt}], temperature=0, max_tokens=50)
            return self._check(r2['content'], answer)
        except:
            return False

    def _check(self, text, answer):
        """检查答案是否正确"""
        import re
        # 找最终答案标记
        ms = list(re.finditer(r"(?:最终答案|答案)[:：]?\s*(.+)", text))
        if ms:
            return normalize(answer) in normalize(ms[-1].group(1))
        # fallback: 全文包含
        return normalize(answer) in normalize(text)


# ============ MAIN ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3, help="对抗轮数")
    ap.add_argument("--attacks-per-round", type=int, default=5, help="每轮每种攻击向量生成几道题")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    client = DeepSeekClient()
    solver = LLMSolver(client)

    strategies = [
        ("CoT", solver.solve_cot),
        ("RAG-naive", solver.solve_rag_naive),
        ("RAG-smart", solver.solve_rag_smart),
    ]

    all_results = []
    successful_attacks = []  # 攻击成功的模式记录

    print("="*70)
    print("  Phase 2: LLM Proposer 攻击 RAG")
    print("="*70)

    for round_num in range(args.rounds):
        print(f"\n{'='*70}")
        print(f"  Round {round_num+1}/{args.rounds}")
        print(f"{'='*70}")

        round_data = {"round": round_num, "tasks": []}

        for vec_idx, attack_vec in enumerate(ATTACK_VECTORS):
            print(f"\n  攻击向量 {vec_idx+1}: {attack_vec[:30]}...")

            for attempt in range(args.attacks_per_round):
                # 生成攻击
                prompt = generate_attack_prompt(
                    attack_vec, round_num,
                    prev_successes=successful_attacks if round_num > 0 else None
                )
                try:
                    r = client.chat([{"role":"user","content":prompt}], temperature=0.7, max_tokens=1000)
                    task = parse_proposer_output(r['content'])
                except Exception as e:
                    print(f"    [{attempt}] Proposer error: {e}")
                    time.sleep(1)
                    continue

                if not task or 'facts' not in task or 'question' not in task or 'answer' not in task:
                    print(f"    [{attempt}] Invalid task format, skip")
                    continue

                # 评估各 Solver
                task_result = {
                    "attack_type": task.get('attack_type', attack_vec[:20]),
                    "question": task['question'],
                    "answer": task['answer'],
                    "trap_answer": task.get('trap_answer', '?'),
                    "explanation": task.get('explanation', ''),
                    "facts": task['facts'],
                    "strategies": {}
                }

                any_failed = False
                for strat_name, strat_fn in strategies:
                    ok = strat_fn(task['facts'], task['question'], task['answer'])
                    task_result["strategies"][strat_name] = ok
                    if not ok:
                        any_failed = True
                    time.sleep(0.3)

                status = "✓全过" if all(task_result["strategies"].values()) else "★有攻破"
                failed_strats = [k for k,v in task_result["strategies"].items() if not v]
                print(f"    [{attempt}] {status} | 失败: {failed_strats if failed_strats else '无'} | "
                      f"Q: {task['question'][:40]}...")

                if any_failed:
                    successful_attacks.append(
                        f"{task.get('attack_type','?')}: {task.get('explanation','')[:80]}"
                    )

                round_data["tasks"].append(task_result)
                time.sleep(0.5)

        all_results.append(round_data)

    # 汇总
    print(f"\n{'='*70}")
    print(f"  汇总: LLM Proposer 攻击效果")
    print(f"{'='*70}")

    total_tasks = sum(len(r['tasks']) for r in all_results)
    for strat_name, _ in strategies:
        successes = sum(
            1 for r in all_results for t in r['tasks']
            if t['strategies'].get(strat_name, False)
        )
        print(f"  {strat_name:12s}: {successes}/{total_tasks} 正确 = {successes/max(total_tasks,1):.0%}")

    # 攻击成功案例
    print(f"\n  攻击成功案例 (至少1个策略失败):")
    attack_wins = [t for r in all_results for t in r['tasks']
                   if not all(t['strategies'].values())]
    for t in attack_wins[:10]:
        failed = [k for k,v in t['strategies'].items() if not v]
        print(f"    - [{t['attack_type']}] Q: {t['question'][:50]}")
        print(f"      答案={t['answer']}, 陷阱={t['trap_answer']}, 失败策略={failed}")

    # 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'compositional', 'proposer_llm_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "rounds": args.rounds,
            "attacks_per_round": args.attacks_per_round,
            "attack_vectors": ATTACK_VECTORS,
            "results": all_results,
            "successful_attacks": successful_attacks,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")


if __name__ == "__main__":
    main()
