"""防御实验: 强化 RAG Solver 对抗 LLM Proposer 攻击
=================================================
设计 3 层防御 Solver, 测试能否防住已知攻击:

1. RAG-disambig: 当检索到多个值时, 让模型根据问题上下文选择正确分支
2. RAG-condaware: 检索时带上条件限定(时间/项目/地点), 过滤不符合条件的结果
3. RAG-resolve: 先做指代解析(把描述转为实体名), 再走 RAG

对比:
- RAG-naive (baseline, 已知 45%)
- RAG-smart (baseline, 已知 57%)
- RAG-disambig (防御: 消歧)
- RAG-condaware (防御: 条件感知)
- RAG-resolve (防御: 指代解析)
- RAG-full (三重防御全开)

用法:
  export DEEPSEEK_API_KEY=xxx
  python defense_experiment.py [--input results/proposer_llm_results.json]
"""
import os, sys, json, re, time, argparse
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here,"..","..","..","..","memory","memory-engine","memory_engine"))
from deepseek_client import DeepSeekClient
from pilot_incontext import normalize


class DefenseSolver:
    """增强版 RAG Solver, 具备消歧/条件感知/指代解析"""

    def __init__(self, client):
        self.c = client

    def _parse_kb(self, facts):
        """从事实列表构建结构化 KB"""
        kb = {}  # entity -> [(relation, target, condition)]
        for f in facts:
            # 匹配 "X的Y是Z" 或带条件 "在条件下,X的Y是Z"
            m = re.match(r"(?:在(.+?)[中下](?:[，,])?\s*)?(.+?)的(.+?)是(.+?)[。.]?$", f)
            if m:
                cond, subj, rel, obj = m.group(1), m.group(2), m.group(3), m.group(4)
                if subj not in kb:
                    kb[subj] = []
                kb[subj].append({"rel": rel, "target": obj, "condition": cond, "raw": f})
            else:
                # fallback: 更宽松的匹配
                m2 = re.match(r"(.+?)的(.+?)是(.+?)[。.]?$", f)
                if m2:
                    subj, rel, obj = m2.group(1), m2.group(2), m2.group(3)
                    if subj not in kb:
                        kb[subj] = []
                    kb[subj].append({"rel": rel, "target": obj, "condition": None, "raw": f})
        return kb

    def _resolve_reference(self, question, facts):
        """指代解析: 把描述性引用转为具体实体名"""
        prompt = (f"以下是一些事实:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
                  f"问题: {question}\n\n"
                  f"请识别问题中的指代表达(如'住在X的那个人'、'那个医生'等),将其替换为具体人名。\n"
                  f"如果问题中已经是具体人名,则原样返回。\n"
                  f"只输出替换后的问题,不要解释:")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=100)
            resolved = r['content'].strip()
            # 如果返回的太长或明显是解释,用原问题
            if len(resolved) > len(question) * 2 or "解释" in resolved:
                return question
            return resolved
        except:
            return question

    def _disambiguate(self, entity, entries, question, chain_so_far):
        """消歧: 当一个实体有多条关系时,选择正确的"""
        if len(entries) == 1:
            return entries[0]['target']

        prompt = (f"知识库中 '{entity}' 有以下关系:\n" +
                  "\n".join(f"  {i+1}. {e['rel']}: {e['target']}" +
                           (f" (条件: {e['condition']})" if e['condition'] else "")
                           for i, e in enumerate(entries)) +
                  f"\n\n原始问题: {question}\n"
                  f"已走过的链: {' → '.join(chain_so_far) if chain_so_far else '(起点)'}\n\n"
                  f"根据问题语境,应该选择哪个关系?只回答目标人名:")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=50)
            chosen = r['content'].strip().rstrip("。.")
            # 匹配最接近的
            for e in entries:
                if normalize(e['target']) in normalize(chosen) or normalize(chosen) in normalize(e['target']):
                    return e['target']
            # fallback: 返回第一个
            return entries[0]['target']
        except:
            return entries[0]['target']

    def _filter_by_condition(self, entries, question):
        """条件过滤: 如果问题含条件,只保留匹配的"""
        # 从问题提取条件关键词
        cond_keywords = []
        cond_patterns = [
            r"在(.+?)(?:中|下|里|的)",
            r"(\d{4}年)",
            r"项目(.+?)(?:中|的|里)",
        ]
        for p in cond_patterns:
            ms = re.findall(p, question)
            cond_keywords.extend(ms)

        if not cond_keywords:
            return entries

        # 过滤
        filtered = []
        for e in entries:
            if e['condition']:
                if any(k in e['condition'] for k in cond_keywords):
                    filtered.append(e)
            else:
                filtered.append(e)  # 无条件的保留

        return filtered if filtered else entries  # 没有匹配时 fallback 全部

    # ============ 各防御策略 ============

    def solve_rag_disambig(self, facts, question, answer):
        """消歧 RAG: 多值时让模型选择"""
        kb = self._parse_kb(facts)
        # 找起始实体
        start = self._find_start_entity(kb, question)
        if not start:
            return self._fallback_cot(facts, question, answer)

        current = start
        chain = [current]
        for _ in range(8):
            if current not in kb:
                break
            entries = kb[current]
            target = self._disambiguate(current, entries, question, chain)
            chain.append(target)
            current = target

        # 让模型从链中提取答案
        return self._answer_from_chain(chain, question, answer)

    def solve_rag_condaware(self, facts, question, answer):
        """条件感知 RAG: 检索时过滤条件"""
        kb = self._parse_kb(facts)
        start = self._find_start_entity(kb, question)
        if not start:
            return self._fallback_cot(facts, question, answer)

        current = start
        chain = [current]
        for _ in range(8):
            if current not in kb:
                break
            entries = self._filter_by_condition(kb[current], question)
            if len(entries) > 1:
                target = entries[0]['target']  # 条件过滤后取第一个
            elif entries:
                target = entries[0]['target']
            else:
                break
            chain.append(target)
            current = target

        return self._answer_from_chain(chain, question, answer)

    def solve_rag_resolve(self, facts, question, answer):
        """指代解析 RAG: 先解析指代再查"""
        resolved_q = self._resolve_reference(question, facts)
        kb = self._parse_kb(facts)
        start = self._find_start_entity(kb, resolved_q)
        if not start:
            # 尝试原问题
            start = self._find_start_entity(kb, question)
        if not start:
            return self._fallback_cot(facts, question, answer)

        current = start
        chain = [current]
        for _ in range(8):
            if current not in kb:
                break
            entries = kb[current]
            target = entries[0]['target']
            chain.append(target)
            current = target

        return self._answer_from_chain(chain, question, answer)

    def solve_rag_full(self, facts, question, answer):
        """三重防御: 指代解析 + 条件过滤 + 消歧"""
        # Step 1: 指代解析
        resolved_q = self._resolve_reference(question, facts)

        # Step 2: 构建 KB
        kb = self._parse_kb(facts)
        start = self._find_start_entity(kb, resolved_q)
        if not start:
            start = self._find_start_entity(kb, question)
        if not start:
            return self._fallback_cot(facts, question, answer)

        # Step 3: 逐步检索 (条件过滤 + 消歧)
        current = start
        chain = [current]
        for _ in range(8):
            if current not in kb:
                break
            entries = self._filter_by_condition(kb[current], resolved_q)
            if len(entries) > 1:
                target = self._disambiguate(current, entries, resolved_q, chain)
            elif entries:
                target = entries[0]['target']
            else:
                break
            chain.append(target)
            current = target

        return self._answer_from_chain(chain, question, answer)

    # ============ 辅助方法 ============

    def _find_start_entity(self, kb, question):
        """从问题中找起始实体"""
        # 策略: 找问题里第一个出现在 KB 中的实体
        best = None
        best_pos = 999
        for entity in kb:
            pos = question.find(entity)
            if pos >= 0 and pos < best_pos:
                best = entity
                best_pos = pos
        return best

    def _answer_from_chain(self, chain, question, answer):
        """从推理链提取答案并验证"""
        if len(chain) >= 2:
            # 最后一个实体作为答案
            last = chain[-1]
            return normalize(answer) in normalize(last)
        return False

    def _fallback_cot(self, facts, question, answer):
        """fallback: 全事实 CoT"""
        fact_lines = "\n".join(f"- {f}" for f in facts)
        prompt = (f"事实:\n{fact_lines}\n\n问题: {question}\n"
                  f"请逐步推理,最后输出'最终答案: <全名>'")
        try:
            r = self.c.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=500)
            ms = list(re.finditer(r"(?:最终答案|答案)[:：]?\s*(.+)", r['content']))
            if ms:
                return normalize(answer) in normalize(ms[-1].group(1))
            return normalize(answer) in normalize(r['content'])
        except:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(_here,"..","..","results","compositional","proposer_llm_results.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    # 加载攻击任务
    with open(args.input, 'r', encoding='utf-8') as f:
        attack_data = json.load(f)

    client = DeepSeekClient()
    defender = DefenseSolver(client)

    strategies = [
        ("RAG-disambig", defender.solve_rag_disambig),
        ("RAG-condaware", defender.solve_rag_condaware),
        ("RAG-resolve", defender.solve_rag_resolve),
        ("RAG-full", defender.solve_rag_full),
    ]

    # 收集所有有效任务
    tasks = []
    for round_data in attack_data['results']:
        for t in round_data['tasks']:
            # 需要从保存的结果里重建 facts
            # 但原始存储只有 strategies 结果,没有 facts...
            pass

    # 问题: 原始 proposer_llm_results.json 没有保存 facts!
    # 需要重新生成攻击 + 同时跑防御

    print("="*70)
    print("  防御实验: 需要重新生成攻击 (原结果未保存 facts)")
    print("  改用: 生成新攻击 + 同时评估所有 Solver")
    print("="*70)

    # 重新生成攻击并评估
    from proposer_llm import ATTACK_VECTORS, generate_attack_prompt, parse_proposer_output, LLMSolver

    base_solver = LLMSolver(client)

    all_strategies = [
        ("CoT", base_solver.solve_cot),
        ("RAG-naive", base_solver.solve_rag_naive),
        ("RAG-disambig", defender.solve_rag_disambig),
        ("RAG-condaware", defender.solve_rag_condaware),
        ("RAG-resolve", defender.solve_rag_resolve),
        ("RAG-full", defender.solve_rag_full),
    ]

    results = {name: {"correct": 0, "total": 0} for name, _ in all_strategies}
    attack_details = []

    n_attacks = 3  # 每向量生成数
    for vec_idx, attack_vec in enumerate(ATTACK_VECTORS):
        print(f"\n  攻击向量 {vec_idx+1}: {attack_vec[:40]}...")
        for attempt in range(n_attacks):
            prompt = generate_attack_prompt(attack_vec, 0)
            try:
                r = client.chat([{"role":"user","content":prompt}], temperature=0.7, max_tokens=1000)
                task = parse_proposer_output(r['content'])
            except Exception as e:
                print(f"    [{attempt}] Proposer error: {e}"); continue

            if not task or 'facts' not in task or 'question' not in task or 'answer' not in task:
                print(f"    [{attempt}] Invalid format, skip"); continue

            # 评估所有策略
            task_result = {"attack_type": task.get('attack_type',''), "question": task['question'],
                          "answer": task['answer'], "strategies": {}}

            for strat_name, strat_fn in all_strategies:
                ok = strat_fn(task['facts'], task['question'], task['answer'])
                task_result["strategies"][strat_name] = ok
                results[strat_name]["correct"] += int(ok)
                results[strat_name]["total"] += 1
                time.sleep(0.3)

            failed = [k for k,v in task_result["strategies"].items() if not v]
            defended = [k for k,v in task_result["strategies"].items() if v and k.startswith("RAG-")]
            status = "★" if failed else "✓"
            print(f"    [{attempt}] {status} 失败={failed[:3]} 防住={defended[:3]}")
            attack_details.append(task_result)
            time.sleep(0.5)

    # 汇总
    print(f"\n{'='*70}")
    print(f"  防御实验汇总 (n={results['CoT']['total']})")
    print(f"{'='*70}")
    print(f"  {'策略':15s} {'正确':>6} {'总数':>6} {'准确率':>8}")
    for name, _ in all_strategies:
        r = results[name]
        acc = r['correct'] / max(r['total'], 1)
        print(f"  {name:15s} {r['correct']:>6} {r['total']:>6} {acc:>7.0%}")

    # 防御提升
    print(f"\n  防御提升 (vs RAG-naive baseline):")
    baseline_acc = results['RAG-naive']['correct'] / max(results['RAG-naive']['total'], 1)
    for name, _ in all_strategies:
        if name.startswith("RAG-") and name != "RAG-naive":
            acc = results[name]['correct'] / max(results[name]['total'], 1)
            delta = acc - baseline_acc
            print(f"    {name:15s}: {acc:.0%} (Δ={delta:+.0%})")

    # 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'compositional', 'defense_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"results": results, "details": attack_details}, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")


if __name__ == "__main__":
    main()
