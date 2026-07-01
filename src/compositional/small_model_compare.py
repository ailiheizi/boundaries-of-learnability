"""小模型对比: 用本地 Qwen-1.5B/3B 测试 LLM Proposer 生成的攻击
================================================================
从 proposer_llm_large.json 加载攻击任务, 用本地模型 (via llama.cpp) 跑 CoT。
对比 DeepSeek vs 小模型在同一攻击下的表现。

用法:
  # 先启动 llama-server
  # llama-server -m qwen2.5-1.5b-q4.gguf --port 8080 -ngl 99 -c 4096
  python small_model_compare.py [--port 8080] [--tag qwen1.5b] [--n 20]
"""
import os, sys, json, time, argparse, re
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from pilot_incontext import normalize

try:
    from local_client import LocalClient
except ImportError:
    # fallback: 简单 OpenAI-compatible client
    import urllib.request
    class LocalClient:
        def __init__(self, port=8080):
            self.url = f"http://localhost:{port}/v1/chat/completions"
        def chat(self, messages, temperature=0, max_tokens=500):
            data = json.dumps({"messages": messages, "temperature": temperature,
                              "max_tokens": max_tokens}).encode()
            req = urllib.request.Request(self.url, data=data,
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                r = json.loads(resp.read())
                return {"content": r["choices"][0]["message"]["content"]}


def solve_cot_local(client, facts, question):
    """本地模型 CoT"""
    fact_lines = "\n".join(f"- {f}" for f in facts)
    prompt = (f"以下是一些关于人物关系的事实:\n{fact_lines}\n\n"
              f"问题: {question}\n"
              f"请一步一步推理,最后另起一行输出: '最终答案: <全名>'")
    try:
        r = client.chat([{"role":"user","content":prompt}], temperature=0, max_tokens=500)
        return r['content']
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer(text):
    """从输出提取答案"""
    ms = list(re.finditer(r"(?:最终答案|答案)[:：]?\s*(.+)", text))
    if ms:
        return ms[-1].group(1).strip()
    # fallback: 最后一行
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tag", default="qwen1.5b")
    ap.add_argument("--n", type=int, default=20, help="从攻击集抽样多少道")
    ap.add_argument("--input", default=os.path.join(_here,"..","..","results","compositional","proposer_llm_large.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 加载攻击数据
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有有效任务 (需要有 facts)
    all_tasks = []
    for rd in data['results']:
        for t in rd['tasks']:
            # 检查是否有需要的字段
            if 'question' in t and 'answer' in t:
                all_tasks.append(t)

    # 检查是否有 facts 字段
    has_facts = any('facts' in t for t in all_tasks)
    if not has_facts:
        print("ERROR: 攻击数据里没有保存 facts 字段!")
        print("需要修改 proposer_llm.py 保存 facts, 然后重新生成")
        sys.exit(1)

    # 抽样
    import random
    random.seed(42)
    tasks = random.sample(all_tasks, min(args.n, len(all_tasks)))

    print(f"="*60)
    print(f"  小模型对比: {args.tag} (port={args.port})")
    print(f"  测试 {len(tasks)} 道 LLM 生成的攻击题")
    print(f"="*60)

    client = LocalClient(port=args.port)

    # 先测试连接
    try:
        test = client.chat([{"role":"user","content":"hi"}], max_tokens=5)
        print(f"  连接OK: {test['content'][:20]}")
    except Exception as e:
        print(f"  ERROR: 无法连接到 localhost:{args.port}: {e}")
        sys.exit(1)

    results = {"correct": 0, "total": 0, "details": []}

    for i, t in enumerate(tasks):
        facts = t['facts']
        question = t['question']
        answer = t['answer']

        output = solve_cot_local(client, facts, question)
        extracted = extract_answer(output)
        hit = normalize(answer) in normalize(extracted)

        results['correct'] += int(hit)
        results['total'] += 1

        if i % 5 == 0 or not hit:
            print(f"  [{i+1}/{len(tasks)}] {'✓' if hit else '✗'} "
                  f"Q: {question[:35]}... → {extracted[:20]} (正确={answer})")

        results['details'].append({
            "question": question,
            "answer": answer,
            "model_output": extracted[:100],
            "correct": hit,
            "deepseek_cot": t['strategies'].get('CoT', None),
        })
        time.sleep(0.1)

    acc = results['correct'] / max(results['total'], 1)
    deepseek_acc = sum(1 for d in results['details'] if d['deepseek_cot']) / max(results['total'], 1)

    print(f"\n{'='*60}")
    print(f"  结果: {args.tag}")
    print(f"{'='*60}")
    print(f"  {args.tag} CoT: {results['correct']}/{results['total']} = {acc:.0%}")
    print(f"  DeepSeek CoT: {deepseek_acc:.0%} (同一批题)")
    print(f"  差距: {deepseek_acc - acc:+.0%}")

    # 保存
    out_path = args.out or os.path.join(_here, '..', '..', 'results', 'compositional',
                                        f'small_model_{args.tag}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"model": args.tag, "accuracy": acc, "deepseek_accuracy": deepseek_acc,
                   "n": results['total'], "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  保存: {out_path}")


if __name__ == "__main__":
    main()
