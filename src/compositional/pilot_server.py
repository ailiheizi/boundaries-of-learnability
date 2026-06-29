"""组合推理崩溃边界 — 通用版 (可指向 DeepSeek 或本地 llama-server)

复用 pilot_incontext 的数据生成 + 反作弊逻辑, client 可换:
  --backend local   -> 本地 llama-server (Vulkan GPU 加速)
  --backend deepseek -> DeepSeek API

用法:
  # 本地 (先起 llama-server)
  python pilot_server.py --backend local --tag qwen1.5b --n 8
  # DeepSeek
  DEEPSEEK_API_KEY=xxx python pilot_server.py --backend deepseek --tag deepseek --n 15
"""
from __future__ import annotations
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from pilot_incontext import (gen_dataset, build_prompt, build_prompt_cot,
                             build_singlehop_check, extract_cot_answer, normalize)

def get_client(backend):
    if backend == "local":
        from local_client import LocalServerClient
        return LocalServerClient()
    elif backend == "deepseek":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "memory", "memory-engine", "memory_engine"))
        from deepseek_client import DeepSeekClient
        return DeepSeekClient()
    raise ValueError(backend)

def run(backend, tag, n_per_hop):
    dataset = gen_dataset(n_per_hop=n_per_hop)
    client = get_client(backend)
    hops = sorted(set(d['hops'] for d in dataset))
    R = {h: dict(single_c=0, single_t=0, comp_c=0, comp_t=0, cot_c=0, cot_t=0) for h in hops}
    t0 = time.time()
    for i, d in enumerate(dataset):
        h = d['hops']
        sf = d['target_facts'][0]
        try:
            sok = normalize(sf[2]) in normalize(client.chat([{"role":"user","content":build_singlehop_check(sf)}], 0, 30)['content'])
        except Exception as e: print(f"  [{i}] single err {e}"); sok=False
        R[h]['single_t']+=1; R[h]['single_c']+=int(sok)
        try:
            cok = normalize(d['answer']) in normalize(client.chat([{"role":"user","content":build_prompt(list(d['facts']),d['question'])}], 0, 30)['content'])
        except Exception as e: print(f"  [{i}] comp err {e}"); cok=False
        R[h]['comp_t']+=1; R[h]['comp_c']+=int(cok)
        try:
            cot_raw = client.chat([{"role":"user","content":build_prompt_cot(list(d['facts']),d['question'])}], 0, 600)['content']
            cotok = normalize(d['answer']) in normalize(extract_cot_answer(cot_raw))
        except Exception as e: print(f"  [{i}] cot err {e}"); cotok=False
        R[h]['cot_t']+=1; R[h]['cot_c']+=int(cotok)
        if i % 3 == 0:
            el=time.time()-t0
            print(f"  [{i}/{len(dataset)}] {h}-hop s={sok} c={cok} cot={cotok} | {el:.0f}s ({el/(i+1):.1f}s/it)", flush=True)
    print(f"\n{'='*60}\n  组合推理崩溃边界: {tag}\n{'='*60}")
    print(f"  {'hops':>5} {'单跳':>8} {'组合(直接)':>12} {'组合(CoT)':>12} {'CoT救回':>10}")
    summary={}
    for h in hops:
        r=R[h]; sa,ca,cota = r['single_c']/r['single_t'], r['comp_c']/r['comp_t'], r['cot_c']/r['cot_t']
        summary[h]=dict(single=sa,comp=ca,cot=cota)
        print(f"  {h:>5} {sa:>7.0%} {ca:>11.0%} {cota:>11.0%} {f'+{(cota-ca)*100:.0f}pt':>10}")
    outp=os.path.join(os.path.dirname(__file__),"..","results",f"{tag}.json")
    json.dump({"backend":backend,"tag":tag,"n_per_hop":n_per_hop,"results":summary},
              open(outp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  保存 {outp}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local","deepseek"])
    ap.add_argument("--tag", default="local")
    ap.add_argument("--n", type=int, default=8)
    a=ap.parse_args()
    run(a.backend, a.tag, a.n)
