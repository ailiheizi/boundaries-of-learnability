"""小模型版组合推理崩溃边界 (本地 CPU + transformers)

复用 pilot_incontext 的数据生成逻辑, 把 API 换成本地小模型。
对比不同规模小模型 vs DeepSeek 的崩溃曲线 + CoT 救援。

用法:
  set HF_ENDPOINT=https://hf-mirror.com   # 国内镜像加速
  uv run --extra-index-url https://download.pytorch.org/whl/cpu \
      --with torch --with transformers --with accelerate \
      python pilot_smallmodel.py Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations
import os, sys, json, random, time, re

# 复用同目录 pilot 的数据生成
sys.path.insert(0, os.path.dirname(__file__))
from pilot_incontext import (gen_dataset, build_prompt, build_prompt_cot,
                             build_singlehop_check, extract_cot_answer, normalize)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(model_name):
    print(f"加载 {model_name} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    print(f"  加载完成 {time.time()-t0:.0f}s")
    return tok, model

@torch.no_grad()
def chat(tok, model, prompt, max_new_tokens=30):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][n_in:], skip_special_tokens=True)

def run(model_name, n_per_hop=15):
    dataset = gen_dataset(n_per_hop=n_per_hop)
    tok, model = load(model_name)
    hops = sorted(set(d['hops'] for d in dataset))
    R = {h: dict(single_c=0, single_t=0, comp_c=0, comp_t=0, cot_c=0, cot_t=0) for h in hops}

    t0 = time.time()
    for i, d in enumerate(dataset):
        h = d['hops']
        # 单跳对照
        sf = d['target_facts'][0]
        sok = normalize(sf[2]) in normalize(chat(tok, model, build_singlehop_check(sf), 30))
        R[h]['single_t'] += 1; R[h]['single_c'] += int(sok)
        # 直接组合
        cok = normalize(d['answer']) in normalize(chat(tok, model, build_prompt(list(d['facts']), d['question']), 30))
        R[h]['comp_t'] += 1; R[h]['comp_c'] += int(cok)
        # CoT
        cot_raw = chat(tok, model, build_prompt_cot(list(d['facts']), d['question']), 600)
        cotok = normalize(d['answer']) in normalize(extract_cot_answer(cot_raw))
        R[h]['cot_t'] += 1; R[h]['cot_c'] += int(cotok)
        if i % 5 == 0:
            el = time.time()-t0
            print(f"  [{i}/{len(dataset)}] {h}-hop s={sok} c={cok} cot={cotok} | {el:.0f}s ({el/(i+1):.1f}s/it)")

    short = model_name.split('/')[-1]
    print(f"\n{'='*60}\n  组合推理崩溃边界: {short}\n{'='*60}")
    print(f"  {'hops':>5} {'单跳':>8} {'组合(直接)':>12} {'组合(CoT)':>12} {'CoT救回':>10}")
    summary = {}
    for h in hops:
        r = R[h]
        sa, ca, cota = r['single_c']/r['single_t'], r['comp_c']/r['comp_t'], r['cot_c']/r['cot_t']
        summary[h] = dict(single=sa, comp=ca, cot=cota)
        print(f"  {h:>5} {sa:>7.0%} {ca:>11.0%} {cota:>11.0%} {f'+{(cota-ca)*100:.0f}pt':>10}")

    outp = os.path.join(os.path.dirname(__file__), "..", "results", f"smallmodel_{short}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "n_per_hop": n_per_hop, "results": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n  保存到 {outp}")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    run(model_name, n)
