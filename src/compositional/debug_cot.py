"""debug: 验证权重内化模型的 CoT 原始输出 — CoT=0%是真崩还是抽取bug?"""
import os, sys, random, re
os.environ.setdefault("HF_HOME","/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0,"/root")
from idea4_weight import (make_chains, atomic_facts, q_single, q_comp_cot, norm, extract_cot)

dev="cuda"
model="Qwen/Qwen2.5-1.5B"
tok=AutoTokenizer.from_pretrained(model)
if tok.pad_token is None: tok.pad_token=tok.eos_token

rng=random.Random(42)
chains=make_chains(8,2,rng)   # 8条2-hop链
facts=atomic_facts(chains)

torch.manual_seed(42)
m=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float32).to(dev)
cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",task_type="CAUSAL_LM")
m=get_peft_model(m,cfg)
m.train()
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4)
for ep in range(12):
    random.shuffle(facts)
    for f in facts:
        ids=tok(f,return_tensors="pt").input_ids.to(dev)
        out=m(ids,labels=ids); out.loss.backward()
        opt.step(); opt.zero_grad()

m.eval()
print("="*60)
print("CoT 原始输出诊断 (权重内化, hop=2)")
print("="*60)
for ch in chains[:5]:
    prompt=q_comp_cot(ch[0],2)
    ids=tok(prompt,return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        o=m.generate(ids,max_new_tokens=400,do_sample=False,pad_token_id=tok.eos_token_id)
    raw=tok.decode(o[0][ids.shape[1]:],skip_special_tokens=True)
    ext=extract_cot(raw)
    # 链: ch[0]->ch[1]->ch[2], 期望答案ch[2]
    print(f"\n链: {ch[0]} -> {ch[1]} -> {ch[2]}")
    print(f"期望答案: {ch[2]} | 抽取出: {repr(ext[:40])} | 命中: {norm(ch[2]) in norm(ext)}")
    print(f"--- 单跳验证: {ch[0]}的搭档 -> ", end="")
    sids=tok(q_single(ch[0]),return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        so=m.generate(sids,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
    print(repr(tok.decode(so[0][sids.shape[1]:],skip_special_tokens=True)[:30]))
    print(f"--- CoT原始(前250字):\n{raw[:250]}")
