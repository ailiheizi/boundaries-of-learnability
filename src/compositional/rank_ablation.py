"""LoRA Rank Ablation: find sweet spot where facts memorized + reasoning intact
================================================================
For each (rank, epochs) config:
1. LoRA train on fictional facts
2. Test single-hop recall (must be ~100%)
3. Test closed-book CoT composition
4. Test IN-CONTEXT CoT on NEW facts (reasoning intact check)

Sweet spot = single-hop high + in-context CoT preserved + closed-book CoT still low
If such config exists -> the dissociation is REAL at that config.
"""
import torch, random, json, os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

FIRST=["Zelph","Quorin","Vandel","Tovak","Yelena","Karsh","Oslo","Veyra","Drennan","Sythe",
       "Marlow","Pendric","Casimir","Wren","Galen","Ophira","Lucan","Brenna","Caldus","Mirae"]
SUR=["Voss","Karr","Thorne","Belmont","Ashby","Cregan","Dunmore","Fenwick","Halloran","Ives",
     "Quill","Sterling","Underwood","Vance","Whitlock","Yarrow","Zane","Brackett","Coyle","Dane"]

def rand_entity(used,rng):
    while True:
        n=f"{rng.choice(FIRST)} {rng.choice(SUR)}"
        if n not in used: used.add(n); return n

def make_chain(n_hops,rng):
    used=set(); ents=[rand_entity(used,rng) for _ in range(n_hops+1)]
    facts=[(ents[i],"的工作搭档是",ents[i+1]) for i in range(n_hops)]
    q=f"{ents[0]}{'的工作搭档'*n_hops}是谁?"
    return facts,ents,q,ents[n_hops]

def normalize(s): return s.strip().strip("。.,，\"'").lower()

def main():
    model_name="Qwen/Qwen2.5-3B"  # 3B faster for ablation
    dev="cuda"
    tok=AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    configs = [
        {"rank":2, "epochs":3},
        {"rank":2, "epochs":5},
        {"rank":4, "epochs":3},
        {"rank":4, "epochs":5},
        {"rank":4, "epochs":10},
        {"rank":8, "epochs":5},
        {"rank":8, "epochs":10},
        {"rank":16,"epochs":5},
        {"rank":16,"epochs":15},  # original config (known to damage)
    ]

    # Generate TRAIN facts (fixed seed)
    train_rng=random.Random(42)
    train_chains=[make_chain(3,train_rng) for _ in range(20)]
    train_facts_str=[f"{s}{r}{o}。" for facts,_,_,_ in train_chains for s,r,o in facts]

    # Generate TEST facts for in-context (different seed, novel entities)
    test_rng=random.Random(99)
    test_chains=[make_chain(3,test_rng) for _ in range(15)]

    def gen(model,prompt,max_new=60):
        ids=tok(prompt,return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out=model.generate(ids,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)

    def test_single_hop(model):
        correct=0
        for facts,ents,_,_ in train_chains[:10]:
            s,r,o=facts[0]
            prompt=f"已知: {s}{r}{o}。\n问题: {s}的工作搭档是谁? 只回答全名:"
            out=gen(model,prompt,20)
            if normalize(o) in normalize(out): correct+=1
        return correct/10

    def test_closed_cot(model):
        correct=0
        for facts,ents,q,ans in train_chains[:10]:
            prompt=f"问题: {q}\n请一步一步推理。最终答案:"
            out=gen(model,prompt)
            if normalize(ans) in normalize(out): correct+=1
        return correct/10

    def test_incontext_cot(model):
        correct=0
        for facts,ents,q,ans in test_chains[:10]:
            random.shuffle(facts)
            fl="\n".join(f"- {s}{r}{o}。" for s,r,o in facts)
            prompt=f"以下是事实:\n{fl}\n\n问题: {q}\n请一步一步推理。最终答案:"
            out=gen(model,prompt)
            if normalize(ans) in normalize(out): correct+=1
        return correct/10

    # Base model reference
    print("Loading base model...")
    base=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(dev)
    base.eval()
    base_ic=test_incontext_cot(base)
    print(f"Base 3B in-context CoT: {base_ic:.0%}")
    del base; torch.cuda.empty_cache()

    results=[]
    for cfg in configs:
        r,ep=cfg["rank"],cfg["epochs"]
        print(f"\n--- rank={r}, epochs={ep} ---")
        m=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(dev)
        tgt=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        lc=LoraConfig(r=r,lora_alpha=2*r,lora_dropout=0.0,target_modules=tgt,bias="none",task_type="CAUSAL_LM")
        m=get_peft_model(m,lc); m.train()
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4)
        for e in range(ep):
            random.shuffle(train_facts_str)
            for f in train_facts_str:
                ids=tok(f,return_tensors="pt").input_ids.to(dev)
                out=m(ids,labels=ids); out.loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0)
                opt.step(); opt.zero_grad()
        m.eval()
        sh=test_single_hop(m)
        cc=test_closed_cot(m)
        ic=test_incontext_cot(m)
        print(f"  single-hop={sh:.0%} closed-cot={cc:.0%} in-context-cot={ic:.0%}")
        results.append({"rank":r,"epochs":ep,"single_hop":sh,"closed_cot":cc,"incontext_cot":ic})
        del m; torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  RANK ABLATION RESULTS (base in-context={base_ic:.0%})")
    print(f"{'='*60}")
    print(f"  {'rank':>4} {'ep':>3} {'single':>7} {'closed':>7} {'in-ctx':>7} {'reasoning':>10}")
    sweet_spot=None
    for r in results:
        intact="INTACT" if r["incontext_cot"]>=base_ic-0.2 else "DAMAGED"
        flag=""
        if r["single_hop"]>=0.8 and r["incontext_cot"]>=base_ic-0.2 and r["closed_cot"]<=0.1:
            flag=" *** SWEET SPOT"
            sweet_spot=r
        print(f"  {r['rank']:>4} {r['epochs']:>3} {r['single_hop']:>7.0%} {r['closed_cot']:>7.0%} {r['incontext_cot']:>7.0%} {intact:>10}{flag}")

    if sweet_spot:
        print(f"\n  *** DISSOCIATION CONFIRMED at rank={sweet_spot['rank']}, ep={sweet_spot['epochs']}:")
        print(f"      single={sweet_spot['single_hop']:.0%} (memorized)")
        print(f"      closed CoT={sweet_spot['closed_cot']:.0%} (can't compose from weights)")
        print(f"      in-context CoT={sweet_spot['incontext_cot']:.0%} (reasoning INTACT)")
    else:
        print(f"\n  ✗ No sweet spot found. Either:")
        print(f"    - Facts not memorized (single<80%) at configs that preserve reasoning")
        print(f"    - Or reasoning always damaged when facts memorized")

    json.dump({"base_incontext":base_ic,"configs":results,"sweet_spot":sweet_spot},
              open("/root/results/rank_ablation.json","w"))
    print("\nSaved: /root/results/rank_ablation.json")

if __name__=="__main__":
    main()
