"""idea4 路线B: 权重内化的组合推理 + CoT 救援
================================================================================
命题: 把事实链原子句 LoRA 写进权重(不在context里), 闭卷测组合推理能否成立 + CoT能否救。
补全 idea4 的 2×2: {in-context, 权重内化} × {直接组合, CoT组合}
  in-context (E1/E2 已做): CoT救援随规模 40→65→100%
  权重内化 (本实验):       ?

设计 (复用 reverse_gen_v4 方法论):
  虚构事实链 P0→P1→P2 (工作搭档链), 只把原子单跳句写进权重:
    "P0的工作搭档是P1。" "P1的工作搭档是P2。"  (绝不写组合句)
  闭卷评测 (事实全在权重, prompt不给):
    单跳:    P0的工作搭档是谁?           (确认写进去了; 必须高才有资格测组合)
    直接组合: P0的工作搭档的工作搭档是谁?  (2-hop, 不给CoT)
    CoT组合:  同问题, 允许逐步推理
  防作弊: 虚构实体 / 闭卷 / 单跳对照门控 / 多seed / forced-choice可选

依赖: torch, transformers, peft
用法:
  无卡smoke验逻辑:  python idea4_weight.py --mode smoke --model EleutherAI/pythia-410m
  GPU full:        HF_ENDPOINT=https://hf-mirror.com python idea4_weight.py --model Qwen/Qwen2.5-1.5B --mode full
"""
import argparse, os, sys, json, random, re
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

FIRST=["Zelph","Quorin","Vandel","Tovak","Yelena","Karsh","Oslo","Veyra","Drennan","Sythe",
       "Marlow","Pendric","Casimir","Wren","Galen","Ophira","Lucan","Brenna","Caldus","Mirae"]
SUR=["Voss","Karr","Thorne","Belmont","Ashby","Cregan","Dunmore","Fenwick","Halloran","Ives",
     "Quill","Sterling","Underwood","Vance","Whitlock","Yarrow","Zane","Brackett","Coyle","Dane"]

def rand_entity(used,rng):
    while True:
        n=f"{rng.choice(FIRST)} {rng.choice(SUR)}"
        if n not in used: used.add(n); return n

def make_chains(n_chains, hops, rng):
    """造 n_chains 条独立链, 每条长 hops。返回 chains[list of entity-list]。"""
    used=set(); chains=[]
    for _ in range(n_chains):
        chains.append([rand_entity(used,rng) for _ in range(hops+1)])
    return chains

def atomic_facts(chains):
    """只生成原子单跳句 (绝不含组合)。"""
    facts=[]
    for ch in chains:
        for i in range(len(ch)-1):
            facts.append(f"{ch[i]}的工作搭档是{ch[i+1]}。")
    return facts

def q_single(e): return f"{e}的工作搭档是"
def q_comp(e,k): return f"{e}{'的工作搭档'*k}是"
def q_comp_cot(e,k):
    return (f"问题: {e}{'的工作搭档'*k}是谁?\n"
            f"请一步步追踪工作搭档链, 逐跳推理。最后另起一行用'最终答案: <名> <姓>'给出答案。")

def norm(s): return s.strip().strip("。.,，\"'：: ").lower()

def extract_cot(text):
    ms=list(re.finditer(r"(?:最终答案|答案|答)[:：]?\s*([A-Za-zÀ-ÿ]+\s+[A-Za-zÀ-ÿ]+)",text))
    if ms: return ms[-1].group(1)
    ms=list(re.finditer(r"是\s*([A-Za-zÀ-ÿ]+\s+[A-Za-zÀ-ÿ]+)",text))
    if ms: return ms[-1].group(1)
    lines=[l for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--hops",default="2,3,4")
    ap.add_argument("--n-chains",type=int,default=20,help="每个hop数的链条数(=样本数)")
    ap.add_argument("--seeds",default="42,43")
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--lora-r",type=int,default=16)
    ap.add_argument("--full-ft",action="store_true")
    ap.add_argument("--mode",default="full",choices=["smoke","full"])
    ap.add_argument("--out",default="../results/idea4_weight.json")
    args=ap.parse_args()
    if args.mode=="smoke":
        args.n_chains=3; args.epochs=3; args.seeds="42"; args.hops="2"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    hops_list=[int(h) for h in args.hops.split(",")]
    seeds=[int(s) for s in args.seeds.split(",")]
    print(f"[idea4_weight] model={args.model} dev={dev} hops={hops_list} n_chains={args.n_chains} "
          f"seeds={seeds} epochs={args.epochs} full_ft={args.full_ft} mode={args.mode}", flush=True)

    def fresh(seed):
        torch.manual_seed(seed)
        # For large full-FT models, use bf16 + gradient checkpointing to fit in VRAM
        big = args.full_ft and any(s in args.model for s in ["7B","7b","14B","14b"])
        dtype = torch.bfloat16 if (big and dev=="cuda") else torch.float32
        m=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype).to(dev)
        if big:
            m.gradient_checkpointing_enable()
            m.config.use_cache=False
        if not args.full_ft:
            from peft import LoraConfig, get_peft_model
            ml=args.model.lower()
            if "qwen" in ml or "llama" in ml:
                tgt=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
            elif "pythia" in ml:
                tgt=["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]
            else: tgt=["c_attn","c_proj","c_fc"]
            cfg=LoraConfig(r=args.lora_r,lora_alpha=2*args.lora_r,lora_dropout=0.0,
                           target_modules=tgt,bias="none",task_type="CAUSAL_LM")
            m=get_peft_model(m,cfg)
        return m

    def train(model, facts):
        model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
        for ep in range(args.epochs):
            random.shuffle(facts)
            for f in facts:
                ids=tok(f,return_tensors="pt").input_ids.to(dev)
                out=model(ids,labels=ids); out.loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                opt.step(); opt.zero_grad()

    @torch.no_grad()
    def gen(model, prompt, max_new=20):
        model.eval()
        ids=tok(prompt,return_tensors="pt").input_ids.to(dev)
        out=model.generate(ids,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)

    results={}
    for h in hops_list:
        agg={"single":[],"comp":[],"cot":[]}
        for sd in seeds:
            rng=random.Random(sd)
            chains=make_chains(args.n_chains,h,rng)
            facts=atomic_facts(chains)
            m=fresh(sd); train(m,facts)
            sc=cc=ctc=0
            for ch in chains:
                # 单跳: 第一跳
                if norm(ch[1]) in norm(gen(m,q_single(ch[0]),12)): sc+=1
                # 直接组合
                if norm(ch[h]) in norm(gen(m,q_comp(ch[0],h)+"谁?只回答全名:",20)): cc+=1
                # CoT
                if norm(ch[h]) in norm(extract_cot(gen(m,q_comp_cot(ch[0],h),400))): ctc+=1
            n=len(chains)
            agg["single"].append(sc/n); agg["comp"].append(cc/n); agg["cot"].append(ctc/n)
            print(f"  hop={h} seed={sd}: single={sc/n:.0%} comp={cc/n:.0%} cot={ctc/n:.0%}",flush=True)
        import statistics
        results[h]={k:statistics.mean(v) for k,v in agg.items()}
    print(f"\n{'='*60}\n  权重内化组合推理: {args.model}\n{'='*60}")
    print(f"  {'hop':>4} {'单跳':>8} {'直接组合':>10} {'CoT组合':>10}")
    for h in hops_list:
        r=results[h]; print(f"  {h:>4} {r['single']:>7.0%} {r['comp']:>9.0%} {r['cot']:>9.0%}")
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    json.dump({"model":args.model,"full_ft":args.full_ft,"n_chains":args.n_chains,
               "epochs":args.epochs,"results":results},
              open(args.out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n  保存 {args.out}")

if __name__=="__main__": main()
