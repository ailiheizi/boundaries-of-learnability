"""idea3 Pilot A: 弹幕结构信号 vs 唤醒信号的分离检验
核心问题: 在"唤醒匹配"条件下, 回指弹幕是否仍然集中在结构节点?

方法:
1. 按时间窗口(30s)切bin
2. 计算每个bin的: 唤醒密度(弹幕量) + 结构回指密度(含结构关键词的比例)
3. 找"结构高点"(回指密度top-k)和"唤醒匹配但结构低"的控制点
4. 对比: 结构高点的回指密度 vs 唤醒匹配控制点的回指密度
5. Go/No-Go: 结构高点 ≥ 2× 控制点 → 结构信号独立于唤醒

反作弊:
- 唤醒匹配对照: 控制弹幕量(±20%), 只比回指密度
- 时间打乱对照: 打乱弹幕时间, 结构信号必须消失
"""
import json, sys, os, random
import statistics
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout,'reconfigure') else None

# 结构回指关键词 (观众主动标注叙事结构的弹幕)
KW_STRUCT = ['伏笔','原来','前面','呼应','回收','二刷','细节','暗示','铺垫','果然',
             '难怪','所以是','怪不得','这里','那个','对应','照应','之前']
# 唤醒关键词 (无差别情绪反应)
KW_AROUSAL = ['哈哈','卧槽','牛','帅','厉害','666','nb','好帅','太强','我靠']

def load_data(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def bin_danmaku(dms, bin_size=30):
    """按 bin_size 秒切窗口, 返回 {bin_idx: [dm_list]}"""
    from collections import defaultdict
    bins = defaultdict(list)
    for dm in dms:
        b = int(dm['t'] // bin_size)
        bins[b].append(dm)
    return dict(bins)

def compute_signals(bins):
    """每个bin算: total(弹幕量=唤醒代理), struct_ratio(结构回指比例)"""
    results = {}
    for b, dms in bins.items():
        total = len(dms)
        struct_count = sum(1 for dm in dms if any(k in dm['text'] for k in KW_STRUCT))
        arousal_count = sum(1 for dm in dms if any(k in dm['text'] for k in KW_AROUSAL))
        struct_ratio = struct_count / total if total > 0 else 0
        results[b] = {
            'total': total,
            'struct_count': struct_count,
            'struct_ratio': struct_ratio,
            'arousal_count': arousal_count,
        }
    return results

def find_matched_pairs(signals, n_pairs=5, tolerance=0.2):
    """找 n_pairs 对 (结构高点, 唤醒匹配控制点):
    结构高点 = struct_ratio 排名top的bin
    控制点 = total(弹幕量)在±tolerance范围内, 但 struct_ratio 低的bin
    """
    items = [(b, s) for b, s in signals.items() if s['total'] >= 5]  # 最少5条才有统计意义
    items.sort(key=lambda x: x[1]['struct_ratio'], reverse=True)

    struct_tops = items[:n_pairs*2]  # 候选结构高点
    struct_lows = items[len(items)//2:]  # 结构低的一半

    pairs = []
    used = set()
    for b_hi, s_hi in struct_tops:
        if len(pairs) >= n_pairs: break
        target_total = s_hi['total']
        lo_range = target_total * (1 - tolerance)
        hi_range = target_total * (1 + tolerance)
        # 找弹幕量匹配的控制点
        for b_lo, s_lo in struct_lows:
            if b_lo in used or b_lo == b_hi: continue
            if lo_range <= s_lo['total'] <= hi_range:
                pairs.append({'struct_bin': b_hi, 'control_bin': b_lo,
                              'struct': s_hi, 'control': s_lo})
                used.add(b_lo); break
    return pairs

def run_pilot(data_path, bin_size=30):
    d = load_data(data_path)
    dms = d['danmaku']
    dur = d['duration']
    print(f"视频: {d['title']}")
    print(f"弹幕: {len(dms)}条, 时长: {dur}s ({dur//60}分)")
    print(f"bin大小: {bin_size}s\n")

    bins = bin_danmaku(dms, bin_size)
    signals = compute_signals(bins)
    pairs = find_matched_pairs(signals, n_pairs=8, tolerance=0.25)

    if not pairs:
        print("❌ 找不到唤醒匹配的对照对 → 无法做分离检验")
        return

    print(f"找到 {len(pairs)} 对唤醒匹配对照 (弹幕量±25%)")
    print(f"\n{'='*70}")
    print(f"  {'时间':>8} {'弹幕量':>6} {'回指比':>7} | {'时间':>8} {'弹幕量':>6} {'回指比':>7} | {'倍数':>5}")
    print(f"  {'[结构高]':>8} {'':>6} {'':>7} | {'[控制点]':>8} {'':>6} {'':>7} |")
    print(f"{'='*70}")

    ratios = []
    for p in pairs:
        sb, cb = p['struct_bin'], p['control_bin']
        ss, cs = p['struct'], p['control']
        t_s = f"{sb*bin_size//60}:{sb*bin_size%60:02d}"
        t_c = f"{cb*bin_size//60}:{cb*bin_size%60:02d}"
        ratio = ss['struct_ratio'] / cs['struct_ratio'] if cs['struct_ratio'] > 0 else float('inf')
        ratios.append(ratio)
        print(f"  {t_s:>8} {ss['total']:>6} {ss['struct_ratio']:>6.1%} | "
              f"{t_c:>8} {cs['total']:>6} {cs['struct_ratio']:>6.1%} | {ratio:>5.1f}x")

    finite_ratios = [r for r in ratios if r != float('inf')]
    if not finite_ratios:
        print(f"\n  所有对照的控制点回指=0 → 反例视频(无结构信号)")
        print(f"  判定: 此视频无叙事结构信号(符合预期的反例)")
        mean_ratio = 0; median_ratio = 0
    else:
        mean_ratio = statistics.mean(finite_ratios)
        median_ratio = statistics.median(finite_ratios)
    print(f"\n{'='*70}")
    print(f"  回指密度倍数: mean={mean_ratio:.2f}x  median={median_ratio:.2f}x")
    print(f"  Go/No-Go 阈值: ≥2x")
    print(f"\n  判定: ", end="")
    if median_ratio >= 2.0:
        print(f"✅ GO — 结构信号独立于唤醒 (median {median_ratio:.1f}x ≥ 2x)")
    elif median_ratio >= 1.5:
        print(f"⚠️ 边缘 — 有信号但不强 (median {median_ratio:.1f}x)")
    else:
        print(f"❌ NO-GO — 结构信号≈唤醒, 无法分离 (median {median_ratio:.1f}x)")

    # === 反作弊: 时间打乱对照 ===
    print(f"\n{'='*70}")
    print(f"  反作弊: 时间打乱对照")
    shuffled_dms = [{'t': random.random()*dur, 'text': dm['text']} for dm in dms]
    bins_shuf = bin_danmaku(shuffled_dms, bin_size)
    sig_shuf = compute_signals(bins_shuf)
    # 打乱后所有bin的struct_ratio应该趋同
    all_ratios_shuf = [s['struct_ratio'] for s in sig_shuf.values() if s['total'] >= 5]
    if all_ratios_shuf:
        std_shuf = statistics.stdev(all_ratios_shuf)
        all_ratios_orig = [s['struct_ratio'] for s in signals.values() if s['total'] >= 5]
        std_orig = statistics.stdev(all_ratios_orig)
        print(f"  原始 struct_ratio 标准差: {std_orig:.4f}")
        print(f"  打乱 struct_ratio 标准差: {std_shuf:.4f}")
        if std_orig > std_shuf * 1.5:
            print(f"  ✅ 打乱后信号坍塌 ({std_orig:.4f} → {std_shuf:.4f}) — 时间对齐是真的")
        else:
            print(f"  ⚠️ 打乱后信号未显著坍塌 — 可能是全局词频非时间相关")

    # 保存结果
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'pilot_a_result.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'video': d['title'], 'bvid': d['bvid'], 'n_danmaku': len(dms),
               'n_pairs': len(pairs), 'mean_ratio': mean_ratio, 'median_ratio': median_ratio,
               'go': median_ratio >= 2.0, 'pairs': pairs},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n  保存 {out}")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'data', 'BV1LSoyYqEuU.json')
    run_pilot(path)
