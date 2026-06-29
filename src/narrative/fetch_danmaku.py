"""B站弹幕爬虫 — 输入BV号, 输出带时间戳的弹幕 (stdlib only)
弹幕XML每条含: 出现时间(秒,视频内位置) + 文本。这是结构信号分析的原料。
用法: python fetch_danmaku.py BV1xx411c7mD
"""
import sys, os, json, re, urllib.request, zlib, gzip

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept-Encoding": "gzip, deflate"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding", "")
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        try: return zlib.decompress(raw)
        except zlib.error: return zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw

def get_cid_title(bvid):
    j = json.loads(get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}").decode("utf-8"))
    if j["code"] != 0:
        raise RuntimeError(f"view api error: {j['message']}")
    d = j["data"]
    return d["cid"], d["title"], d.get("duration", 0)

def fetch_danmaku(cid):
    """旧接口 XML, 采样几千条(试点足够)。每条: <d p="时间,模式,字号,颜色,时间戳,池,uid,dmid">文本</d>"""
    xml = get(f"https://comment.bilibili.com/{cid}.xml").decode("utf-8")
    dms = []
    for m in re.finditer(r'<d p="([^"]+)">([^<]*)</d>', xml):
        attrs = m.group(1).split(",")
        t = float(attrs[0])           # 视频内出现时间(秒)
        text = m.group(2)
        dms.append({"t": t, "text": text})
    dms.sort(key=lambda x: x["t"])
    return dms

def main():
    if len(sys.argv) < 2:
        print("用法: python fetch_danmaku.py <BV号>"); sys.exit(1)
    bvid = sys.argv[1]
    cid, title, dur = get_cid_title(bvid)
    print(f"BV={bvid} cid={cid} 时长={dur}s")
    print(f"标题: {title}")
    dms = fetch_danmaku(cid)
    print(f"弹幕数: {len(dms)} (采样)")
    outdir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{bvid}.json")
    json.dump({"bvid": bvid, "cid": cid, "title": title, "duration": dur, "danmaku": dms},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"保存 {out}")
    # 简单分布预览: 每分钟弹幕量
    if dms and dur:
        from collections import Counter
        bins = Counter(int(d["t"]//60) for d in dms)
        print("每分钟弹幕量:", " ".join(f"{m}:{bins.get(m,0)}" for m in range(int(dur//60)+1)))

if __name__ == "__main__":
    main()
