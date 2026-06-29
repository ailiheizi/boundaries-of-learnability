"""本地 llama-server (OpenAI兼容) 客户端 — stdlib only, 替代 DeepSeekClient

llama.cpp 的 llama-server 提供 /v1/chat/completions (OpenAI兼容)。
用法和 DeepSeekClient.chat 一致, 方便 pilot 直接换 endpoint。
"""
from __future__ import annotations
import os, json, time, urllib.request, urllib.error

class LocalServerClient:
    def __init__(self, base_url=None, model="local"):
        self.url = (base_url or os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")) + "/v1/chat/completions"
        self.model = model

    def chat(self, messages, temperature=0.0, max_tokens=512):
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens, "stream": False}
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"server error {e.code}: {e.read().decode('utf-8','replace')}")
        return {"content": result["choices"][0]["message"]["content"],
                "latency_ms": (time.perf_counter()-start)*1000,
                "usage": result.get("usage", {})}

if __name__ == "__main__":
    c = LocalServerClient()
    r = c.chat([{"role":"user","content":"2+2=? 只回答数字"}], max_tokens=10)
    print("OK:", repr(r["content"]), f"{r['latency_ms']:.0f}ms")
