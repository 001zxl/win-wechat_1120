"""控制台 API 冒烟测试（mock 模式，不碰真实微信）。

用法：
    1) 先启动：.venv\\Scripts\\python.exe server.py --mock --no-browser --port 8791
    2) 再运行：.venv\\Scripts\\python.exe scripts\\smoke_console.py 8791

注意：会在本机 data/ 里留下【模拟】冒烟A/B 的会话与接管记录（审计、记忆、handoff）。
跑完想清干净：删掉 data/handoff.json 里的【模拟】* 项，或直接用一份临时工作目录。
"""
import json
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8791"
BASE = f"http://127.0.0.1:{PORT}"


def call(action, **payload):
    body = json.dumps({"action": action, **payload}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/action", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def state():
    with urllib.request.urlopen(BASE + "/api/state", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


print("1) 无依据报价（ZX-999）")
result = call("simulate", peer="冒烟A", text="ZX-999 多少钱", auto=True)
print("  ", result.get("ok"), result.get("result", {}).get("escalated"),
      result.get("result", {}).get("reply"))

print("2) 有报价表（FT-120 800 件）")
result = call("simulate", peer="冒烟B", text="FT-120 800件多少钱", auto=False)
print("  ", result.get("ok"), (result.get("result") or {}).get("reply"))
peer = (result.get("result") or {}).get("peer")
call("select", peer=peer)
snapshot = state()
pending = snapshot.get("pending") or {}
print("   pending draft_id:", (pending.get("draft_id") or "")[:8],
      "version:", pending.get("draft_version"),
      "kb:", (pending.get("knowledge") or {}).get("provider"),
      "sources:", len((pending.get("knowledge") or {}).get("sources") or []))

print("3) 不带 draft_id 批准（应被拒）")
result = call("approve", peer=peer, text="FT-120 800 件 2.9-3.3 美金")
print("  ", result.get("ok"), (result.get("result") or {}).get("status"),
      (result.get("result") or {}).get("detail"))

print("4) 带当前 draft_id 批准（模拟会话）")
result = call("approve", peer=peer, text="FT-120 800 件 2.9-3.3 美金",
              draft_id=pending.get("draft_id"), event_seq=pending.get("event_seq"))
print("  ", result.get("ok"), (result.get("result") or {}).get("status"))

print("5) 撤销/删除 key 接口（不真的删，只验证接口形状）")
result = call("api_keys", keys={}, delete=[])
print("  ", result.get("ok"), result.get("llm_ready"))

print("6) 健康检查")
with urllib.request.urlopen(BASE + "/api/health", timeout=10) as response:
    print("  ", response.status, response.read().decode("utf-8"))
