from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from ..config import AIConfig


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    error: Optional[str] = None
    tool_calls: int = 0


class LLMClient:
    """任意 OpenAI 兼容网关；支持 function calling（工具调用）。"""

    def __init__(self, config: AIConfig, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run

    @property
    def ready(self) -> bool:
        return bool(self.config.api_key) and "REPLACE" not in self.config.api_key

    def chat(
        self,
        messages: List[Dict[str, Any]],
        retries: int = 2,
        tools: Optional[List[Dict[str, Any]]] = None,
        executor: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> LLMResult:
        start = time.time()
        if self.dry_run or not self.ready:
            return LLMResult(self._dry_reply(messages), "dry-run", int((time.time() - start) * 1000))
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        working = list(messages)
        tool_rounds = 0
        for attempt in range(retries + 1):
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": working,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout_s)
                if response.status_code >= 400:
                    if tools and response.status_code in (400, 422):
                        tools = None  # 模型不支持工具，退回普通对话
                        working = list(messages)
                        continue
                    if response.status_code in (429, 500, 502, 503):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    return LLMResult("", self.config.model, int((time.time() - start) * 1000),
                                     error=f"HTTP {response.status_code}: {response.text[:200]}")
                data = response.json()
                message = (data.get("choices") or [{}])[0].get("message", {}) or {}
                calls = message.get("tool_calls") or []
                if calls and executor and tool_rounds < 4:
                    tool_rounds += 1
                    working.append({
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": calls,
                    })
                    for call in calls:
                        function = call.get("function") or {}
                        name = function.get("name") or ""
                        try:
                            import json

                            arguments = json.loads(function.get("arguments") or "{}")
                        except Exception:
                            arguments = {}
                        result = executor(name, arguments)
                        working.append({
                            "role": "tool",
                            "tool_call_id": call.get("id") or name,
                            "content": result,
                        })
                    continue
                text = message.get("content") or ""
                return LLMResult(text.strip(), self.config.model,
                                 int((time.time() - start) * 1000), tool_calls=tool_rounds)
            except Exception as exc:
                if attempt >= retries:
                    return LLMResult("", self.config.model, int((time.time() - start) * 1000), error=str(exc))
                time.sleep(1.0 * (attempt + 1))
        return LLMResult("", self.config.model, int((time.time() - start) * 1000), error="重试耗尽")

    def _dry_reply(self, messages: List[Dict[str, Any]]) -> str:
        context = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                context = str(message.get("content", ""))
                break
        snippet = context.replace("\n", " ")[:60]
        return f"【dry-run】收到：{snippet} —— 这是模拟回复，配置 api_key 后替换为真实模型输出。"

    def probe(self) -> LLMResult:
        return self.chat([{"role": "user", "content": "你好，用一句话回复"}], retries=0)


class ModelRouter:
    """模型切换：同一套 key 下切换 model / base_url。"""

    def __init__(self, config: AIConfig, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self.options = config.options()
        self.current = config.model
        # 启动时就要把"当前模型对应的已保存 Key / base_url"应用上，
        # 否则会出现"填了 Key 却仍显示未就绪"的问题。
        for option in self.options:
            if option.id == self.current or option.model == self.current:
                config.apply_option(option)
                break
        self.client = LLMClient(config, dry_run=dry_run)

    def switch(self, model_id: str) -> bool:
        for option in self.options:
            if option.id == model_id or option.model == model_id:
                self.config.apply_option(option)
                self.current = option.model
                self.client = LLMClient(self.config, dry_run=self.dry_run)
                return True
        return False

    def set_dry_run(self, value: bool) -> None:
        self.dry_run = value
        self.client.dry_run = value
