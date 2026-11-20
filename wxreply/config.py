from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelOption:
    id: str = ""
    label: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"id": self.id, "label": self.label or self.model, "model": self.model}


@dataclass
class AIConfig:
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    models: List[Any] = field(default_factory=list)   # [{id,label,model,base_url,api_key}]
    model_keys: Dict[str, str] = field(default_factory=dict)  # 控制台里按模型填的 key
    vision_model: str = ""                             # 图片理解用的视觉模型
    vision_base_url: str = ""
    embedding_model: str = ""                          # 向量检索用的 embedding 模型
    temperature: float = 0.7
    max_tokens: int = 300
    timeout_s: float = 30.0

    def options(self) -> List[ModelOption]:
        result: List[ModelOption] = []
        for item in self.models or []:
            if isinstance(item, str):
                result.append(ModelOption(id=item, label=item, model=item))
            elif isinstance(item, dict):
                model = str(item.get("model") or item.get("id") or "")
                option_id = str(item.get("id") or model)
                result.append(ModelOption(
                    id=option_id,
                    label=str(item.get("label") or model),
                    model=model,
                    base_url=str(item.get("base_url") or ""),
                    api_key=str(item.get("api_key") or self.model_keys.get(option_id) or self.api_key or ""),
                ))
        if not result:
            result = [ModelOption(id=self.model, label=self.model, model=self.model)]
        return result

    def apply_option(self, option: ModelOption) -> None:
        if option.model:
            self.model = option.model
        if option.base_url:
            self.base_url = option.base_url
        self.api_key = option.api_key or self.api_key


@dataclass
class RuntimeConfig:
    poll_interval_s: float = 1.0
    settle_s: float = 0.35
    aggregate_max_wait_s: float = 1.2
    cooldown_s: float = 3.0
    max_actions_per_min: int = 20
    max_unread_per_cycle: int = 3
    quiet_hours: List[str] = field(default_factory=list)
    dry_run: bool = True
    allow_group_chat: bool = False
    group_whitelist: List[str] = field(default_factory=list)
    group_reply_mode: str = "mention"      # mention | keyword | all
    group_keywords: List[str] = field(default_factory=list)
    send_retries: int = 2
    allow_real_actions: bool = False   # macOS 默认只读：不点击/不滚动/不发送
    offscreen: bool = False     # Windows: 启动后把微信移入离屏空间（-3000,-3000）


@dataclass
class SafetyConfig:
    whitelist: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    escalate_keywords: List[str] = field(default_factory=list)
    max_reply_chars: int = 120
    require_header_match: bool = True
    verify_send: bool = True
    forbidden_self_disclosure: List[str] = field(default_factory=list)
    # 发送风控（拟人节奏/频次上限，降低自动化特征）
    risk_enabled: bool = True
    risk_min_interval_s: float = 20.0
    risk_max_per_hour: int = 10
    risk_max_per_day: int = 50
    typing_chars_per_second: float = 4.0
    risk_jitter: float = 0.3
    risk_max_wait_s: float = 15.0


@dataclass
class AuditConfig:
    db_path: str = "data/audit.db"
    evidence_dir: str = "data/evidence"
    save_screenshots: bool = True


@dataclass
class MemoryConfig:
    db_path: str = "data/memory.db"
    recent_limit: int = 8
    summarize_after: int = 12


@dataclass
class RagConfig:
    mode: str = "hybrid"          # bm25 | hybrid
    top_k: int = 3
    embedding_cache: str = "data/kb_embeddings.json"


@dataclass
class ToolsConfig:
    price_file: str = "data/price_list.json"
    leads_file: str = "data/leads.jsonl"
    enabled: bool = True


@dataclass
class WeChatConfig:
    title_keywords: List[str] = field(default_factory=lambda: ["微信", "Weixin", "WeChat"])
    class_keywords: List[str] = field(default_factory=lambda: ["WeChatMainWndForPC", "Qt5"])
    process_names: List[str] = field(default_factory=lambda: ["WeChat.exe", "Weixin.exe"])
    mac_activate_on_click: bool = True


@dataclass
class LayoutConfig:
    nav_ratio: float = 0.055
    list_ratio: float = 0.34
    header_ratio: float = 0.09
    input_ratio: float = 0.22
    overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    wechat: WeChatConfig = field(default_factory=WeChatConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    persona: str = "personas/sales_zhang.yaml"
    kb_dir: str = "kb"
    base_dir: str = "."
    config_path: str = ""
    settings_path: str = "data/settings.json"
    channels: List[Dict[str, Any]] = field(default_factory=list)  # 全 IM 通道配置
    friend_personas: Dict[str, str] = field(default_factory=dict)  # 好友专属人设

    def resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key) or {}
    return value if isinstance(value, dict) else {}


def _apply(obj: Any, data: Dict[str, Any]) -> Any:
    for key, value in data.items():
        if hasattr(obj, key) and value is not None:
            current = getattr(obj, key)
            if isinstance(current, list):
                setattr(obj, key, list(value))
            elif isinstance(current, dict):
                setattr(obj, key, dict(value))
            elif isinstance(current, bool):
                setattr(obj, key, bool(value))
            elif isinstance(current, int):
                setattr(obj, key, int(value))
            elif isinstance(current, float):
                setattr(obj, key, float(value))
            else:
                setattr(obj, key, value)
    return obj


def load_config(path: str = "config.yaml") -> Config:
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path) or "."
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    cfg = Config(base_dir=base_dir)
    _apply(cfg.ai, _section(data, "ai"))
    _apply(cfg.runtime, _section(data, "runtime"))
    _apply(cfg.safety, _section(data, "safety"))
    _apply(cfg.audit, _section(data, "audit"))
    _apply(cfg.memory, _section(data, "memory"))
    _apply(cfg.rag, _section(data, "rag"))
    _apply(cfg.tools, _section(data, "tools"))
    _apply(cfg.wechat, _section(data, "wechat"))
    _apply(cfg.layout, _section(data, "layout"))
    if isinstance(data.get("persona"), str):
        cfg.persona = data["persona"]
    if isinstance(data.get("kb_dir"), str):
        cfg.kb_dir = data["kb_dir"]
    raw_personas = data.get("friend_personas")
    if isinstance(raw_personas, dict):
        cfg.friend_personas = {str(k): str(v) for k, v in raw_personas.items() if str(v).strip()}
    raw_channels = data.get("channels")
    if isinstance(raw_channels, list):
        cfg.channels = [item for item in raw_channels if isinstance(item, dict)]
    cfg.config_path = path
    _apply_settings_overlay(cfg)
    return cfg


def _apply_settings_overlay(cfg: Config) -> None:
    """控制台写回的本地设置（key / 白名单 / 布局），不改 config.yaml 注释。"""
    path = cfg.resolve(cfg.settings_path)
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return
    keys = data.get("api_keys") or {}
    if isinstance(keys, dict):
        cfg.ai.model_keys = {str(k): str(v) for k, v in keys.items() if v}
    whitelist = data.get("whitelist")
    if isinstance(whitelist, list):
        cfg.safety.whitelist = [str(item) for item in whitelist if str(item).strip()]
    layout = data.get("layout") or {}
    if isinstance(layout, dict):
        for key, value in layout.items():
            if hasattr(cfg.layout, key) and isinstance(value, (int, float)):
                setattr(cfg.layout, key, float(value))
    if isinstance(data.get("vision_model"), str):
        cfg.ai.vision_model = data["vision_model"]
    if isinstance(data.get("embedding_model"), str):
        cfg.ai.embedding_model = data["embedding_model"]
