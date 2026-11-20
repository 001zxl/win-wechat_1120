#!/usr/bin/env python3
"""CLI：doctor / capture / mock / run"""
from __future__ import annotations

import argparse
import os
import sys

from wxreply.agent import Agent
from wxreply.config import Config, load_config
from wxreply.perception.layout import LayoutCalculator
from wxreply.perception.ocr import OcrEngine
from wxreply.platform.mock import MockPlatform
from wxreply.platform.win32 import IS_WINDOWS, Win32Platform
from wxreply.policy import Whitelist

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_real_platform(cfg: Config):
    if sys.platform == "darwin":
        from wxreply.platform.mac import MacPlatform

        platform = MacPlatform(activate_on_click=cfg.wechat.mac_activate_on_click)
        if platform.find_window() is None:
            raise SystemExit("未找到微信窗口：请先打开微信，并停在聊天界面")
        if not MacPlatform.accessibility_trusted():
            print("⚠️ 辅助功能未授权：能抓图但无法点击/按键。")
            print("   请到 系统设置 → 隐私与安全性 → 辅助功能，勾选运行本程序的应用（Python/终端/DSH）")
        return platform
    if not IS_WINDOWS:
        raise SystemExit("真机模式仅支持 Windows / macOS。")
    platform = Win32Platform(cfg.wechat.title_keywords, cfg.wechat.class_keywords, cfg.wechat.process_names)
    window = platform.find_window()
    if window is None:
        raise SystemExit("未找到微信窗口，请先登录微信 PC 版并保持窗口打开")
    print(f"已绑定窗口：hwnd={window.hwnd} title={window.title!r} class={window.class_name!r}")
    return platform


def annotate(image, regions, layout: LayoutCalculator):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    colors = {
        "nav": (255, 0, 0),
        "session_list": (255, 128, 0),
        "chat": (0, 128, 255),
        "header": (0, 200, 0),
        "messages": (128, 0, 255),
        "input": (0, 200, 200),
    }
    for name, box in regions.to_dict().items():
        draw.rectangle(box, outline=colors.get(name, (0, 0, 0)), width=3)
        draw.text((box[0] + 4, box[1] + 4), name, fill=colors.get(name, (0, 0, 0)))
    return image


def cmd_doctor(args) -> int:
    cfg = load_config(args.config)
    print("== 环境 ==")
    print(f"python={sys.version.split()[0]} platform={sys.platform}")
    print(f"config={os.path.abspath(args.config)} exists={os.path.exists(args.config)}")
    persona_path = cfg.resolve(cfg.persona)
    print(f"persona={persona_path} exists={os.path.exists(persona_path)}")
    print(f"kb_dir={cfg.resolve(cfg.kb_dir)}")
    whitelist = Whitelist(cfg.safety.whitelist, cfg.runtime.allow_group_chat)
    print(f"白名单={cfg.safety.whitelist} (群聊放行={cfg.runtime.allow_group_chat})")
    settings_path = cfg.resolve(cfg.settings_path)
    print(f"控制台设置={settings_path} exists={os.path.exists(settings_path)}")
    print("模型 Key 状态=" + ", ".join(
        f"{item.id}:{'有' if item.api_key else '缺'}" for item in cfg.ai.options()
    ))
    print(f"视觉模型={cfg.ai.vision_model or '未配置'}")
    print(f"Embedding 模型={cfg.ai.embedding_model or '未配置'}")

    ocr = OcrEngine()
    print(f"OCR={'可用' if ocr.available else '不可用：' + str(ocr.load_error)}")

    from wxreply.brain.llm import LLMClient, ModelRouter

    # 用路由解析后的凭据判断就绪：控制台里填的按模型 key 也算数
    router = ModelRouter(cfg.ai, dry_run=cfg.runtime.dry_run)
    llm = LLMClient(router.client.config)
    print(f"模型={cfg.ai.model} base={cfg.ai.base_url} ready={llm.ready} dry_run={cfg.runtime.dry_run}")

    layout = LayoutCalculator(
        cfg.layout.nav_ratio, cfg.layout.list_ratio,
        cfg.layout.header_ratio, cfg.layout.input_ratio, cfg.layout.overrides,
    )
    if args.mock or not IS_WINDOWS:
        platform = MockPlatform()
        image = platform.capture()
        source = "mock"
    else:
        platform = build_real_platform(cfg)
        image = platform.capture()
        source = "real"
    if image is None:
        print("[FAIL] 抓图失败")
        return 1
    regions = layout.regions(image.width, image.height)
    print(f"窗口快照({source}) {image.width}x{image.height}")
    for name, box in regions.to_dict().items():
        print(f"  {name:14s} {box}")
    print("\n== 首次启动检查 ==")
    window_found = platform.find_window() is not None if hasattr(platform, "find_window") else False
    settings_file = cfg.resolve(cfg.settings_path)
    layout_saved = False
    if os.path.exists(settings_file):
        try:
            import json

            with open(settings_file, "r", encoding="utf-8") as handle:
                layout_saved = bool((json.load(handle) or {}).get("layout"))
        except Exception:
            layout_saved = False
    key_ok = any(item.api_key for item in cfg.ai.options())
    items = [
        ("Windows 环境", IS_WINDOWS, "真实微信需在 Windows 上运行"),
        ("微信窗口", window_found, "登录微信 PC 并保持窗口打开"),
        ("RapidOCR", ocr.available, "运行 setup.bat 或 pip install rapidocr_onnxruntime"),
        ("布局校准", layout_saved, "控制台点「校准布局」拖滑块后保存"),
        ("白名单", bool(cfg.safety.whitelist), "控制台左栏添加真实联系人"),
        ("模型 Key", key_ok, "控制台「设置」里填 API Key"),
    ]
    missing = 0
    for label, ok, hint in items:
        if ok:
            print(f"  [x] {label}")
        else:
            missing += 1
            print(f"  [ ] {label}  →  {hint}")
    print("  全部就绪，可以开始接待" if missing == 0 else f"  还差 {missing} 项")
    print("[OK] 自检完成")
    return 0


def cmd_capture(args) -> int:
    cfg = load_config(args.config)
    platform = MockPlatform() if args.mock or not IS_WINDOWS else build_real_platform(cfg)
    image = platform.capture()
    if image is None:
        print("抓图失败")
        return 1
    layout = LayoutCalculator(
        cfg.layout.nav_ratio, cfg.layout.list_ratio,
        cfg.layout.header_ratio, cfg.layout.input_ratio, cfg.layout.overrides,
    )
    regions = layout.regions(image.width, image.height)
    out = args.out or "capture.png"
    image.save(out)
    annotated = annotate(image.copy(), regions, layout)
    annotated_path = out.rsplit(".", 1)[0] + "-regions.png"
    annotated.save(annotated_path)
    print(f"已保存：{out}")
    print(f"已保存标注：{annotated_path}")
    for name, box in regions.to_dict().items():
        print(f"  {name:14s} {box}")
    return 0


def cmd_mock(args) -> int:
    cfg = load_config(args.config)
    platform = MockPlatform()
    agent = Agent(cfg, platform, dry_run=not args.live)
    agent.run(cycles=args.cycles)
    print("\n== 审计最近记录 ==")
    for row in agent.audit.recent(6):
        ts, peer, action, status, reply, detail = row
        print(f"{action:8s} {status:12s} peer={peer} reply={reply[:40]!r} detail={detail[:40]!r}")
    agent.close()
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    platform = build_real_platform(cfg)
    agent = Agent(cfg, platform, dry_run=not args.live)
    agent.run(once=args.once, cycles=args.cycles)
    agent.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="微信 AI 自动回复（融合版）")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="环境自检")
    doctor.add_argument("--mock", action="store_true", help="用 mock 窗口自检")
    doctor.set_defaults(func=cmd_doctor)

    capture = sub.add_parser("capture", help="抓窗口图并标注区域（校准布局）")
    capture.add_argument("--out", default="capture.png")
    capture.add_argument("--mock", action="store_true")
    capture.set_defaults(func=cmd_capture)

    mock = sub.add_parser("mock", help="在合成窗口上跑全链路（macOS 可用的演示）")
    mock.add_argument("--live", action="store_true", help="在 mock 世界里真实走发送链")
    mock.add_argument("--cycles", type=int, default=2, help="最多处理几条消息后退出")
    mock.set_defaults(func=cmd_mock)

    run = sub.add_parser("run", help="真机运行（Windows）")
    run.add_argument("--live", action="store_true", help="真实发送；默认 dry-run 只出草稿")
    run.add_argument("--once", action="store_true", help="只跑一轮")
    run.add_argument("--cycles", type=int, default=None)
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
