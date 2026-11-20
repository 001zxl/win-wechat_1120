#!/usr/bin/env python3
"""生成 mock 窗口截图，便于离线调整布局比例。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wxreply.perception.layout import LayoutCalculator
from wxreply.platform.mock import MockPlatform


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mock")
    os.makedirs(out_dir, exist_ok=True)
    platform = MockPlatform()
    platform.capture().save(os.path.join(out_dir, "list.png"))
    platform.force_open("张三")
    platform.capture().save(os.path.join(out_dir, "chat.png"))
    regions = LayoutCalculator().regions(platform.WIDTH, platform.HEIGHT)
    print("已生成：", out_dir)
    for name, box in regions.to_dict().items():
        print(f"  {name:14s} {box}")


if __name__ == "__main__":
    main()
