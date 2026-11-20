from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .layout import Regions


@dataclass
class Badge:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self):
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)
