from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

try:  # macOS 自带 Vision OCR（需 pyobjc-framework-Vision）
    import Quartz
    import Vision
    from CoreFoundation import CFDataCreate

    HAVE_VISION = True
except Exception:  # pragma: no cover
    HAVE_VISION = False


@dataclass
class OcrLine:
    text: str
    bbox: tuple  # x1, y1, x2, y2
    score: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class VisionOcr:
    """macOS Vision 框架 OCR，无需安装 RapidOCR。"""

    def __init__(self) -> None:
        self._request = None

    @staticmethod
    def available() -> bool:
        return HAVE_VISION

    def recognize(self, image: Image.Image) -> List[OcrLine]:
        import io

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        cf_data = CFDataCreate(None, data, len(data))
        source = Quartz.CGImageSourceCreateWithData(cf_data, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        try:
            request.setRecognitionLevel_(0)  # accurate
            request.setUsesLanguageCorrection_(True)
            request.setRecognitionLanguages_(["zh-Hans", "en-US"])
        except Exception:
            pass
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        handler.performRequests_error_([request], None)
        width, height = image.width, image.height
        lines: List[OcrLine] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string() or "").strip()
            if not text:
                continue
            confidence = float(candidate.confidence())
            bbox = observation.boundingBox()
            x = float(bbox.origin.x) * width
            w = float(bbox.size.width) * width
            h = float(bbox.size.height) * height
            # Vision 原点在左下角，转成左上角像素坐标
            y = (1.0 - float(bbox.origin.y) - float(bbox.size.height)) * height
            lines.append(OcrLine(text, (x, y, x + w, y + h), confidence))
        return lines


class OcrEngine:
    """文字轨：macOS 用 Vision，Windows 用 RapidOCR。"""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._backend = None  # "vision" | "rapidocr"
        self._engine = None
        self._vision = None
        self._load_error: Optional[str] = None
        self._tried = False

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def load_error(self) -> Optional[str]:
        self._load()
        return self._load_error

    def _load(self):
        if self._tried:
            return self._engine or self._vision
        self._tried = True
        if sys.platform == "darwin" and VisionOcr.available():
            try:
                self._vision = VisionOcr()
                self._backend = "vision"
                return self._vision
            except Exception as exc:
                self._load_error = f"Vision OCR 初始化失败: {exc}"
        try:
            from rapidocr_onnxruntime import RapidOCR

            try:
                # 限制推理线程：默认会开满全部核心，28 线程空转会吃满 CPU，拖慢整个控制台
                self._engine = RapidOCR(intra_op_num_threads=4)
            except TypeError:
                self._engine = RapidOCR()
            self._backend = "rapidocr"
        except Exception as exc:
            if self._load_error is None:
                self._load_error = f"RapidOCR 不可用: {exc}"
            return None
        return self._engine

    def recognize(self, image: Image.Image) -> List[OcrLine]:
        backend = self._load()
        if backend is None:
            raise RuntimeError(self._load_error or "OCR 不可用")
        if self._backend == "vision":
            return self._vision.recognize(image)
        array = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        raw = self._engine(array)
        return self._normalize(raw)

    def _normalize(self, raw) -> List[OcrLine]:
        lines: List[OcrLine] = []
        boxes = texts = scores = None
        if hasattr(raw, "txts"):
            boxes, texts, scores = raw.boxes, raw.txts, raw.scores
        elif isinstance(raw, tuple) and len(raw) == 2:
            result = raw[0]
            if hasattr(result, "txts"):
                boxes, texts, scores = result.boxes, result.txts, result.scores
            else:
                for item in result or []:
                    if len(item) < 3:
                        continue
                    box, text, score = item[0], item[1], item[2]
                    lines.append(self._line(box, text, score))
                return lines
        if boxes is None:
            return lines
        for box, text, score in zip(boxes, texts, scores):
            lines.append(self._line(box, text, score))
        return lines

    def _line(self, box, text: str, score: float) -> OcrLine:
        points = np.array(box).reshape(-1, 2)
        x1, y1 = points[:, 0].min(), points[:, 1].min()
        x2, y2 = points[:, 0].max(), points[:, 1].max()
        return OcrLine(str(text), (float(x1), float(y1), float(x2), float(y2)), float(score))

    def read_text(self, image: Image.Image) -> str:
        lines = [line for line in self.recognize(image) if line.score >= self.min_confidence]
        lines.sort(key=lambda item: item.center[1])
        return "".join(line.text for line in lines).strip()
