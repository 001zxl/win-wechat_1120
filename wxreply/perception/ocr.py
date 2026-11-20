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


class PaddleOcrBackend:
    """PaddleOCR 可选后端（3.x / 2.x 都兼容），用于和 RapidOCR 做 A/B 对比。

    3.x 的返回体是 `OCRResult`（dict 子类，键为 rec_texts / rec_scores / rec_polys /
    dt_polys / rec_boxes），2.x 是 [[box, (text, score)], ...] 的嵌套列表。
    两种格式都在这里归一化成 OcrLine，业务侧不用关心装的是哪一版。
    """

    name = "paddleocr"
    # 3.x 新增 / 2.x 保留的参数名，初始化时按版本尝试
    _V3_KWARGS = dict(lang="ch", use_doc_orientation_classify=False,
                      use_doc_unwarping=False, use_textline_orientation=False)

    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # type: ignore

        self.version = str(getattr(PaddleOCR, "__version__", "")) or \
            str(getattr(__import__("paddleocr"), "__version__", ""))
        self._engine = None
        errors = []
        for kwargs in (self._V3_KWARGS,
                       {"use_angle_cls": True, "lang": "ch", "show_log": False},
                       {"lang": "ch"}):
            try:
                self._engine = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError) as exc:   # 版本不认这些参数就换下一组
                errors.append(str(exc))
        if self._engine is None:
            raise RuntimeError("PaddleOCR 初始化失败：" + "；".join(errors[:2]))

    # ---------- 推理 ----------
    def recognize(self, image: "Image.Image") -> List["OcrLine"]:
        import numpy as np

        array = np.array(image.convert("RGB"))[:, :, ::-1]   # RGB -> BGR
        raw = self._infer(array)
        return self.parse_result(raw)

    def _infer(self, array):
        """3.x 推荐 predict()，2.x 只有 ocr()；方法不认就互换一次。"""
        engine = self._engine
        predict = getattr(engine, "predict", None)
        ocr = getattr(engine, "ocr", None)
        if callable(predict):
            try:
                return predict(array)
            except (TypeError, ValueError):
                if not callable(ocr):
                    raise
        return ocr(array)

    # ---------- 结果解析（纯函数，便于离线回放官方样例） ----------
    @classmethod
    def parse_result(cls, raw) -> List["OcrLine"]:
        lines: List[OcrLine] = []
        for page in cls._pages(raw):
            parsed = cls._parse_page(page)
            lines.extend(parsed)
        return lines

    @staticmethod
    def _pages(raw) -> List:
        if raw is None:
            return []
        if hasattr(raw, "json") or isinstance(raw, dict):
            return [raw]
        try:
            seq = list(raw)
        except TypeError:
            return []
        if not seq:
            return []
        first = seq[0]
        # 2.x 单页形态：[ [box, (text, score)], ... ] —— 外面少一层 page 列表时补上
        if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[1], tuple):
            return [seq]
        return seq

    @classmethod
    def _as_dict(cls, page) -> Optional[dict]:
        """把 3.x 的 OCRResult 摊平成 {rec_texts, rec_scores, rec_polys, ...}。

        3.x 的 OCRResult 是 dict 子类，内容挂在 `res` 键下（print 出来是 {'res': {...}}）；
        它**不支持** res.rec_texts 属性访问，所以这里只认 key / .json / .res。
        """
        if isinstance(page, dict):
            if "rec_texts" in page:
                return page
            inner = page.get("res")
            if isinstance(inner, dict):
                return inner
            data = getattr(page, "json", None)
            if isinstance(data, dict):
                nested = data.get("res")
                return nested if isinstance(nested, dict) else data
            return None
        data = getattr(page, "json", None)
        if isinstance(data, dict):
            inner = data.get("res")
            return inner if isinstance(inner, dict) else data
        try:
            res = getattr(page, "res", None)
            if isinstance(res, dict):
                return res
        except Exception:
            pass
        return None

    @classmethod
    def _parse_page(cls, page) -> List["OcrLine"]:
        data = cls._as_dict(page)
        if data is not None:
            return cls._parse_v3(data)
        return cls._parse_v2(page)

    @classmethod
    def _parse_v3(cls, data: dict) -> List["OcrLine"]:
        texts = data.get("rec_texts")
        if texts is None:
            return []
        scores = data.get("rec_scores")
        scores = [] if scores is None else scores
        polys = data.get("rec_polys")
        if polys is None or len(polys) == 0:
            polys = data.get("dt_polys")
        if polys is None or len(polys) == 0:
            polys = data.get("rec_boxes")
        polys = [] if polys is None else list(polys)
        lines: List[OcrLine] = []
        for index, text in enumerate(texts):
            text = str(text or "").strip()
            if not text:
                continue
            try:
                score = float(scores[index])
            except (IndexError, TypeError, ValueError):
                score = 1.0
            box = polys[index] if index < len(polys) else None
            lines.append(OcrLine(text, cls._box(box), score))
        return lines

    @classmethod
    def _parse_v2(cls, page) -> List["OcrLine"]:
        lines: List[OcrLine] = []
        for item in page or []:
            try:
                box, payload = item[0], item[1]
                text = str(payload[0] or "").strip()
                score = float(payload[1])
            except Exception:
                continue
            if not text:
                continue
            lines.append(OcrLine(text, cls._box(box), score))
        return lines

    @staticmethod
    def _box(box) -> tuple:
        if box is None:
            return (0.0, 0.0, 0.0, 0.0)
        points = np.array(box, dtype=float)
        if points.ndim == 1 and points.size >= 4:      # [x1, y1, x2, y2]
            x1, y1, x2, y2 = points[:4]
            return (float(min(x1, x2)), float(min(y1, y2)),
                    float(max(x1, x2)), float(max(y1, y2)))
        points = points.reshape(-1, 2)
        return (float(points[:, 0].min()), float(points[:, 1].min()),
                float(points[:, 0].max()), float(points[:, 1].max()))


class OcrEngine:
    """文字轨：macOS 用 Vision；Windows 默认 RapidOCR，可切换到 PaddleOCR 做对比。"""

    def __init__(self, min_confidence: float = 0.5, backend: str = "auto") -> None:
        self.min_confidence = min_confidence
        self.preferred = (backend or "auto").strip().lower()
        self._backend = None  # "vision" | "rapidocr" | "paddleocr"
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

    @property
    def last_error(self) -> Optional[str]:
        """不触发加载的最近一次错误（状态轮询用）。"""
        return self._load_error

    @property
    def backend_name(self) -> str:
        """实际生效的后端（控制台要显示，避免"配了 Paddle 却在跑 RapidOCR"）。

        不强制加载模型：状态轮询不能因为没装 OCR 就卡住首屏。
        """
        if self._backend:
            return self._backend
        if self._tried:
            return "none"
        return self.preferred or "auto"

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
        if self.preferred in ("paddle", "paddleocr"):
            try:
                self._engine = PaddleOcrBackend()
                self._backend = "paddleocr"
                return self._engine
            except Exception as exc:
                self._load_error = f"PaddleOCR 不可用，已退回 RapidOCR: {exc}"
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
        if self._backend == "paddleocr":
            return self._engine.recognize(image)
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
