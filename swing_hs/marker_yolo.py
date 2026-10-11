# -*- coding: utf-8 -*-
"""swing_hs 在线黑标候选：黑标 YOLO 在 FK 锚附近的原始 Bayer 小块上找黑标（替换阈值暗块候选器）。

用户 1011：「我们 train 一个 final 版本的，然后上线 100fps 的在线识别」。阈值暗块候选器（racket_ht_black_marker.
_marker_candidates）夜里增益顶满时满块噪点（每块 ~12 个候选、面积中位 12 px），在线解多数是错的，自学的「黑标 − FK」
偏差也会学偏（1011 115528 白天场同样：偏差 [+47, −59, +14] mm，在线解和人工核对过的黑标 ≤ 5 px 的只有 25 %）。

模型：tennis-man scratch/marker_label_1011/yolo（yolov8n，人工复核页确认的 swing_hs 5 fps 原分辨率录像帧训练，整幅 ROI
原分辨率 imgsz 1152），导出成固定输入 S×S（96）的 ONNX，见 yolo_model/black_marker_yolo.json。
小块（crop_marker_patch：±(48+4) px，左上角偶数对齐，Bayer 相位与全幅一致）解码后以锚点为中心原尺寸截 S×S，不缩放
（训练时 ROI 长边拉到 1152，黑标尺度是原分辨率的 1.0–1.22 倍，原尺寸在其中）；块不够 S（贴 ROI 边）的那一维补 114 灰
（同 ultralytics letterbox）。1011 用 115405 录制时的在线小块回放、人工核对的 361 个黑标比过：截 96 找对 330，
整块补成 128（灰 / 镜像 / 复制边）只有 310–315，整幅 ROI 推理 337——补边伤准度，截 96 几乎不丢。
输出：置信 ≥ CONF_MIN 的框，贪心 NMS 后最多 MAX_CANDIDATES 个，同旧候选行格式 (u, v, score, area, bx, by, bw, bh)
（全幅像素；u, v = 框心，score = 置信 0–1，area = 框面积），主进程三角化只用 (u, v)，不用改。
CPU 单线程推理（每台相机 DETECT_WORKERS 个识别进程各一个会话，不碰 GPU）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "yolo_model" / "black_marker_yolo.onnx"
#: 给多视图留余地的低门槛：主进程三角化要 ≥ 3 台在 2.5 px 内一致，单台低分的假框凑不成解；高于这个的才进候选
CONF_MIN = 0.10
NMS_IOU = 0.5
MAX_CANDIDATES = 3
PAD_GRAY = 114


def model_info(path: Path = MODEL_PATH) -> dict:
    """模型旁边的说明（训练数据、验证结果）+ 现场算的 sha256，开录时写进 meta.json；模型不在就在开相机之前报错。"""
    import hashlib

    if not path.is_file():
        raise FileNotFoundError(f"黑标 YOLO 模型不在：{path}")
    side = path.with_suffix(".json")
    info = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    return {**info, "model": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class MarkerYolo:
    """一个 ONNX Runtime 会话（CPU、单线程）。"""

    def __init__(self, path: Path = MODEL_PATH, *, threads: int = 1) -> None:
        import onnxruntime as ort

        if not Path(path).is_file():
            raise FileNotFoundError(f"黑标 YOLO 模型不在：{path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self._canvas = np.full((self.size, self.size, 3), PAD_GRAY, dtype=np.uint8)

    def candidates(self, bgr: np.ndarray, origin: tuple[int, int], anchor_uv: tuple[float, float]) -> list[tuple]:
        """解码好的小块（BGR）+ 块左上全幅坐标 + 锚点（全幅）→ 候选行 [(u, v, score, area, bx, by, bw, bh)]，全幅像素。"""
        S = self.size
        gx, gy = int(origin[0]), int(origin[1])
        h, w = bgr.shape[:2]
        sx = sy = 0
        if w > S or h > S:     # 块比画布大：以锚点为中心截 S×S
            sx = int(np.clip(round(float(anchor_uv[0]) - gx - S / 2), 0, max(0, w - S)))
            sy = int(np.clip(round(float(anchor_uv[1]) - gy - S / 2), 0, max(0, h - S)))
            bgr = bgr[sy:sy + S, sx:sx + S]
            h, w = bgr.shape[:2]
        px, py = (S - w) // 2, (S - h) // 2
        canvas = self._canvas
        canvas[:] = PAD_GRAY
        canvas[py:py + h, px:px + w] = bgr
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] * (1.0 / 255.0)
        out = self.session.run(None, {self.input_name: x})[0][0]      # (5, N)：cx, cy, w, h, conf（画布像素）
        conf = out[4]
        idx = np.flatnonzero(conf >= CONF_MIN)
        if idx.size == 0:
            return []
        idx = idx[np.argsort(-conf[idx])]
        boxes = np.stack([out[0, idx] - out[2, idx] / 2, out[1, idx] - out[3, idx] / 2,
                          out[0, idx] + out[2, idx] / 2, out[1, idx] + out[3, idx] / 2], axis=1)
        keep: list[int] = []
        for k in range(len(idx)):
            if all(_iou(boxes[k], boxes[j]) <= NMS_IOU for j in keep):
                keep.append(k)
                if len(keep) >= MAX_CANDIDATES:
                    break
        dx, dy = gx + sx - px, gy + sy - py        # 画布像素 → 全幅像素
        rows = []
        for k in keep:
            x0, y0, x1, y1 = (float(v) for v in boxes[k])
            rows.append((round(0.5 * (x0 + x1) + dx, 2), round(0.5 * (y0 + y1) + dy, 2), round(float(conf[idx[k]]), 3),
                         int(round((x1 - x0) * (y1 - y0))), int(round(x0 + dx)), int(round(y0 + dy)),
                         int(round(x1 - x0)), int(round(y1 - y0))))
        return rows


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / union) if union > 0 else 0.0
