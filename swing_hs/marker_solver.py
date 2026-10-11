# -*- coding: utf-8 -*-
"""swing_hs 主进程的黑标在线三角化：每个触发格把各相机的候选凑齐，快速解一个点（100 Hz 要 ~1 ms）。

收场黑标工具（racket_ht_black_marker._attempt_solve）每帧全组合 + LM 精修 + 留一，一帧 ~0.1 s，在线跑不动。这里：
  1. 各相机候选按离锚点投影的远近取前 TOP_K 个，去畸变；
  2. 两两相机 × 候选对做两视 DLT 出假设点（≤ 6 × TOP_K² 个，批量 SVD）；
  3. 每个假设投回各相机（去畸变像素），最近候选 ≤ ASSOC_PX 算内点；
  4. 内点相机多者优先、再比误差平方和、再比离锚距离；过锚点门；
  5. 全部内点重做 DLT，最大重投影 > MAX_REPROJ_PX 就丢最差那台重来（≥ min_cams 台为止）。
锚点 = 同曝光 FK 黑标；本场「黑标 − FK」车体系偏差（收场同一个稳健中位数 _session_marker_offset）稳定后锚点
加上它、门换成视线深度 / 横向分开的 AnchorGate（在线 70 / 120 mm，比收场的 35 / 80 mm 宽，见 ONLINE_REANCHOR_*）。
偏差没定之前用球形门：4 台 110 mm、3 台 80 mm（同收场 recovery 门）。
至少 3 台（同收场工具）：两台的解只受一条极线约束，截块里随便两个暗点就能凑出 < 2.5 px 的解（1010 假相机
实测锚点偏 ~10 cm 时两台解几乎全是假点）。在线结果供监看、进 bag；全部候选在 marker.jsonl 里，可离线重解。
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from swing_hs import geometry

#: 每台取离锚点最近的几个候选做假设、内点关联门。1010 合成扫参（4 台真实标定、每台 8 个杂点、0.4 px 噪声）：
#: 4/4 px → 3/3 px 纯杂点误收 19% → 14%，有黑标时 3 台 / 4 台解出率不变（100%）——那是阈值候选器满块杂点时的账。
#: 1011 放宽到 5 / 5 px（用户「把门放开一点」）：候选换成黑标 YOLO 后每台 ≤ 3 个、几乎没有杂点；18F 标定本身有几 px
#: 系统误差（YOLO 框留一投影中位 2.5–5.6 px），3 / 2.5 px 会把一致的 3–4 台拆散。swing_hs_20261011_155720 用录制时的
#: YOLO 候选重放：像素门 5 / 5 + 锚点门 70 / 120 → 反手 J1 ∈ [−210°, −135°] 解出 56.7 → 95.3 %、正手 [−45°, 30°]
#: 83.4 → 99.9 %；±50 ms 二次拟合偏 > 20 mm 的可疑解 0.6 % / 0 %，新增解和原有解在时间上连续（不是拍框上别的暗点）。
TOP_K = 3
ASSOC_PX = 5.0
#: 「和上一帧连续」：30 ms 内上一个解的（黑标 − FK 锚）偏差变化 ≤ 25 mm。真黑标的偏差（柔度 / 时间差）随时间平滑变，
#: 杂点凑出来的解在锚点门里乱跳（合成扫参里离锚中位 63 mm）
SMOOTH_MAX_GAP_S = 0.03
SMOOTH_MAX_STEP_MM = 25.0
MAX_REPROJ_PX = 5.0          # 收场 MARKER_MAX_REPROJ_PX 仍是 2.5；在线放宽理由见 ASSOC_PX
MAX_ANCHOR_MM = 110.0        # 同收场 MARKER_MAX_EXPECTED_DISTANCE_MM（4 台）
MAX_ANCHOR_RECOVERY_MM = 80.0  # 同收场 MARKER_MAX_EXPECTED_RECOVERY_MM（少于 4 台）
#: 偏差学到后的 AnchorGate（在线专用；收场仍用 racket_ht_black_marker.REANCHOR_MAX_LATERAL / DEPTH_MM = 35 / 80 mm）。
#: 挥拍高速段 FK 锚落后于真黑标（在线关节角晚于曝光 + 柔度）：155720 里锚速 > 2 m/s 的解沿运动方向超前 5–16 ms、
#: 垂直偏 20–60 mm，横向 35 mm 正好挡掉挥得最快的那段（辨识时延 / 柔度最要的数据）。
ONLINE_REANCHOR_MAX_LATERAL_MM = 70.0
ONLINE_REANCHOR_MAX_DEPTH_MM = 120.0
MIN_CAMS = 3
#: 只用 ≥ 3 台相机的解估本场偏差
OFFSET_MIN_CAMS = 3
OFFSET_WINDOW = 400
OFFSET_REFRESH = 25
#: 按排名最多试这么多个假设
MAX_ATTEMPTS = 6


@dataclass
class MarkerSolution:
    xyz_mm: np.ndarray
    serials: list[str]
    pixels: dict[str, tuple[float, float]]        # 原始（带畸变）像素
    cand_index: dict[str, int]
    errors_px: dict[str, float]                   # 去畸变像素里的重投影误差
    anchor_dist_mm: float
    corrected: bool                               # 锚点用了本场偏差
    smooth: bool | None = None                    # 与 30 ms 内上一个解连续；None = 前面没有解

    @property
    def rms_px(self) -> float:
        values = np.asarray(list(self.errors_px.values()), dtype=np.float64)
        return float(np.sqrt(np.mean(values * values)))

    @property
    def max_px(self) -> float:
        return float(max(self.errors_px.values()))

    def to_json(self) -> dict:
        return {
            "xyz_mm": [round(float(v), 1) for v in self.xyz_mm],
            "cams": self.serials,
            "pixels": {s: [round(u, 2), round(v, 2)] for s, (u, v) in self.pixels.items()},
            "cand": self.cand_index,
            "err_px": {s: round(e, 2) for s, e in self.errors_px.items()},
            "rms_px": round(self.rms_px, 2),
            "anchor_dist_mm": round(self.anchor_dist_mm, 1),
            "corrected": self.corrected,
            "smooth": self.smooth,
        }


class FastMarkerSolver:
    def __init__(self, cameras: dict, *, min_cams: int = MIN_CAMS) -> None:
        """cameras = racket_ht_black_marker._load_cameras 的 CameraModel（P = K[R|t]，mm）。"""
        self.cameras = cameras
        self.min_cams = max(2, int(min_cams))
        self.P = {s: np.asarray(c.P, dtype=np.float64) for s, c in cameras.items()}

    def undistort(self, serial: str, uv: np.ndarray) -> np.ndarray:
        camera = self.cameras[serial]
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, camera.K, camera.D, P=camera.K).reshape(-1, 2)

    def project(self, serial: str, xyz_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """无畸变投影 → (去畸变像素 (N, 2), 深度 (N,))。"""
        P = self.P[serial]
        X = np.atleast_2d(np.asarray(xyz_mm, dtype=np.float64))
        h = X @ P[:, :3].T + P[:, 3]
        depth = h[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = h[:, :2] / depth[:, None]
        return uv, depth

    def triangulate(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        """去畸变像素 → DLT（同 racket_ht_black_marker._dlt）。"""
        rows = []
        for serial, (u, v) in observations.items():
            P = self.P[serial]
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        h = vt[-1]
        if abs(float(h[3])) < 1e-12:
            raise ValueError("degenerate triangulation")
        return h[:3] / h[3]

    @staticmethod
    def _anchor_rank(offset_mm: np.ndarray, n_cams: int, gate) -> float | None:
        if gate is None:
            limit = MAX_ANCHOR_MM if n_cams >= 4 else MAX_ANCHOR_RECOVERY_MM
            distance = float(np.linalg.norm(offset_mm))
            return distance if distance <= limit else None
        return gate.rank(offset_mm)

    def solve(self, candidates: dict[str, list], anchor_mm: np.ndarray, *, gate=None,
              corrected: bool = False) -> MarkerSolution | None:
        """candidates: {序列号: [(u, v, …), …]}（原始像素）；anchor_mm: 锚点（世界 mm）。"""
        serials = [s for s in self.P if candidates.get(s)]
        if len(serials) < self.min_cams:
            return None
        anchor = np.asarray(anchor_mm, dtype=np.float64)
        raw: dict[str, np.ndarray] = {}
        und: dict[str, np.ndarray] = {}
        near: dict[str, np.ndarray] = {}
        for s in serials:
            raw[s] = np.asarray([c[:2] for c in candidates[s]], dtype=np.float64)
            und[s] = self.undistort(s, raw[s])
            a_uv, _ = self.project(s, anchor)
            near[s] = np.argsort(np.linalg.norm(und[s] - a_uv[0], axis=1))[:TOP_K]
        blocks = []
        for s1, s2 in itertools.combinations(serials, 2):
            P1, P2 = self.P[s1], self.P[s2]
            i1, i2 = (a.ravel() for a in np.meshgrid(near[s1], near[s2], indexing="ij"))
            u1, u2 = und[s1][i1], und[s2][i2]
            blocks.append(np.stack([
                u1[:, :1] * P1[2] - P1[0], u1[:, 1:2] * P1[2] - P1[1],
                u2[:, :1] * P2[2] - P2[0], u2[:, 1:2] * P2[2] - P2[1],
            ], axis=1))
        _, _, vt = np.linalg.svd(np.concatenate(blocks))
        homogeneous = vt[:, -1, :]
        good = np.abs(homogeneous[:, 3]) > 1e-12
        if not good.any():
            return None
        X = homogeneous[good, :3] / homogeneous[good, 3:4]
        n_in = np.zeros(len(X), dtype=np.int64)
        sse = np.zeros(len(X), dtype=np.float64)
        assoc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in serials:
            uv, depth = self.project(s, X)
            d = np.linalg.norm(uv[:, None, :] - und[s][None, :, :], axis=2)
            j = np.argmin(d, axis=1)
            dmin = d[np.arange(len(X)), j]
            inlier = (depth > 0.0) & (dmin <= ASSOC_PX)
            n_in += inlier
            sse += np.where(inlier, dmin * dmin, 0.0)
            assoc[s] = (j, inlier)
        ranks = np.full(len(X), np.inf)
        for h in range(len(X)):
            if n_in[h] >= self.min_cams:
                r = self._anchor_rank(X[h] - anchor, int(n_in[h]), gate)
                if r is not None:
                    ranks[h] = r
        order = [h for h in np.lexsort((ranks, sse, -n_in)) if np.isfinite(ranks[h])]
        tried: set[tuple] = set()
        for h in order:
            chosen = {s: int(assoc[s][0][h]) for s in serials if assoc[s][1][h]}
            key = tuple(sorted(chosen.items()))
            if key in tried:
                continue
            tried.add(key)
            solution = self._finalize(chosen, raw, und, anchor, gate, corrected)
            if solution is not None:
                return solution
            if len(tried) >= MAX_ATTEMPTS:
                break
        return None

    def _finalize(self, chosen: dict[str, int], raw, und, anchor, gate, corrected) -> MarkerSolution | None:
        chosen = dict(chosen)
        while len(chosen) >= self.min_cams:
            try:
                X = self.triangulate({s: und[s][i] for s, i in chosen.items()})
            except (ValueError, np.linalg.LinAlgError):
                return None
            errors: dict[str, float] = {}
            for s, i in chosen.items():
                uv, depth = self.project(s, X)
                errors[s] = float(np.linalg.norm(uv[0] - und[s][i])) if depth[0] > 0.0 else math.inf
            worst = max(errors, key=errors.get)
            if errors[worst] <= MAX_REPROJ_PX:
                break
            if len(chosen) == self.min_cams:
                return None
            del chosen[worst]
        else:
            return None
        if self._anchor_rank(X - anchor, len(chosen), gate) is None:
            return None
        return MarkerSolution(
            xyz_mm=X,
            serials=sorted(chosen),
            pixels={s: (float(raw[s][i][0]), float(raw[s][i][1])) for s, i in chosen.items()},
            cand_index=dict(chosen),
            errors_px=errors,
            anchor_dist_mm=float(np.linalg.norm(X - anchor)),
            corrected=corrected,
        )


class SessionOffset:
    """本场「黑标 − 同曝光 FK」车体系偏差（mm），在线滚动：收场同一个稳健中位数。"""

    def __init__(self, window: int = OFFSET_WINDOW, refresh: int = OFFSET_REFRESH) -> None:
        self._samples: deque[np.ndarray] = deque(maxlen=window)
        self._refresh = refresh
        self._since = 0
        self.value: np.ndarray | None = None
        self.inliers = 0

    def add(self, offset_car_mm: np.ndarray) -> None:
        rbm = geometry.load_test_src("racket_ht_black_marker")
        self._samples.append(np.asarray(offset_car_mm, dtype=np.float64))
        self._since += 1
        if self._since < self._refresh and self.value is not None:
            return
        if len(self._samples) < rbm.REANCHOR_MIN_INLIERS:
            return
        self._since = 0
        result = rbm._session_marker_offset(list(self._samples))
        if result is not None:
            self.value, self.inliers = result


class MarkerFusion:
    """一个触发格：锚点（FK）+ 车 yaw + 各相机候选 → 黑标世界坐标。"""

    def __init__(self, cameras: dict, serials: list[str]) -> None:
        self.cameras = cameras
        self.serials = list(serials)
        self.solver = FastMarkerSolver({s: cameras[s] for s in serials})
        self.offset = SessionOffset()
        self._last: tuple[float, np.ndarray] | None = None   # (t, 黑标 − FK 锚) 上一个解

    def solve_slot(self, anchor_mm: np.ndarray, yaw: float, candidates: dict[str, list],
                   t: float | None = None) -> tuple[MarkerSolution | None, np.ndarray]:
        """t = 曝光中点（PC 轴），给了就判「和上一帧连续」。"""
        rbm = geometry.load_test_src("racket_ht_black_marker")
        anchor = np.asarray(anchor_mm, dtype=np.float64)
        gate = None
        corrected = self.offset.value is not None
        use = anchor
        if corrected:
            use = anchor + rbm._car_to_world(yaw, self.offset.value)
            seeing = [s for s in self.serials if candidates.get(s)] or self.serials
            gate = rbm.AnchorGate(
                view_dir=rbm._mean_view_dir(self.cameras, seeing, use),
                max_lateral_mm=ONLINE_REANCHOR_MAX_LATERAL_MM,
                max_depth_mm=ONLINE_REANCHOR_MAX_DEPTH_MM,
            )
        solution = self.solver.solve(candidates, use, gate=gate, corrected=corrected)
        if solution is not None:
            deviation = solution.xyz_mm - anchor
            if t is not None:
                if self._last is not None and 0.0 < t - self._last[0] <= SMOOTH_MAX_GAP_S:
                    solution.smooth = bool(np.linalg.norm(deviation - self._last[1]) <= SMOOTH_MAX_STEP_MM)
                if self._last is None or t > self._last[0]:
                    self._last = (t, deviation)
            if len(solution.serials) >= OFFSET_MIN_CAMS:
                self.offset.add(rbm._world_to_car(yaw, deviation))
        return solution, use
