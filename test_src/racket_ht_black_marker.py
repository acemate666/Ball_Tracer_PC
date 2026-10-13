# -*- coding: utf-8 -*-
"""Measure the fixed black racket-centre marker around report final HTs.

候选（黑方块 2D 位置）不再从半分辨率拼接录像里找：tracker 在线在全分辨率图上截小块找好、
写进 <run>_racket_marker.json（src/racket_marker_online.py）；这里只用报告对时后的
模型黑标作锚在这些候选里三角化、选点。模型黑标 = 臂生产代码的端点动态模型（test_src/arm_production.py：
/joint_states 实测关节 → 时延 / 柔度 / 间隙 / J1 力矩轴倾斜 → 黑标，与端点标定、RL 训练器同一份）。
不再做本场「黑标 − FK」偏差校正（re-anchor）：端点模型本身就是拿黑标拟合的；视觉 − 模型的残差逐帧写进输出、
按正 / 反手与离 HT 的时间窗汇总（model_residual），没解释的量单独列出来，不拿锚吸收。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


# Scan the whole pre-HT approach (park + swing start at ~ht-270ms) plus the
# contact neighbourhood, so every throw carries its racket trajectory before ht.
PRE_WINDOW_S = 0.45
POST_WINDOW_S = 0.12
MARKER_ROI_RADIUS_PX = 250
MARKER_CANDIDATES_PER_CAMERA = 12
MARKER_ASSOCIATION_PX = 20.0
MARKER_MAX_REPROJ_PX = 2.5
MARKER_MAX_LOO_MM = 10.0
MARKER_MAX_HELDOUT_PX = 6.0
# Automatic post-run measurement has no human overlay review. 110 mm retains the
# validated marker in recent sessions while rejecting the known racket-rim alias.
MARKER_MAX_EXPECTED_DISTANCE_MM = 110.0
# Three-camera and trajectory-anchored recoveries carry weaker geometry, so they
# only accept fits much closer to their anchor (validated on 20260901_115811:
# the worst 3-cam outliers all sat at expected distance 100-106 mm).
MARKER_MAX_EXPECTED_RECOVERY_MM = 80.0
# Neighbour visual fits usable as a trajectory anchor: interpolate across at
# most one missing frame per side; extrapolate at most two frames beyond.
TRAJ_NEIGHBOR_MAX_SIDE_S = 0.095
TRAJ_EXTRAP_MAX_S = 0.062
ARM_STATE_MAX_GAP_S = 0.1
CAR_LOC_MAX_GAP_S = 0.5
# 模型黑标锚的门（v05 四相机都在底线后、离拍 7–8.5 m、两两视线夹角 16–55°，tracker_20260929_221737 逐帧目视 50 帧定）：
# 深度方向三角化噪声大（真检出 y 散 ±57 mm），横向真检出 ≤25 mm、拍框 / 背景假点 47–82 mm ⇒ 沿相机平均视线的深度分量与
# 垂直视线的横向分量分开限；黑方块只有 ~5 px，1 px 质心误差在留一三角化里就是 ~10 mm 深度，不设 LOO 毫米门（一致性由
# 重投影 2.5 px + 留一回投 6 px 把关）。锚 = 端点动态模型黑标本身，不加本场偏差。
ANCHOR_MAX_LATERAL_MM = 35.0
ANCHOR_MAX_DEPTH_MM = 80.0
MARKER_LOO_MAX_MM: float | None = None
# 视觉 − 模型残差的汇总时间窗（相对报告 FinalHT，ms）
RESIDUAL_WINDOWS_MS = ((-450.0, -100.0), (-100.0, -20.0), (-20.0, 20.0), (20.0, 120.0))


@dataclass(frozen=True)
class AnchorGate:
    """校正锚点的各向异性门：沿相机平均视线的深度分量与垂直于视线的横向分量分开限。"""

    view_dir: np.ndarray
    max_lateral_mm: float
    max_depth_mm: float

    def split(self, offset_mm: np.ndarray) -> tuple[float, float]:
        depth = float(offset_mm @ self.view_dir)
        lateral = float(np.linalg.norm(offset_mm - depth * self.view_dir))
        return lateral, abs(depth)

    def rank(self, offset_mm: np.ndarray) -> float | None:
        lateral, depth = self.split(offset_mm)
        if lateral > self.max_lateral_mm or depth > self.max_depth_mm:
            return None
        return math.hypot(lateral / self.max_lateral_mm, depth / self.max_depth_mm)


class NothingToMeasure(Exception):
    """报告里一条抛都没有（本场 RK 没发过 /predict_hit_pos）：没有 HT 可测。

    与"合同缺失/对齐不可信"是两回事——那些说明报告本身不能用，必须报错；这里报告是好的，
    只是这一场没东西可量，后处理应当继续往下走。
    """


@dataclass(frozen=True)
class CameraModel:
    serial: str
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    rvec: np.ndarray
    P: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class MarkerCandidate:
    uv: tuple[float, float]
    score: float
    area: int
    bbox_xywh: tuple[int, int, int, int]


@dataclass(frozen=True)
class PointFit:
    xyz_mm: np.ndarray
    rms_px: float
    max_px: float


@dataclass(frozen=True)
class MarkerFit:
    point: PointFit
    pixels: dict[str, tuple[float, float]]
    loo_delta_mm: dict[str, float]
    loo_heldout_px: dict[str, float]
    expected_distance_mm: float


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _session_path(raw: Any, tracker_path: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (tracker_path.parent / path).resolve()


def _load_cameras(
    calibration_path: Path, serials: list[str]
) -> dict[str, CameraModel]:
    raw = _load_json(calibration_path).get("cameras")
    if not isinstance(raw, dict) or set(raw) != set(serials) or len(serials) != 4:
        raise ValueError("session must use exactly the four calibrated 18F cameras")
    cameras: dict[str, CameraModel] = {}
    for serial in serials:
        item = raw[serial]
        K = np.asarray(item["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(item["D"], dtype=np.float64).reshape(-1)
        R = np.asarray(item["R_world"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(item["t_world"], dtype=np.float64).reshape(3, 1)
        image_size = tuple(int(v) for v in item["image_size"])
        if D.shape != (5,) or image_size != (2048, 1536):
            raise ValueError(f"unexpected calibration shape for {serial}")
        if not all(np.all(np.isfinite(v)) for v in (K, D, R, t)):
            raise ValueError(f"nonfinite camera calibration for {serial}")
        cameras[serial] = CameraModel(
            serial=serial,
            K=K,
            D=D,
            R=R,
            t=t,
            rvec=cv2.Rodrigues(R)[0],
            P=K @ np.hstack([R, t]),
            image_size=image_size,
        )
    return cameras


def _project_raw(camera: CameraModel, xyz_mm: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(xyz_mm, dtype=np.float64).reshape(1, 3),
        camera.rvec,
        camera.t,
        camera.K,
        camera.D,
    )
    return projected.reshape(2)


def _dlt(
    observations: dict[str, tuple[float, float]],
    cameras: dict[str, CameraModel],
) -> np.ndarray:
    rows = []
    for serial, uv in observations.items():
        camera = cameras[serial]
        undistorted = cv2.undistortPoints(
            np.asarray([[uv]], dtype=np.float64), camera.K, camera.D, P=camera.K
        ).reshape(2)
        u, v = undistorted
        rows.extend([u * camera.P[2] - camera.P[0], v * camera.P[2] - camera.P[1]])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        raise ValueError("degenerate marker triangulation")
    return homogeneous[:3] / homogeneous[3]


def _triangulate_refined(
    observations: dict[str, tuple[float, float]],
    cameras: dict[str, CameraModel],
) -> PointFit:
    initial = _dlt(observations, cameras)

    def residual(xyz: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                _project_raw(cameras[serial], xyz) - np.asarray(uv)
                for serial, uv in observations.items()
            ]
        )

    optimized = least_squares(
        residual,
        initial,
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=100,
    )
    xyz = np.asarray(optimized.x, dtype=np.float64)
    if not optimized.success or not np.all(np.isfinite(xyz)):
        raise ValueError("marker refinement failed")
    errors = []
    for serial, uv in observations.items():
        camera = cameras[serial]
        depth = float((camera.R @ xyz.reshape(3, 1) + camera.t)[2, 0])
        if depth <= 0.0:
            raise ValueError("marker is behind a camera")
        errors.append(float(np.linalg.norm(_project_raw(camera, xyz) - np.asarray(uv))))
    values = np.asarray(errors, dtype=np.float64)
    return PointFit(
        xyz_mm=xyz,
        rms_px=float(np.sqrt(np.mean(values * values))),
        max_px=float(np.max(values)),
    )


def _component_candidates(
    image: np.ndarray,
    anchor_uv: tuple[float, float],
    *,
    radius: int,
    thresholds: list[float],
    anchor_limit_px: float | None,
    result_limit: int,
    prefer_near: bool = False,
    dedupe_px: float = 5.0,
) -> list[MarkerCandidate]:
    height, width = image.shape[:2]
    anchor_x, anchor_y = anchor_uv
    if not np.all(np.isfinite([anchor_x, anchor_y])):
        return []
    x0 = max(0, int(math.floor(anchor_x - radius)))
    y0 = max(0, int(math.floor(anchor_y - radius)))
    x1 = min(width, int(math.ceil(anchor_x + radius)))
    y1 = min(height, int(math.ceil(anchor_y + radius)))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = float(np.median(gray))
    found: list[MarkerCandidate] = []
    for threshold in thresholds:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (gray < threshold).astype(np.uint8), 8
        )
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if not (8 <= area <= 1200 and 3 <= w <= 50 and 3 <= h <= 50):
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            fill = area / float(w * h)
            if aspect > 2.4 or fill < 0.30:
                continue
            # 质心/对比度只在该连通域自己的包围盒里算：包围盒外权重本来全是 0。原先在整块
            # ROI 上做 labels==label 和加权和，O(连通域数 × ROI 像素)，一路画面 0.35–0.7 s
            box = gray[y : y + h, x : x + w]
            component = labels[y : y + h, x : x + w] == label
            weights = np.where(component, np.maximum(background - box, 1.0), 0.0)
            mass = float(weights.sum())
            if mass <= 0.0:
                continue
            ys, xs = np.indices(box.shape)
            u = x0 + x + float((weights * xs).sum() / mass)
            v = y0 + y + float((weights * ys).sum() / mass)
            distance = math.hypot(u - anchor_x, v - anchor_y)
            if anchor_limit_px is not None and distance > anchor_limit_px:
                continue
            contrast = float(np.median(background - box[component]))
            scale = 25.0 if anchor_limit_px is not None else 140.0
            found.append(
                MarkerCandidate(
                    uv=(u, v),
                    score=(
                        max(contrast, 1.0)
                        * fill
                        * min(area, 250)
                        / (1.0 + (distance / scale) ** 2)
                    ),
                    area=area,
                    bbox_xywh=(x0 + x, y0 + y, w, h),
                )
            )
    found.sort(
        key=(
            (lambda item: (math.hypot(item.uv[0] - anchor_x, item.uv[1] - anchor_y), -item.score))
            if prefer_near
            else (lambda item: -item.score)
        )
    )
    deduped: list[MarkerCandidate] = []
    for item in found:
        if all(
            np.linalg.norm(np.subtract(item.uv, old.uv)) > dedupe_px
            for old in deduped
        ):
            deduped.append(item)
        if len(deduped) == result_limit:
            break
    return deduped


def _marker_candidates(
    image: np.ndarray,
    anchor_uv: tuple[float, float],
    *,
    local_radius: int = 55,
    broad_radius: int | None = MARKER_ROI_RADIUS_PX,
) -> list[MarkerCandidate]:
    """锚点附近的黑方块候选：贴锚小块（local_radius，就近优先）+ 大范围按分数补找（broad_radius）。

    tracker 在线识别锚点准（同帧车位姿 + 同一曝光时刻关节角），只截小块：broad_radius=None。
    """
    if not np.all(np.isfinite(anchor_uv)):
        return []
    height, width = image.shape[:2]
    if (
        anchor_uv[0] < -local_radius
        or anchor_uv[0] >= width + local_radius
        or anchor_uv[1] < -local_radius
        or anchor_uv[1] >= height + local_radius
    ):
        return []
    x0 = max(0, int(anchor_uv[0]) - local_radius)
    y0 = max(0, int(anchor_uv[1]) - local_radius)
    x1 = min(width, int(anchor_uv[0]) + local_radius + 1)
    y1 = min(height, int(anchor_uv[1]) + local_radius + 1)
    local_gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    adaptive = (
        list(np.percentile(local_gray, [5, 10, 15, 20, 25, 30]))
        if local_gray.size
        else []
    )
    local = _component_candidates(
        image,
        anchor_uv,
        radius=local_radius,
        thresholds=sorted(
            set(float(v) for v in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56] + adaptive)
        ),
        anchor_limit_px=local_radius - 10.0,
        result_limit=12,
        prefer_near=True,
        dedupe_px=3.0,
    )
    broad = (
        []
        if broad_radius is None
        else _component_candidates(
            image,
            anchor_uv,
            radius=broad_radius,
            thresholds=[8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
            anchor_limit_px=None,
            result_limit=32,
        )
    )
    combined = [*local, *broad]
    combined.sort(
        key=lambda item: float(
            np.linalg.norm(np.asarray(item.uv) - np.asarray(anchor_uv))
        )
    )
    deduped: list[MarkerCandidate] = []
    for item in combined:
        if all(np.linalg.norm(np.subtract(item.uv, old.uv)) > 3.0 for old in deduped):
            deduped.append(item)
        if len(deduped) == MARKER_CANDIDATES_PER_CAMERA:
            break
    return deduped


def _solve_marker(
    candidates: dict[str, list[MarkerCandidate]],
    cameras: dict[str, CameraModel],
    expected_xyz_mm: np.ndarray,
    serials: list[str],
    max_expected_mm: float,
    *,
    loo_max_mm: float | None = MARKER_MAX_LOO_MM,
    gate: AnchorGate | None = None,
) -> MarkerFit | None:
    """gate 给定时，锚距门与排名改用它的深度/横向分量（max_expected_mm 只剩种子预筛）。"""
    if any(not candidates.get(serial) for serial in serials):
        return None
    proposed: dict[tuple[int, ...], float] = {}
    for left_index, left_serial in enumerate(serials):
        for right_serial in serials[left_index + 1 :]:
            for left in candidates[left_serial]:
                for right in candidates[right_serial]:
                    try:
                        seed = _dlt(
                            {left_serial: left.uv, right_serial: right.uv}, cameras
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if (
                        float(np.linalg.norm(seed - expected_xyz_mm))
                        > 1.5 * max_expected_mm
                    ):
                        continue
                    choice: list[int] = []
                    total_distance = 0.0
                    for serial in serials:
                        prediction = _project_raw(cameras[serial], seed)
                        distances = [
                            float(np.linalg.norm(prediction - np.asarray(item.uv)))
                            for item in candidates[serial]
                        ]
                        best_index = int(np.argmin(distances))
                        if distances[best_index] > MARKER_ASSOCIATION_PX:
                            break
                        choice.append(best_index)
                        total_distance += distances[best_index]
                    if len(choice) == len(serials):
                        key = tuple(choice)
                        proposed[key] = min(proposed.get(key, math.inf), total_distance)

    best: tuple[tuple[float, float, float, float], MarkerFit] | None = None
    for choice, _ in sorted(proposed.items(), key=lambda item: item[1])[:32]:
        pixels = {
            serial: candidates[serial][candidate_index].uv
            for serial, candidate_index in zip(serials, choice)
        }
        try:
            fit = _triangulate_refined(pixels, cameras)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fit.max_px > MARKER_MAX_REPROJ_PX:
            continue
        expected_distance = float(np.linalg.norm(fit.xyz_mm - expected_xyz_mm))
        if gate is None:
            if expected_distance > max_expected_mm:
                continue
            anchor_rank = expected_distance
        else:
            gated = gate.rank(fit.xyz_mm - expected_xyz_mm)
            if gated is None:
                continue
            anchor_rank = gated
        loo_delta: dict[str, float] = {}
        heldout: dict[str, float] = {}
        try:
            for dropped in serials:
                fit3 = _triangulate_refined(
                    {serial: uv for serial, uv in pixels.items() if serial != dropped},
                    cameras,
                )
                loo_delta[dropped] = float(np.linalg.norm(fit3.xyz_mm - fit.xyz_mm))
                heldout[dropped] = float(
                    np.linalg.norm(
                        _project_raw(cameras[dropped], fit3.xyz_mm) - pixels[dropped]
                    )
                )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if loo_max_mm is not None and max(loo_delta.values()) >= loo_max_mm:
            continue
        if max(heldout.values()) > MARKER_MAX_HELDOUT_PX:
            continue
        score_sum = sum(
            candidates[serial][candidate_index].score
            for serial, candidate_index in zip(serials, choice)
        )
        marker = MarkerFit(
            point=fit,
            pixels=pixels,
            loo_delta_mm=loo_delta,
            loo_heldout_px=heldout,
            expected_distance_mm=expected_distance,
        )
        rank = (anchor_rank, max(loo_delta.values()), fit.rms_px, -score_sum)
        if best is None or rank < best[0]:
            best = (rank, marker)
    return None if best is None else best[1]


def _interpolate(
    rows: list[dict],
    target: float,
    time_key: str,
    value_keys: tuple[str, ...],
    max_gap_s: float,
) -> dict | None:
    times = np.asarray([float(row[time_key]) for row in rows], dtype=np.float64)
    right = int(np.searchsorted(times, target))
    if right <= 0 or right >= len(rows):
        return None
    before, after = rows[right - 1], rows[right]
    span = float(after[time_key]) - float(before[time_key])
    if not (0.0 < span <= max_gap_s):
        return None
    fraction = (target - float(before[time_key])) / span
    result: dict[str, Any] = {"gap_s": span}
    for key in value_keys:
        a = np.asarray(before[key], dtype=np.float64)
        b = np.asarray(after[key], dtype=np.float64)
        value = a + fraction * (b - a)
        if not np.all(np.isfinite(value)):
            return None
        result[key] = float(value) if value.ndim == 0 else value
    return result


def _interpolate_yaw(rows: list[dict], target: float, max_gap_s: float) -> float | None:
    """Wrap-aware yaw interpolation over the yaw-valid car rows only.

    Single-tag degraded frames publish yaw=null; requiring yaw on the same rows
    as position knocked out ~29% of otherwise measurable frames (0901_115811),
    so yaw interpolates independently across those holes.
    """
    times = np.asarray([float(row["elapsed_s"]) for row in rows], dtype=np.float64)
    right = int(np.searchsorted(times, target))
    if right <= 0 or right >= len(rows):
        return None
    before, after = rows[right - 1], rows[right]
    span = float(after["elapsed_s"]) - float(before["elapsed_s"])
    if not (0.0 < span <= max_gap_s):
        return None
    fraction = (target - float(before["elapsed_s"])) / span
    a, b = float(before["yaw"]), float(after["yaw"])
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + fraction * delta


def arm_states_with_marker(arm: dict, run: str) -> list[dict]:
    """arm JSON 的 states（按 t 排序），每行带 marker（车体系黑标，x 右 y 前、原点 = J1 轴）与 marker_h（离地高）：
    臂生产代码的端点动态模型现算（arm_production.endpoint_for_arm；就地写进 arm 的 states）。生产代码算不了这一场
    （车型不在端点模型里、录包零位帧未登记、调度不认识……）抛 NothingToMeasure 带原因——不另用别的 FK 凑。"""
    import arm_production as ap
    from extract_arm_bag import car_for_session

    car, _ = car_for_session(arm)
    try:
        arm["endpoint"] = ap.endpoint_for_arm(arm, car, run)
    except ap.Unavailable as exc:
        raise NothingToMeasure(f"臂端点模型算不了本场黑标锚：{exc}") from exc
    rows = [row for row in arm.get("states", [])
            if _finite(row.get("t")) and isinstance(row.get("marker"), list) and _finite(row.get("marker_h"))]
    return sorted(rows, key=lambda row: float(row["t"]))


def _expected_world_mm(
    pc_elapsed: float,
    arm_states: list[dict],
    car_pos_rows: list[dict],
    car_yaw_rows: list[dict],
    rk_t0: float,
    rk_to_pc_bias: float,
) -> np.ndarray | None:
    """模型黑标（端点动态模型）在世界系的位置（mm）：车位姿 + R(yaw)·黑标车体系 xy，z = 车 z + 黑标离地高。"""
    rk_absolute = rk_t0 + pc_elapsed - rk_to_pc_bias
    state = _interpolate(
        arm_states, rk_absolute, "t", ("marker", "marker_h"), ARM_STATE_MAX_GAP_S
    )
    car = _interpolate(
        car_pos_rows, pc_elapsed, "elapsed_s", ("x", "y", "z"), CAR_LOC_MAX_GAP_S
    )
    yaw = _interpolate_yaw(car_yaw_rows, pc_elapsed, CAR_LOC_MAX_GAP_S)
    if state is None or car is None or yaw is None:
        return None
    local = np.asarray(state["marker"], dtype=np.float64)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    relative_world = np.asarray(
        [
            cosine * local[0] - sine * local[1],
            sine * local[0] + cosine * local[1],
            float(state["marker_h"]),
        ],
        dtype=np.float64,
    )
    return 1000.0 * (
        np.asarray([car["x"], car["y"], car["z"]], dtype=np.float64)
        + relative_world
    )


def _car_to_world(yaw: float, vector_mm: np.ndarray) -> np.ndarray:
    """车体系（x 右, y 前, z 上）向量转世界轴，与 _expected_world_mm 同一旋转。"""
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            cosine * vector_mm[0] - sine * vector_mm[1],
            sine * vector_mm[0] + cosine * vector_mm[1],
            vector_mm[2],
        ],
        dtype=np.float64,
    )


def _world_to_car(yaw: float, vector_mm: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            cosine * vector_mm[0] + sine * vector_mm[1],
            -sine * vector_mm[0] + cosine * vector_mm[1],
            vector_mm[2],
        ],
        dtype=np.float64,
    )


def model_residual_summary(observations: list[dict]) -> dict:
    """视觉黑标 − 模型黑标（端点动态模型）的车体系残差（mm，x 右 y 前 z 上）按击球侧 × 离 FinalHT 的时间窗汇总：
    帧数、3D 距离中位 / rms、各分量中位（= 模型没解释的系统量）。只汇总有模型黑标与车 yaw 的观测。"""
    groups: dict[str, dict] = {}
    for side_name, side in (("forehand", 1), ("backhand", -1), ("unknown", None)):
        for lo, hi in RESIDUAL_WINDOWS_MS:
            rows = [
                np.asarray(obs["residual_car_mm"], dtype=np.float64)
                for obs in observations
                if obs.get("residual_car_mm") is not None and obs.get("side") == side
                and lo <= float(obs["dt_ht_ms"]) < hi
            ]
            if not rows:
                continue
            values = np.asarray(rows)
            norms = np.linalg.norm(values, axis=1)
            groups[f"{side_name} [{lo:+.0f},{hi:+.0f}) ms"] = {
                "n": len(rows),
                "median_3d_mm": round(float(np.median(norms)), 1),
                "rms_3d_mm": round(float(np.sqrt(np.mean(norms ** 2))), 1),
                "median_xyz_car_mm": [round(float(v), 1) for v in np.median(values, axis=0)],
                "rms_xyz_car_mm": [round(float(v), 1) for v in np.sqrt(np.mean(values ** 2, axis=0))],
            }
    return {
        "definition": "visual black marker − arm endpoint dynamic-model marker, car frame (x right, y forward, z up), mm",
        "groups": groups,
    }


def residual_lines(residual: dict) -> list[str]:
    return [
        f"视觉 − 模型黑标 {name}: n={g['n']} 3D 中位 {g['median_3d_mm']:.1f} / rms {g['rms_3d_mm']:.1f} mm，"
        f"分量中位 ({g['median_xyz_car_mm'][0]:+.1f}, {g['median_xyz_car_mm'][1]:+.1f}, {g['median_xyz_car_mm'][2]:+.1f}) mm"
        for name, g in residual.get("groups", {}).items()
    ]


def _mean_view_dir(
    cameras: dict[str, CameraModel], serials: list[str], point_mm: np.ndarray
) -> np.ndarray:
    """各相机中心指向该点的单位视线取平均再归一：三角化深度误差主要沿它。"""
    directions = []
    for serial in serials:
        camera = cameras[serial]
        center = (-camera.R.T @ camera.t).reshape(3)
        ray = np.asarray(point_mm, dtype=np.float64) - center
        directions.append(ray / np.linalg.norm(ray))
    mean = np.mean(directions, axis=0)
    return mean / np.linalg.norm(mean)


def _traj_predict(
    fits: list[tuple[float, np.ndarray]], t: float
) -> np.ndarray | None:
    """Predict the marker position from neighbouring accepted visual fits.

    Interpolates across at most ~3 frames total, or linearly extrapolates at
    most two frames beyond the nearest same-side pair. The prediction is only a
    search anchor; geometric gates still decide acceptance.
    """
    before = [f for f in fits if f[0] < t]
    after = [f for f in fits if f[0] > t]
    if (
        before
        and after
        and t - before[-1][0] <= TRAJ_NEIGHBOR_MAX_SIDE_S
        and after[0][0] - t <= TRAJ_NEIGHBOR_MAX_SIDE_S
    ):
        a, b = before[-1], after[0]
        fraction = (t - a[0]) / (b[0] - a[0])
        return a[1] + fraction * (b[1] - a[1])
    if (
        len(before) >= 2
        and t - before[-1][0] <= TRAJ_EXTRAP_MAX_S
        and before[-1][0] - before[-2][0] <= TRAJ_NEIGHBOR_MAX_SIDE_S
    ):
        a, b = before[-2], before[-1]
        velocity = (b[1] - a[1]) / (b[0] - a[0])
        return b[1] + velocity * (t - b[0])
    if (
        len(after) >= 2
        and after[0][0] - t <= TRAJ_EXTRAP_MAX_S
        and after[1][0] - after[0][0] <= TRAJ_NEIGHBOR_MAX_SIDE_S
    ):
        a, b = after[0], after[1]
        velocity = (b[1] - a[1]) / (b[0] - a[0])
        return a[1] - velocity * (a[0] - t)
    return None


def _attempt_solve(
    candidates: dict[str, list[MarkerCandidate]],
    cameras: dict[str, CameraModel],
    serials: list[str],
    fk_expected: np.ndarray | None,
    traj_expected: np.ndarray | None,
    *,
    loo_max_mm: float | None = MARKER_MAX_LOO_MM,
    fk_gate: AnchorGate | None = None,
) -> tuple[MarkerFit, int, str, str | None] | None:
    """Acceptance ladder: 4-cam/FK, 4-cam/traj, then 3-cam per anchor.

    fk_gate（v05 校正锚点）给定时，FK 的四相机与三相机两级都用它代替 110/80 mm 球形门。
    """
    if fk_gate is None:
        fk_limits = (MARKER_MAX_EXPECTED_DISTANCE_MM, MARKER_MAX_EXPECTED_RECOVERY_MM)
    else:
        fk_limits = (math.hypot(fk_gate.max_lateral_mm, fk_gate.max_depth_mm),) * 2
    if fk_expected is not None:
        fit = _solve_marker(
            candidates, cameras, fk_expected, serials, fk_limits[0],
            loo_max_mm=loo_max_mm, gate=fk_gate,
        )
        if fit is not None:
            return fit, 4, "fk", None
    if traj_expected is not None:
        fit = _solve_marker(
            candidates, cameras, traj_expected, serials, MARKER_MAX_EXPECTED_RECOVERY_MM,
            loo_max_mm=loo_max_mm,
        )
        if fit is not None:
            return fit, 4, "traj", None
    for anchor_name, expected, limit, gate in (
        ("fk", fk_expected, fk_limits[1], fk_gate),
        ("traj", traj_expected, MARKER_MAX_EXPECTED_RECOVERY_MM, None),
    ):
        if expected is None:
            continue
        best: tuple[tuple[float, float], str, MarkerFit] | None = None
        for dropped in serials:
            subset = [serial for serial in serials if serial != dropped]
            fit = _solve_marker(
                candidates, cameras, expected, subset, limit,
                loo_max_mm=loo_max_mm, gate=gate,
            )
            if fit is None:
                continue
            anchor_rank = (
                fit.expected_distance_mm
                if gate is None
                else gate.rank(fit.point.xyz_mm - expected)
            )
            rank = (anchor_rank, fit.point.rms_px)
            if best is None or rank < best[0]:
                best = (rank, dropped, fit)
        if best is not None:
            return best[2], 3, anchor_name, best[1]
    return None


def _grid_panels(
    image: np.ndarray, serials: list[str], cameras: dict[str, CameraModel]
) -> dict[str, np.ndarray]:
    """Undo the 2x2 mosaic: each panel is one captured frame at half scale.

    The writer tiles ``frame_w//2 x frame_h//2`` copies, so a panel restores by
    exactly 2x — the same rule as ``annotate_video.scale_panel_to_full``. Do NOT
    restore to the calibration's ``image_size``: with a bottom-cropped sensor ROI
    (``camera_18.json roi_height``, OffsetY pinned to 0) the capture is shorter
    than the calibrated full frame, and stretching it back to the calibrated
    height scales every row by height_calib/height_capture. 2026-09-05 that ratio
    was 1536/1304 = 1.178, which put every projected anchor ~0.18*v px above its
    real feature (75-120 px) and left the marker untriangulable: sessions from
    20260905_201401 on measured 1-10% of frames, against 87-93% before the ROI.
    The intrinsics stay valid on the shorter frame because only bottom rows are
    dropped, so pixel coordinates are unchanged.
    """
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError(f"unexpected grid video shape {image.shape[:2]}")
    half_height, half_width = image.shape[0] // 2, image.shape[1] // 2
    frame_height, frame_width = half_height * 2, half_width * 2
    for serial in serials:
        calib_width, calib_height = cameras[serial].image_size
        if calib_width != frame_width or not 0 < frame_height <= calib_height:
            raise ValueError(
                f"grid panels restore to {frame_width}x{frame_height}, which is not "
                f"the calibrated {serial} frame {calib_width}x{calib_height} nor a "
                "bottom-cropped ROI of it"
            )
    panels: dict[str, np.ndarray] = {}
    for index, serial in enumerate(serials):
        x = (index % 2) * half_width
        y = (index // 2) * half_height
        panel = image[y : y + half_height, x : x + half_width]
        panels[serial] = cv2.resize(
            panel, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
        )
    return panels


def _measurement_context(
    tracker_path: Path,
    arm_path: Path,
    rk_path: Path,
    tables_path: Path,
) -> tuple[dict, dict, dict, list[str], list[dict]]:
    tracker = _load_json(tracker_path)
    arm = _load_json(arm_path)
    rk = _load_json(rk_path)
    tables = _load_json(tables_path)
    config = tracker.get("config") or {}
    # 锚点 = 车位姿 + R(yaw)·模型黑标（臂端点动态模型的车体系黑标，z = 它给的离地高；arm_states_with_marker）。
    # 车定位原点都是 z=0 地面点（v05 即 J1 轴）。
    car = str(arm.get("car", "")).lower()
    if car not in ("v04", "v05") or car not in Path(
        str(config.get("car_config_path", ""))
    ).stem.lower():
        raise ValueError("fixed black-marker HT measurement is supported only for V04/V05")
    serials = list(config.get("serials") or [])
    if len(serials) != 4 or len(set(serials)) != 4:
        raise ValueError("fixed black-marker HT measurement requires four unique cameras")
    if tables.get("script_error") or tables.get("tab_errors"):
        raise ValueError("report page snapshot did not complete cleanly")
    contract = tables.get("arm_contract") or {}
    if contract.get("schema") != "arm_final_ht/v4":
        raise ValueError("report tables lack arm_final_ht/v4 contract")
    if not contract.get("rows"):
        # 报告一行都没有 = 本场没有抛；对齐质量此时无从谈起，也无关紧要。
        raise NothingToMeasure("report has no throws")
    align = tables.get("align") or {}
    auto = align.get("auto") or {}
    required_points = 30 if auto.get("windowSource") == "scan" else 15
    alignment_trusted = (
        auto.get("windowSource") is not None
        and _finite(auto.get("bias"))
        and _finite(auto.get("err"))
        and float(auto["err"]) <= 0.08
        and int(auto.get("n") or 0) >= required_points
        and int(auto.get("flights") or 0) >= int(auto.get("requiredFlights") or 3)
        and not (
            auto.get("windowSource") == "scan"
            and _finite(auto.get("margin"))
            and float(auto["margin"]) < 1.35
        )
    )
    if not alignment_trusted:
        raise ValueError("report PC/RK alignment is not trusted")
    time_map = align.get("timeMap") or {}
    if not _finite(time_map.get("scale")) or abs(float(time_map["scale"]) - 1.0) > 1e-12:
        raise ValueError("report time map must use scale=1")
    if not _finite(time_map.get("bias")):
        raise ValueError("report time map has no finite bias")
    baseline_bias = float(time_map["bias"])
    phase_policy = contract.get("zPhasePolicy") or {}
    if (
        phase_policy.get("appliesTo") != "all_pc_sampling"
        or phase_policy.get("rkUse") != "global_baseline"
        or not _finite(phase_policy.get("maxAbsOffsetMs"))
        or abs(float(phase_policy["maxAbsOffsetMs"]) - 100.0) > 1e-9
    ):
        raise ValueError("report arm contract has the wrong zPhase policy")
    max_offset_s = 0.001 * float(phase_policy["maxAbsOffsetMs"])
    rk_t0 = float(rk["t0"])
    if not _finite(contract.get("rkT0")) or abs(float(contract["rkT0"]) - rk_t0) > 1e-6:
        raise ValueError("report arm contract and RK sidecar use different t0")
    targets = []
    for row in contract.get("rows", []):
        phase = row.get("zPhase") or {}
        if (
            row.get("accepted") is not True
            or row.get("finalMismatch") is True
            or not _finite(row.get("finalHtRkAbs"))
            or not _finite(row.get("finalHtPcBaselineElapsed"))
            or phase.get("usable") is not True
            or not _finite(phase.get("deltaS"))
            or not _finite(row.get("finalHtPcSampleElapsed"))
        ):
            continue
        delta_s = float(phase["deltaS"])
        if abs(delta_s) > max_offset_s + 1e-12:
            raise ValueError("visual HT zPhase exceeds the contract limit")
        expected_baseline = float(row["finalHtRkAbs"]) - rk_t0 + baseline_bias
        if abs(float(row["finalHtPcBaselineElapsed"]) - expected_baseline) > 1e-6:
            raise ValueError("visual HT baseline does not match the global PC/RK map")
        expected_sample = expected_baseline + delta_s
        if abs(float(row["finalHtPcSampleElapsed"]) - expected_sample) > 1e-6:
            raise ValueError("visual HT sample does not match baseline plus zPhase")
        targets.append(
            {
                "report_row": int(row["reportRow"]),
                "final_ht_pc_elapsed_s": expected_sample,
                "rk_to_pc_bias_s": baseline_bias + delta_s,
                # 击球侧（臂 node::final_plan_side，报告按 FinalHitPlan 的生产判决写进合同）：+1 正手 / −1 反手 / None 未知
                "side": row.get("side") if row.get("side") in (1, -1) else None,
            }
        )
    targets.sort(key=lambda row: row["final_ht_pc_elapsed_s"])
    if not targets:
        raise ValueError("report arm contract has no accepted final HT with usable zPhase")
    return tracker, arm, rk, serials, targets


def _online_candidates(frame: dict) -> dict[str, list[MarkerCandidate]]:
    """_racket_marker.json 一帧的在线候选（每台 [u, v, score, area, x, y, w, h]，全分辨率 px）→ MarkerCandidate。"""
    return {
        serial: [
            MarkerCandidate(
                uv=(float(row[0]), float(row[1])),
                score=float(row[2]),
                area=int(row[3]),
                bbox_xywh=(int(row[4]), int(row[5]), int(row[6]), int(row[7])),
            )
            for row in rows
        ]
        for serial, rows in frame["candidates"].items()
    }


def measure(
    tracker_path: Path,
    marker_path: Path,
    arm_path: Path,
    rk_path: Path,
    tables_path: Path,
) -> dict:
    started = time.perf_counter()
    (
        tracker,
        arm,
        rk,
        serials,
        targets,
    ) = _measurement_context(tracker_path, arm_path, rk_path, tables_path)
    car = str(arm["car"]).lower()
    config = tracker["config"]
    calibration_path = _session_path(config["calib_config_path"], tracker_path)
    cameras = _load_cameras(calibration_path, serials)
    tracker_t0 = float(config["first_frame_exposure_pc"])
    # 车定位线算过的帧才有在线候选（tracker.json car_localizer.sample_every_frames）
    frames = sorted(
        [
            row
            for row in _load_json(marker_path).get("frames", [])
            if isinstance(row, dict)
            and isinstance(row.get("candidates"), dict)
            and _finite(row.get("exposure_pc"))
        ],
        key=lambda row: float(row["exposure_pc"]),
    )
    video_frame_idx = {
        row["idx"]: row.get("video_frame_idx")
        for row in tracker.get("frames", [])
        if isinstance(row, dict) and "idx" in row
    }
    arm_states = arm_states_with_marker(arm, tracker_path.stem)
    car_locs = sorted(
        [row for row in tracker.get("car_locs", []) if _finite(row.get("elapsed_s"))],
        key=lambda row: float(row["elapsed_s"]),
    )
    car_pos_rows = [
        row
        for row in car_locs
        if all(_finite(row.get(key)) for key in ("x", "y", "z"))
    ]
    car_yaw_rows = [row for row in car_locs if _finite(row.get("yaw"))]
    if not frames or not arm_states or not car_pos_rows or not car_yaw_rows:
        raise ValueError(
            "session lacks online racket-marker candidates, arm states, or car locations"
        )

    rk_t0 = float(rk["t0"])
    windows: list[tuple[dict, list[dict]]] = []
    for target in targets:
        items = [
            frame
            for frame in frames
            if -PRE_WINDOW_S
            <= (
                float(frame["exposure_pc"])
                - tracker_t0
                - target["final_ht_pc_elapsed_s"]
            )
            <= POST_WINDOW_S
        ]
        if items:
            windows.append((target, items))

    print(
        f"[racket] scan {len(targets)} report rows / "
        f"{sum(len(items) for _, items in windows)} frames "
        f"(ht{-PRE_WINDOW_S:+.2f}s..{POST_WINDOW_S:+.2f}s)",
        flush=True,
    )

    jobs: list[dict] = []
    for target, target_frames in windows:
        for frame in target_frames:
            center_abs_pc = float(frame["exposure_pc"])  # tracker 帧时间就是曝光中点
            center_elapsed = center_abs_pc - tracker_t0
            jobs.append(
                {
                    "target": target,
                    "frame": frame,
                    "center_abs_pc": center_abs_pc,
                    "center_elapsed": center_elapsed,
                    "fk_expected": _expected_world_mm(
                        center_elapsed,
                        arm_states,
                        car_pos_rows,
                        car_yaw_rows,
                        rk_t0,
                        target["rk_to_pc_bias_s"],
                    ),
                    "car_yaw": _interpolate_yaw(
                        car_yaw_rows, center_elapsed, CAR_LOC_MAX_GAP_S
                    ),
                    "candidates": _online_candidates(frame),
                    "accepted": None,
                }
            )

    def accept(job: dict, solved: tuple[MarkerFit, int, str, str | None]) -> None:
        fit, n_cam, anchor_name, dropped = solved
        dt_ms = 1000.0 * (
            job["center_elapsed"] - job["target"]["final_ht_pc_elapsed_s"]
        )
        # expected_distance_mm = 距同曝光模型黑标（端点动态模型；traj 锚的解也按模型黑标算）；视觉 − 模型的车体系残差另记。
        model = job["fk_expected"]
        expected_distance = (
            float(np.linalg.norm(fit.point.xyz_mm - model)) if model is not None else None
        )
        residual_car = (
            _world_to_car(job["car_yaw"], fit.point.xyz_mm - model)
            if model is not None and job["car_yaw"] is not None else None
        )
        observation = {
            "x": float(fit.point.xyz_mm[0] / 1000.0),
            "y": float(fit.point.xyz_mm[1] / 1000.0),
            "z": float(fit.point.xyz_mm[2] / 1000.0),
            "t": job["center_abs_pc"],
            "elapsed_s": job["center_elapsed"],
            "frame_idx": int(job["frame"]["frame_idx"]),
            "video_frame_idx": video_frame_idx.get(job["frame"]["frame_idx"]),
            "report_row": job["target"]["report_row"],
            "final_ht_pc_elapsed_s": job["target"]["final_ht_pc_elapsed_s"],
            "dt_ht_ms": dt_ms,
            "ht_side": "before" if dt_ms < 0.0 else "after",
            "n_cam": n_cam,
            "anchor": anchor_name,
            "reproj_err": fit.point.rms_px,
            "reproj_max_px": fit.point.max_px,
            "loo_max_mm": max(fit.loo_delta_mm.values()),
            "heldout_max_px": max(fit.loo_heldout_px.values()),
            "expected_distance_mm": expected_distance,
            "black_marker": True,
            "side": job["target"]["side"],
        }
        if model is not None:
            observation["model_marker_world"] = [round(float(v) / 1000.0, 5) for v in model]
        if residual_car is not None:
            observation["residual_car_mm"] = [round(float(v), 2) for v in residual_car]
        if dropped is not None:
            observation["dropped_serial"] = dropped
        job["accepted"] = observation

    def row_fits(report_row: int) -> list[tuple[float, np.ndarray]]:
        fits = [
            (
                job["center_elapsed"],
                np.asarray(
                    [
                        job["accepted"]["x"],
                        job["accepted"]["y"],
                        job["accepted"]["z"],
                    ],
                    dtype=np.float64,
                )
                * 1000.0,
            )
            for job in jobs
            if job["accepted"] is not None
            and job["target"]["report_row"] == report_row
        ]
        return sorted(fits, key=lambda item: item[0])

    # Pass 1: solve every frame that has a model anchor (端点动态模型黑标；深度 / 横向分开设门，不加本场偏差).
    anchored = [job for job in jobs if job["fk_expected"] is not None]
    for job in anchored:
        gate = AnchorGate(
            _mean_view_dir(cameras, serials, job["fk_expected"]),
            ANCHOR_MAX_LATERAL_MM,
            ANCHOR_MAX_DEPTH_MM,
        )
        solved = _attempt_solve(
            job["candidates"], cameras, serials, job["fk_expected"], None,
            loo_max_mm=MARKER_LOO_MAX_MM, fk_gate=gate,
        )
        if solved is not None:
            accept(job, solved)

    # Pass 2: chain neighbour fits as trajectory anchors. The FK ladder already
    # ran in pass 1, so pass 2 solves against the trajectory anchor only, and
    # skips a frame whose anchor has not moved since its last attempt.
    for _ in range(6):
        progress = False
        for job in jobs:
            if job["accepted"] is not None:
                continue
            traj = _traj_predict(
                row_fits(job["target"]["report_row"]), job["center_elapsed"]
            )
            if traj is None:
                continue
            tried = job.get("traj_tried_mm")
            if tried is not None and float(np.linalg.norm(traj - tried)) < 2.0:
                continue
            job["traj_tried_mm"] = traj
            solved = _attempt_solve(
                job["candidates"], cameras, serials, None, traj,
                loo_max_mm=MARKER_LOO_MAX_MM,
            )
            if solved is not None:
                accept(job, solved)
                progress = True
        if not progress:
            break

    observations = [job["accepted"] for job in jobs if job["accepted"] is not None]
    for target, target_frames in windows:
        row_accepted = sum(
            1
            for job in jobs
            if job["accepted"] is not None and job["target"] is target
        )
        print(
            f"[racket] row {target['report_row']}: "
            f"{row_accepted}/{len(target_frames)} frames",
            flush=True,
        )

    observations.sort(key=lambda row: (row["t"], row["report_row"]))
    rows_with_observations = {row["report_row"] for row in observations}
    elapsed = time.perf_counter() - started
    print(
        f"[racket] fixed black marker: {len(observations)}/{len(jobs)} frames, "
        f"{len(rows_with_observations)}/{len(targets)} report rows, {elapsed:.1f}s"
    )
    n_traj = sum(1 for row in observations if row.get("anchor") == "traj")
    n_3cam = sum(1 for row in observations if row.get("n_cam") == 3)
    residual = model_residual_summary(observations)
    for line in residual_lines(residual):
        print(f"[racket] {line}", flush=True)
    return {
        "config": {
            "measurement": "fixed black marker center (V04/V05, TCP calibrated onto the marker)",
            "timing": "tracker frame time (exposure_pc = four-camera exposure center)",
            "selection": (
                "tracker online full-resolution candidates (car-loc frames) from report "
                f"raw final HT {-PRE_WINDOW_S:g}s to {POST_WINDOW_S:+g}s "
                f"(per-throw zPhase applied); max reprojection {MARKER_MAX_REPROJ_PX:g} px; "
                "no leave-one-out mm gate; "
                f"held-out <= {MARKER_MAX_HELDOUT_PX:g} px; "
                "anchor = arm endpoint dynamic-model marker (production arm_endpoint, no session offset), "
                f"gated lateral <= {ANCHOR_MAX_LATERAL_MM:g} mm / view depth <= {ANCHOR_MAX_DEPTH_MM:g} mm; "
                "neighbour-trajectory-anchored recoveries <= "
                f"{MARKER_MAX_EXPECTED_RECOVERY_MM:g} mm (rows carry n_cam/anchor"
                "/dropped_serial provenance)"
            ),
            "car": car,
            "model_anchor": arm.get("endpoint"),
            "model_residual": residual,
            "coordinate": (
                "multi-camera marker world position; report subtracts visual car "
                "center on field world axes"
            ),
        },
        "summary": {
            "racket_observations_3d": len(observations),
            "racket_frames_processed": len(jobs),
            "racket_min_cams": 3,
            "racket_obs_traj_anchor": n_traj,
            "racket_obs_3cam": n_3cam,
            "report_rows_scanned": len(targets),
            "report_rows_with_observations": len(rows_with_observations),
        },
        "racket_observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="tracker session JSON")
    parser.add_argument(
        "--marker-json", type=Path, required=True,
        help="tracker online black-marker candidates (<run>_racket_marker.json)",
    )
    parser.add_argument("--arm-json", type=Path, required=True)
    parser.add_argument("--rk-tracking-json", type=Path, required=True)
    parser.add_argument("--tables-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = measure(
            args.input,
            args.marker_json,
            args.arm_json,
            args.rk_tracking_json,
            args.tables_json,
        )
    except NothingToMeasure as exc:
        # 不写 --output：调用方（run_tracker 后处理）据此知道本场没有视觉拍心可合入报告。
        print(f"[racket] skip fixed black marker: {exc}")
        return 0
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[racket] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
