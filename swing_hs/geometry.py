# -*- coding: utf-8 -*-
"""swing_hs 的几何：车体系↔世界、投影、黑标包络、AprilTag 角点、ROI 规划、帧率上限。

纯函数，不碰相机。相机模型 / 投影 / 候选检测与收场黑标工具共用一份实现
（test_src/racket_ht_black_marker.py），FK 用标准运动学（test_src/extract_arm_bag.fk →
tennis-man/arm_controller 的 compact_arm_kinematics + 车型 yaml）。
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEST_SRC = REPO / "test_src"

SENSOR_W, SENSOR_H = 2048, 1536
#: 帧率上限 = min(单台 1GbE 117.5 MB/s ÷ 像素数, 1/(行数 × 8.0 µs + 0.94 ms))，
#: 0906 / 1006 两次在 CS032-60GC 上实测同一组常数（tennis-man memory/pc-cam-120fps-marker-roi-1006.md）
LINK_BYTES_PER_S = 117.5e6
SENSOR_ROW_S = 8.0e-6
SENSOR_FIXED_S = 0.94e-3
#: ROI 对齐：OffsetX / Width 按 16（相机拒过 998），OffsetY / Height 按 4；偏移都是偶数，Bayer 相位不变
ROI_X_STEP = 16
ROI_Y_STEP = 4


def load_test_src(name: str):
    """按路径加载 test_src 下的模块（与 src/racket_marker_online 同名同法，sys.modules 里只有一份）。"""
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, TEST_SRC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


# ───────────────── 帧率 / ROI ─────────────────


def link_max_fps(width: int, height: int) -> float:
    return LINK_BYTES_PER_S / float(width * height)


def sensor_max_fps(height: int) -> float:
    return 1.0 / (height * SENSOR_ROW_S + SENSOR_FIXED_S)


def predicted_max_fps(width: int, height: int) -> float:
    """开场预判（真值以相机读回的 ResultingFrameRate 为准）。"""
    return min(link_max_fps(width, height), sensor_max_fps(height))


def fit_roi(u0: float, v0: float, u1: float, v1: float, margin: float) -> list[int]:
    """像素框外扩 margin → [offset_x, offset_y, width, height]，按相机步进对齐、夹在传感器内。"""
    u0, v0 = max(0.0, u0 - margin), max(0.0, v0 - margin)
    u1, v1 = min(float(SENSOR_W), u1 + margin), min(float(SENSOR_H), v1 + margin)
    if u1 <= u0 or v1 <= v0:
        raise ValueError(f"空 ROI：u {u0:.0f}..{u1:.0f} v {v0:.0f}..{v1:.0f}")
    ox = int(math.floor(u0 / ROI_X_STEP)) * ROI_X_STEP
    oy = int(math.floor(v0 / ROI_Y_STEP)) * ROI_Y_STEP
    w = int(math.ceil((u1 - ox) / ROI_X_STEP)) * ROI_X_STEP
    h = int(math.ceil((v1 - oy) / ROI_Y_STEP)) * ROI_Y_STEP
    w, h = min(w, SENSOR_W), min(h, SENSOR_H)
    ox, oy = min(ox, SENSOR_W - w), min(oy, SENSOR_H - h)
    return [ox, oy, w, h]


def roi_contains(roi: list[int], uv: np.ndarray, pad: float = 0.0) -> np.ndarray:
    ox, oy, w, h = roi
    uv = np.atleast_2d(uv)
    return (
        (uv[:, 0] >= ox + pad) & (uv[:, 0] < ox + w - pad)
        & (uv[:, 1] >= oy + pad) & (uv[:, 1] < oy + h - pad)
    )


def in_sensor(uv: np.ndarray) -> np.ndarray:
    return roi_contains([0, 0, SENSOR_W, SENSOR_H], uv)


# ───────────────── 车体系 / 世界 / 投影 ─────────────────


def car_to_world(points_car_m: np.ndarray, pose: tuple[float, float, float], z_offset_m: float = 0.0) -> np.ndarray:
    """车体系（x 右、y 前、z 上，原点 = 车心地面 = J1 轴）→ 世界系（m）。

    pose = (x, y, yaw)；z_offset_m = 臂 FK 的 z 零点偏移（黑标传 car_hit_pos_z_offset，tag 传 0）。
    与收场 racket_ht_black_marker._expected_world_mm、在线 racket_marker_online 同一换算。
    """
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.atleast_2d(np.asarray(points_car_m, dtype=np.float64))
    return np.column_stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1], p[:, 2] - z_offset_m])


def project_m(camera, world_m: np.ndarray) -> np.ndarray:
    """世界点（m）→ 该相机全幅原始（带畸变）像素 (N, 2)。"""
    pts = 1000.0 * np.asarray(world_m, dtype=np.float64).reshape(-1, 3)
    uv, _ = cv2.projectPoints(pts, camera.rvec, camera.t, camera.K, camera.D)
    return uv.reshape(-1, 2)


def depth_mm(camera, world_m: np.ndarray) -> np.ndarray:
    pts = 1000.0 * np.asarray(world_m, dtype=np.float64).reshape(-1, 3)
    return (camera.R @ pts.T + camera.t).T[:, 2]


# ───────────────── 黑标包络 ─────────────────


def load_envelope(path: Path, side: str, j1_range_deg: tuple[float, float]) -> np.ndarray:
    """实打挥拍的关节采样（控制器 6 槽，度）→ J1 落在 j1_range_deg 内的那些（弧度）。

    envelope_<car>.json 由 swing_hs/build_envelope.py 从场次 _arm.json 生成：fore / back 各一组。
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = np.asarray(doc[side], dtype=np.float64)
    lo, hi = sorted(j1_range_deg)
    keep = rows[(rows[:, 0] >= lo) & (rows[:, 0] <= hi)]
    if len(keep) == 0:
        raise ValueError(f"{path} 的 {side} 没有 J1 在 [{lo:g}, {hi:g}]° 的采样")
    return np.radians(keep)


def marker_points_car(q_rad: np.ndarray, car: str) -> np.ndarray:
    """各关节采样的黑标位置（车体系 m，z 为 FK 自己的零点，世界 z 还要减 car_hit_pos_z_offset）。"""
    eab = load_test_src("extract_arm_bag")
    return np.asarray([eab.fk(q, car=car)["marker"] for q in q_rad], dtype=np.float64)


def marker_z_offset_m(car: str) -> float:
    eab = load_test_src("extract_arm_bag")
    return float(eab.standard_kinematics(car).car_hit_pos_z_offset(car))


def inflate_hull(points: np.ndarray, radius: float) -> np.ndarray:
    """点云每点外扩 ±radius 立方体后的凸包顶点（投影后的包围框只由凸包顶点决定）。"""
    from scipy.spatial import ConvexHull

    offs = np.asarray(list(itertools.product((-radius, radius), repeat=3)), dtype=np.float64)
    cloud = (points[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    try:
        return cloud[ConvexHull(cloud).vertices]
    except Exception:  # 退化（点太少 / 共面）：直接用全部点
        return cloud


# ───────────────── AprilTag ─────────────────


def load_vehicle_reference(path: Path) -> dict:
    """车型 tag 布局（src/config/vehicle_<car>.json 的 vehicle_reference）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))["vehicle_reference"]


def tag_corners_car(vehicle_reference: dict, scale: float = 1.0) -> dict[int, np.ndarray]:
    """各 tag 黑边四角（车体系 m）；scale > 1 外扩（1.5 ≈ 静区 + ArUco 余量）。

    角点顺序 = ArUco 左上、右上、右下、左下（印面坐标），同 src/car_localizer。
    """
    half = 0.5 * float(vehicle_reference["apriltag_black_edge_m"]) * scale
    out: dict[int, np.ndarray] = {}
    for key, spec in vehicle_reference["apriltags"].items():
        center = np.asarray(spec["center_car_m"], dtype=np.float64)
        R = np.asarray(spec["R_tag_car"], dtype=np.float64)
        out[int(key)] = np.asarray(
            [center + u * R[:, 0] + v * R[:, 1] for u, v in ((-half, half), (half, half), (half, -half), (-half, -half))]
        )
    return out


def tag_view_angle_deg(vehicle_reference: dict, tag_id: int, pose, camera_center_m: np.ndarray) -> float:
    """tag 朝外法线与「tag → 相机」视线的夹角（度）；> ~70° 基本认不出。"""
    spec = vehicle_reference["apriltags"][str(tag_id)]
    center = car_to_world(np.asarray(spec["center_car_m"]), pose)[0]
    normal_car = np.asarray(spec["R_tag_car"], dtype=np.float64)[:, 2]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    normal = np.array([c * normal_car[0] - s * normal_car[1], s * normal_car[0] + c * normal_car[1], normal_car[2]])
    ray = np.asarray(camera_center_m, dtype=np.float64) - center
    return math.degrees(math.acos(float(np.clip(normal @ ray / np.linalg.norm(ray), -1.0, 1.0))))


def camera_center_m(camera) -> np.ndarray:
    return (-camera.R.T @ camera.t).reshape(3) / 1000.0


# ───────────────── ROI 规划 ─────────────────


@dataclass
class RoiMargins:
    marker_m: float = 0.05      # 黑标 3D 外扩：FK 与真黑标差（触球时真黑标比 FK 低几 cm）+ 臂柔度
    car_push_m: float = 0.04    # 车没锁会被挥拍推（1006–1007 专场单场 3–11 cm）
    car_push_deg: float = 3.0
    pixel: float = 20.0
    tag_scale: float = 1.5


@dataclass
class CameraPlan:
    serial: str
    roi: list[int]
    marker_bbox: list[float] | None          # 黑标包络（外扩 + 扰动后）的像素框，夹在传感器内
    marker_in_sensor: float                  # 名义车位、不外扩的黑标点落在传感器内的比例
    marker_in_roi: float                     # …落在 ROI 内（离边 ≥ 10 px）的比例
    tags_in_roi: dict[int, bool]             # 名义车位下 tag 黑边四角都在 ROI 内
    tag_side_px: dict[int, float]
    tag_view_deg: dict[int, float]
    tags_dropped: bool = False               # 带 tag 超预算、只留黑标
    link_fps: float = 0.0
    sensor_fps: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def max_fps(self) -> float:
        return min(self.link_fps, self.sensor_fps)

    @property
    def megapixels(self) -> float:
        return self.roi[2] * self.roi[3] / 1e6

    def to_json(self) -> dict:
        return {
            "serial": self.serial,
            "roi": self.roi,
            "megapixels": round(self.megapixels, 3),
            "predicted_max_fps": round(self.max_fps, 2),
            "link_fps": round(self.link_fps, 2),
            "sensor_fps": round(self.sensor_fps, 2),
            "marker_bbox": None if self.marker_bbox is None else [round(v, 1) for v in self.marker_bbox],
            "marker_in_sensor": round(self.marker_in_sensor, 4),
            "marker_in_roi": round(self.marker_in_roi, 4),
            "tags_in_roi": {str(k): v for k, v in self.tags_in_roi.items()},
            "tag_side_px": {str(k): round(v, 1) for k, v in self.tag_side_px.items()},
            "tag_view_deg": {str(k): round(v, 1) for k, v in self.tag_view_deg.items()},
            "tags_dropped": self.tags_dropped,
            "notes": self.notes,
        }


def _perturbations(margins: RoiMargins) -> list[tuple[float, float, float]]:
    p, r = margins.car_push_m, math.radians(margins.car_push_deg)
    return [(dx, dy, dr) for dx in (-p, 0.0, p) for dy in (-p, 0.0, p) for dr in (-r, 0.0, r)]


def _bbox(uv: np.ndarray) -> list[float] | None:
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if len(uv) == 0:
        return None
    return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]


def _clip_bbox(box: list[float] | None) -> list[float] | None:
    if box is None:
        return None
    u0, v0, u1, v1 = box
    u0, v0 = max(0.0, u0), max(0.0, v0)
    u1, v1 = min(float(SENSOR_W), u1), min(float(SENSOR_H), v1)
    if u1 <= u0 or v1 <= v0:
        return None
    return [u0, v0, u1, v1]


def _union(*boxes: list[float] | None) -> list[float] | None:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def plan_rois(
    cameras: dict,
    pose: tuple[float, float, float],
    marker_car_m: np.ndarray,
    marker_z_offset: float,
    vehicle_reference: dict,
    margins: RoiMargins,
    *,
    fps: float,
    fps_headroom: float = 1.0,
) -> dict[str, CameraPlan]:
    """每台相机一个 ROI = 黑标包络 ∪ 两块 tag，外扩 + 车位扰动并集，按步进对齐。

    带 tag 超 100 fps 预算时这台只留黑标（tags_dropped），车定位靠其余相机；黑标单独都超预算的
    相机 roi 照给、由调用方拒绝开录。
    """
    hull = inflate_hull(marker_car_m, margins.marker_m)
    perts = _perturbations(margins)
    tags_big = tag_corners_car(vehicle_reference, margins.tag_scale)
    tags_true = tag_corners_car(vehicle_reference, 1.0)
    plans: dict[str, CameraPlan] = {}
    for serial, camera in cameras.items():
        marker_uv = []
        for dx, dy, dr in perts:
            p = (pose[0] + dx, pose[1] + dy, pose[2] + dr)
            world = car_to_world(hull, p, marker_z_offset)
            uv = project_m(camera, world)
            uv[depth_mm(camera, world) <= 0.0] = np.nan
            marker_uv.append(uv)
        marker_box = _clip_bbox(_bbox(np.vstack(marker_uv)))
        tag_box = None
        for corners in tags_big.values():
            uv = np.vstack([
                project_m(camera, car_to_world(corners, (pose[0] + dx, pose[1] + dy, pose[2] + dr)))
                for dx, dy, dr in perts
            ])
            tag_box = _union(tag_box, _clip_bbox(_bbox(uv)))
        nominal = project_m(camera, car_to_world(marker_car_m, pose, marker_z_offset))
        centre = camera_center_m(camera)
        tag_uv_true = {tid: project_m(camera, car_to_world(c, pose)) for tid, c in tags_true.items()}
        tag_side = {tid: float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)))
                    for tid, uv in tag_uv_true.items()}
        view = {tid: tag_view_angle_deg(vehicle_reference, tid, pose, centre) for tid in tags_true}

        def make(box: list[float] | None, dropped: bool) -> CameraPlan | None:
            if box is None:
                return None
            roi = fit_roi(*box, margins.pixel)
            plan = CameraPlan(
                serial=serial,
                roi=roi,
                marker_bbox=marker_box,
                marker_in_sensor=float(np.mean(in_sensor(nominal))),
                marker_in_roi=float(np.mean(roi_contains(roi, nominal, pad=10.0))),
                tags_in_roi={tid: bool(np.all(roi_contains(roi, uv))) for tid, uv in tag_uv_true.items()},
                tag_side_px=tag_side,
                tag_view_deg=view,
                tags_dropped=dropped,
                link_fps=link_max_fps(roi[2], roi[3]),
                sensor_fps=sensor_max_fps(roi[3]),
            )
            return plan

        plan = make(_union(marker_box, tag_box), False)
        if plan is None:
            raise ValueError(f"{serial}: 黑标包络和 tag 都不在画面里（车位 {pose}）")
        if plan.max_fps < fps + fps_headroom and marker_box is not None:
            alone = make(marker_box, True)
            if alone is not None and alone.max_fps >= fps + fps_headroom:
                alone.notes.append(f"带 tag 的 ROI {plan.roi[2]}x{plan.roi[3]} 只到 {plan.max_fps:.1f} fps，这台只框黑标")
                plan = alone
        if plan.marker_in_sensor < 0.999:
            plan.notes.append(f"黑标包络只有 {100 * plan.marker_in_sensor:.0f}% 在这台画面内（车离相机太近或太偏）")
        for tid, ok in plan.tags_in_roi.items():
            if not ok and not plan.tags_dropped:
                plan.notes.append(f"tag{tid} 不完整在 ROI 内（出画面或被裁）")
        plans[serial] = plan
    return plans


def plan_problems(plans: dict[str, CameraPlan], fps: float, fps_headroom: float = 1.0) -> list[str]:
    """不能开录的硬问题：帧率不够、能同时看到两块 tag 的相机不足 2 台、能看到黑标的相机不足 2 台。"""
    problems = []
    for serial, plan in plans.items():
        if plan.max_fps < fps + fps_headroom:
            problems.append(
                f"{serial} ROI {plan.roi[2]}x{plan.roi[3]} 预判上限 {plan.max_fps:.1f} fps < {fps + fps_headroom:g}")
    both = [s for s, p in plans.items() if not p.tags_dropped and all(p.tags_in_roi.values())]
    if len(both) < 2:
        problems.append(f"同时框进两块 tag 的相机只有 {len(both)} 台（车 yaw 必须双 tag，至少 2 台）")
    seeing = [s for s, p in plans.items() if p.marker_in_roi > 0.5]
    if len(seeing) < 2:
        problems.append(f"看得到大半黑标包络的相机只有 {len(seeing)} 台")
    return problems
