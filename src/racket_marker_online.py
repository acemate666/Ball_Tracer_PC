# -*- coding: utf-8 -*-
"""拍上黑标在线识别（全分辨率）。

车定位线每算出一帧车位姿，就把这一帧和前面没做车定位的帧（车位姿按时间插值）一起处理：取各帧曝光
时刻的关节角做 FK，得黑标的大致世界位置，投到四台相机，只在全分辨率图上截一小块找黑方块候选，
每帧都有（40 fps），收尾写进 <run>_racket_marker.json。三角化、选点和本场
「黑标 − FK」偏差校正留给收场的 test_src/racket_ht_black_marker.py（用报告对时后的精确 FK 锚）。

候选检测与收场工具是同一个函数（racket_ht_black_marker._marker_candidates），FK 是同一份标准运动学
（extract_arm_bag.fk → tennis-man/arm_controller 的 compact_arm_kinematics + 车型 yaml）。
"""

from __future__ import annotations

import bisect
import importlib.util
import math
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

_TEST_SRC = Path(__file__).resolve().parent.parent / "test_src"

# 截块半径（全分辨率 px；候选离锚点 ≤ 半径 − 10）。1010 用 113329 回放定：在线锚离真黑标每台
# p50 7.6 / p95 16.4 / p99 28.7 / 最大 47 px（主要是本场「黑标 − FK」常偏 ~20 mm）；收场解出帧数随半径
# 96/64/55/48 px = 334/364/390/402（原离线 40 fps 录像大范围搜索在同一批帧上 390）——块越大，自适应阈值
# 被背景稀释、杂点越多，反而认不出。
CROP_RADIUS_PX = 48
# /joint_states 200 Hz，留 2 s
JOINT_BUFFER_S = 2.0
# 前后两条关节角相隔超过这么久就不插值（掉包 / 臂没在发）
JOINT_MAX_GAP_S = 0.05
# 曝光中点比最新一条关节角还晚时最多沿用这么久（车定位线处理到这一帧时，之后的关节角通常早到了）
JOINT_MAX_HOLD_S = 0.01
# RK→PC：这条流自己的 (PC 收到时刻 − header.stamp) 取低分位 ≈ 两轴偏移 + 最小网络延时
JOINT_OFFSET_PERCENTILE = 10.0
# 本帧车定位失败 / 单 tag 没有 yaw 时，沿用最近一次位置 / yaw 的最长时长
CAR_POSE_MAX_AGE_S = 0.25


def _load_test_src(name: str):
    """按路径加载 test_src 下的模块：在线识别与收场工具共用一份实现。"""
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, _TEST_SRC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


class JointStateBuffer:
    """/joint_states 缓存：ROS 回调线程写，车定位线按曝光时刻（PC 轴）取关节角。

    关节角按控制器 6 个槽排（五轴臂槽 4 恒 0），槽名来自车型 yaml，与 extract_arm_bag 同一规则。
    """

    def __init__(self, car: str) -> None:
        # 第一次取 CAR_MODELS[car] 就加载标准运动学：缺 arm_controller checkout 开场直接报错
        self._slot_names = _load_test_src("extract_arm_bag").CAR_MODELS[car].slot_joint_names
        self._expected_names = sorted(name for name in self._slot_names if name is not None)
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque()
        self._offsets: deque[float] = deque()
        self._positions: deque[np.ndarray] = deque()
        # 收到的关节名不是本车型那一组（车型选错）：整场不出黑标，summary 里写出收到的名字
        self.name_mismatch: list[str] | None = None

    def add(self, stamp_rk: float, names: list[str], positions: list[float], recv_pc: float) -> None:
        if sorted(names) != self._expected_names:
            self.name_mismatch = names
            return
        by_name = dict(zip(names, positions))
        q = np.asarray(
            [0.0 if name is None else by_name[name] for name in self._slot_names],
            dtype=np.float64,
        )
        with self._lock:
            if self._stamps and stamp_rk <= self._stamps[-1]:
                return
            self._stamps.append(stamp_rk)
            self._offsets.append(recv_pc - stamp_rk)
            self._positions.append(q)
            while stamp_rk - self._stamps[0] > JOINT_BUFFER_S:
                self._stamps.popleft()
                self._offsets.popleft()
                self._positions.popleft()

    def at_pc(self, pc_t: float) -> np.ndarray | None:
        """PC 轴 pc_t 时刻的关节角（前后两条线性插值）；缓存里没有这一刻返回 None。"""
        with self._lock:
            if len(self._stamps) < 20:
                return None
            stamps = list(self._stamps)
            positions = list(self._positions)
            offset = float(np.percentile(self._offsets, JOINT_OFFSET_PERCENTILE))
        rk_t = pc_t - offset
        right = bisect.bisect_left(stamps, rk_t)
        if right == len(stamps):
            return positions[-1] if rk_t - stamps[-1] <= JOINT_MAX_HOLD_S else None
        if right == 0 or stamps[right] - stamps[right - 1] > JOINT_MAX_GAP_S:
            return None
        fraction = (rk_t - stamps[right - 1]) / (stamps[right] - stamps[right - 1])
        return positions[right - 1] + fraction * (positions[right] - positions[right - 1])


def _sample_at(before, after, t: float, *, wrap: bool = False) -> list[float] | None:
    """(t, *values) 两个车位姿样本在 t 时刻的值：t 夹在中间就按时间线性插值（wrap=True 按角度走短弧），
    否则取离 t 近的那个；离 t 超过 CAR_POSE_MAX_AGE_S 或都没有返回 None。"""
    if (
        before is not None
        and after is not None
        and before[0] < t < after[0]
        and after[0] - before[0] <= CAR_POSE_MAX_AGE_S
    ):
        fraction = (t - before[0]) / (after[0] - before[0])
        return [
            a + fraction * (math.atan2(math.sin(b - a), math.cos(b - a)) if wrap else b - a)
            for a, b in zip(before[1:], after[1:])
        ]
    nearest = min(
        (sample for sample in (before, after) if sample is not None),
        key=lambda sample: abs(t - sample[0]),
        default=None,
    )
    if nearest is None or abs(t - nearest[0]) > CAR_POSE_MAX_AGE_S:
        return None
    return list(nearest[1:])


class RacketMarkerLocator:
    """车定位线里每个车定位帧调用一次，把它和前面没做车定位的帧一起截块（黑标 40 fps）：
    车位姿 + 同曝光时刻关节角 → FK 黑标 → 各相机全分辨率截块候选。"""

    def __init__(
        self,
        *,
        car: str,
        calib_path: str,
        serials: list[str],
        joint_states: JointStateBuffer,
    ) -> None:
        self._ht = _load_test_src("racket_ht_black_marker")
        eab = _load_test_src("extract_arm_bag")
        self._car = car
        self._fk = eab.fk
        self._z_offset_m = float(eab.standard_kinematics(car).car_hit_pos_z_offset(car))
        self._cameras = self._ht._load_cameras(Path(calib_path), serials)
        self._joints = joint_states
        self._last_xyz: tuple[float, float, float, float] | None = None  # (t, x, y, z)
        self._last_yaw: tuple[float, float] | None = None  # (t, yaw)
        self.stats = {"frames": 0, "no_car_pose": 0, "no_joint_state": 0, "with_candidates": 0}

    def process(self, loc, frames: list[tuple[int, float, dict]]) -> list[dict]:
        """frames = 上一个车定位帧之后到本车定位帧的各帧 (frame_idx, exposure_pc, 全分辨率图)，按时间排、
        最后一帧就是本车定位帧；loc = 它的 CarLoc（None = 没定到车）。

        没做车定位的帧，车位姿在上一个和这一个车定位位姿之间按曝光时刻插值（车慢、黑标快：拿上一帧黑标
        位置截块，113329 有 25% 的帧截不到，插值车位姿 + 本帧关节角只有 0.5%）；关节角各取各的曝光时刻。
        返回各帧 {frame_idx, exposure_pc, anchor_mm, candidates}；缺车位姿 / 关节角的帧不出。
        exposure_pc 是 tracker 帧时间（各相机曝光中点的均值），直接当曝光时刻用。
        """
        t_now = frames[-1][1]
        before_xyz, before_yaw = self._last_xyz, self._last_yaw
        if loc is not None:
            self._last_xyz = (t_now, loc.x, loc.y, loc.z)
            if loc.yaw is not None:
                self._last_yaw = (t_now, loc.yaw)
        rows = []
        for frame_idx, exposure_pc, images in frames:
            self.stats["frames"] += 1
            t = exposure_pc
            xyz = _sample_at(before_xyz, self._last_xyz, t)
            yaw = _sample_at(before_yaw, self._last_yaw, t, wrap=True)
            if xyz is None or yaw is None:
                self.stats["no_car_pose"] += 1
                continue
            q = self._joints.at_pc(t)
            if q is None:
                self.stats["no_joint_state"] += 1
                continue
            # 与收场 _expected_world_mm 同一换算：车位姿 + R(yaw)·黑标 FK，z 减臂基座高度偏移
            marker = self._fk(q, car=self._car)["marker"]
            x, y, z = xyz
            cosine, sine = math.cos(yaw[0]), math.sin(yaw[0])
            anchor_mm = 1000.0 * np.asarray(
                [
                    x + cosine * marker[0] - sine * marker[1],
                    y + sine * marker[0] + cosine * marker[1],
                    z + marker[2] - self._z_offset_m,
                ],
                dtype=np.float64,
            )
            candidates: dict[str, list[list]] = {}
            for serial, camera in self._cameras.items():
                image = images.get(serial)
                if image is None:
                    continue
                found = self._ht._marker_candidates(
                    image,
                    tuple(self._ht._project_raw(camera, anchor_mm)),
                    local_radius=CROP_RADIUS_PX,
                    broad_radius=None,
                )
                candidates[serial] = [
                    [round(c.uv[0], 2), round(c.uv[1], 2), round(c.score, 1), c.area, *c.bbox_xywh]
                    for c in found
                ]
            if any(candidates.values()):
                self.stats["with_candidates"] += 1
            rows.append({
                "frame_idx": frame_idx,
                "exposure_pc": exposure_pc,
                "anchor_mm": [round(float(v), 1) for v in anchor_mm],
                "candidates": candidates,
            })
        return rows
