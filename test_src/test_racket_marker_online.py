"""黑标在线候选的核心 contract：按曝光时刻取到对应的关节角；锚（臂生产代码的端点模型黑标）附近的真实黑标被 YOLO 认出来；
收场在这些候选里解出三维、给出「视觉 − 端点动态模型黑标」残差。要臂 checkout（TENNIS_MAN_ARM_CONTROLLER）与 g++，缺了整文件 skip。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import arm_production as ap
import racket_ht_black_marker as ht
from src.racket_marker_online import JointStateBuffer, RacketMarkerLocator

try:
    ap.endpoint("v05")
except ap.Unavailable as _exc:  # pragma: no cover —— 没有臂 checkout 的机器
    pytest.skip(f"臂生产代码不可用：{_exc}", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "src" / "config" / "four_camera_calib_18.json"
SERIALS = list(json.loads(CALIB.read_text(encoding="utf-8"))["cameras"])
# 1010 113329 场 ready 位的一条实测关节角（槽 4 是五轴臂的空槽）
READY_Q = [-1.56487, -0.15568, 2.48771, 2.42524, 0.0, 0.05177]
# 车在这里时 ready 位黑标四台相机都看得到
CAR_POSE = SimpleNamespace(x=-0.3, y=1.5, z=0.0, yaw=0.02)
PC_MINUS_RK = 50.0
#: 真实录像里人工核对过的黑标（swing_hs 单测同一份：014335 夜场 2 张、115405 白天 2 张，原始 Bayer 160×160 块 + 人标框）
MARKER_SAMPLES = ROOT / "test_src" / "fixtures" / "swing_hs_marker_samples.npz"


def _feed(buffer: JointStateBuffer, q_of_t, *, latency_s=lambda k: 0.002) -> None:
    """200 Hz 的 /joint_states：RK 时刻 100.000…101.995 s，关节名倒序发（按名字取槽）。"""
    slots = ap.slot_names("v05")
    names = [name for name in reversed(slots) if name is not None]
    for k in range(400):
        stamp = 100.0 + 0.005 * k
        by_slot = dict(zip(slots, q_of_t(stamp)))
        buffer.add(
            stamp, names, [by_slot[name] for name in names], stamp + PC_MINUS_RK + latency_s(k)
        )


def test_joint_state_buffer_reads_the_exposure_time_on_the_rk_axis():
    buffer = JointStateBuffer("v05")
    # 每 10 条有一条网络晚到 30 ms：换算取低分位，不被它带偏
    _feed(
        buffer,
        lambda t: [t, 2 * t, 3 * t, 4 * t, 0.0, -t],
        latency_s=lambda k: 0.030 if k % 10 == 0 else 0.002,
    )
    q = buffer.at_pc(101.2345 + PC_MINUS_RK + 0.002)
    assert q == pytest.approx([101.2345, 202.469, 303.7035, 404.938, 0.0, -101.2345], abs=1e-6)
    assert buffer.at_pc(103.0 + PC_MINUS_RK) is None


def _locator(q_of_t) -> RacketMarkerLocator:
    buffer = JointStateBuffer("v05")
    _feed(buffer, q_of_t)
    return RacketMarkerLocator(
        car="v05", calib_path=str(CALIB), serials=SERIALS, joint_states=buffer
    )


def test_locator_finds_real_markers_near_the_fk_anchor():
    """四台各贴一块真实黑标（解码成 BGR，人标框心落在真黑标的投影上）：真黑标离 FK 锚 2–3 cm（本场常偏的量级），
    YOLO top-1 落在人标框心 1.5 px 内；空白图没有候选。"""
    import cv2
    from swing_hs.camera_proc import bayer_code

    locator = _locator(lambda t: READY_Q)
    exposure_pc = 101.0 + PC_MINUS_RK
    blank = {sn: np.full((1536, 2048, 3), 120, np.uint8) for sn in SERIALS}
    first = locator.process(CAR_POSE, [(0, exposure_pc, blank)])[0]
    assert all(rows == [] for rows in first["candidates"].values())
    anchor_mm = np.asarray(first["anchor_mm"])

    cameras = ht._load_cameras(CALIB, SERIALS)
    true_mm = anchor_mm + np.array([8.0, -5.0, -25.0])
    samples = np.load(MARKER_SAMPLES)
    images, expected = {}, {}
    for sn, raw, box, pixel_type in zip(SERIALS, samples["raw"], samples["box"], samples["pixel_type"]):
        u, v = (float(c) for c in ht._project_raw(cameras[sn], true_mm))
        block = cv2.cvtColor(raw, bayer_code(int(pixel_type)))
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        x0, y0 = int(round(u - cx)), int(round(v - cy))
        assert 0 <= x0 <= 2048 - block.shape[1] and 0 <= y0 <= 1536 - block.shape[0], f"{sn} 看不到黑标"
        image = np.full((1536, 2048, 3), np.median(block, axis=(0, 1)).astype(np.uint8), np.uint8)
        image[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
        images[sn], expected[sn] = image, (x0 + cx, y0 + cy)

    result = locator.process(CAR_POSE, [(2, exposure_pc, images)])[0]
    for sn in SERIALS:
        u, v, score, area, bx, by, bw, bh = result["candidates"][sn][0]
        assert 0.25 <= score <= 1.0 and area > 0
        assert math.hypot(u - expected[sn][0], v - expected[sn][1]) < 1.5
        assert abs(bx + 0.5 * bw - u) <= 1.0 and abs(by + 0.5 * bh - v) <= 1.0


def test_frame_between_car_locs_uses_interpolated_car_pose_and_its_own_joints():
    # 车 50 ms 走 10 cm、J1 同时在转：中间那帧（没做车定位）车位姿取前后两个车定位的插值，关节角取它自己的
    locator = _locator(lambda t: [READY_Q[0] + 2.0 * (t - 101.0), *READY_Q[1:]])
    blank = {sn: np.full((1536, 2048, 3), 120, np.uint8) for sn in SERIALS}
    t0 = 101.0 + PC_MINUS_RK
    first = SimpleNamespace(x=-0.3, y=1.5, z=0.0, yaw=0.02)
    second = SimpleNamespace(x=-0.2, y=1.5, z=0.0, yaw=0.04)
    locator.process(first, [(0, t0, blank)])
    rows = locator.process(second, [(1, t0 + 0.025, blank), (2, t0 + 0.05, blank)])

    assert [row["frame_idx"] for row in rows] == [1, 2]
    middle = SimpleNamespace(x=-0.25, y=1.5, z=0.0, yaw=0.03)
    # 帧时间就是曝光中点；_feed 的最小网络延时 2 ms 被当成钟差的一部分，关节角早取 2 ms
    q_middle = [READY_Q[0] + 2.0 * (0.025 - 0.002), *READY_Q[1:]]
    # 在线锚 = 端点模型的刚体位形黑标（只有关节角缓存，没有力矩历史），z 减车型 hit_pos_z_offset_m 成离地高
    module = ap.arm().endpoint
    marker = module.field(ap.endpoint("v05").pose(np.asarray([q_middle])), "marker", module.POSE)[0]
    z_off = ap.endpoint("v05").z_offset
    c, s = np.cos(middle.yaw), np.sin(middle.yaw)
    expected_mm = 1000.0 * np.array([
        middle.x + c * marker[0] - s * marker[1],
        middle.y + s * marker[0] + c * marker[1],
        middle.z + marker[2] - z_off,
    ])
    assert rows[0]["anchor_mm"] == pytest.approx(expected_mm.tolist(), abs=0.2)


def test_post_run_solves_the_marker_from_online_candidates(tmp_path: Path):
    """收场锚 = 端点动态模型黑标本身（不再估本场偏差）：真黑标偏模型 (6, 4, −22) mm（车体系）时照样解出来，
    逐帧残差与按侧 × 时间窗的汇总都回到这个量。关节 100 Hz、力矩 0、静止位形：动态项为零，模型黑标 = 刚体位形。"""
    tracker_t0, rk_t0, rk_to_pc_bias = 1000.0, 100.0, 2.525
    final_ht_elapsed = 12.525
    exposure_us = 1000.0
    marker_minus_model_car_mm = np.array([6.0, 4.0, -22.0])
    frame = ap.arm().endpoint.zero_frame("v05")[0]
    arm = {
        "car": "v05",
        # 臂起栈 run_config 行（node/run_config.hpp run_identity_text）：录包零位帧 = 当前车型 yaml 的帧
        "run_configs": [{"recv": 0.0, "text": f"run_config: car=v05 config=x | zero_frame={frame} offset_rad=[0,0,0,0,0,0]"}],
        "states": [{"t": 109.0 + 0.01 * k, "position": READY_Q, "velocity": [0.0] * 6, "effort": [0.0] * 6}
                   for k in range(200)],
    }
    car_locs = [
        {"elapsed_s": 12.0 + 0.05 * k, "x": CAR_POSE.x, "y": CAR_POSE.y, "z": CAR_POSE.z,
         "yaw": CAR_POSE.yaw}
        for k in range(21)
    ]
    cameras = ht._load_cameras(CALIB, SERIALS)
    states = ht.arm_states_with_marker(arm, "tracker")
    assert arm["endpoint"]["frame"]["recorded"] == frame
    frames, truth, model = [], [], []
    for k in range(-8, 3):  # ht−400 … +100 ms，车定位帧 20 Hz
        elapsed = final_ht_elapsed + 0.05 * k
        fk_mm = ht._expected_world_mm(
            elapsed, states, car_locs, car_locs, rk_t0, rk_to_pc_bias
        )
        model.append(fk_mm)
        true_mm = fk_mm + ht._car_to_world(CAR_POSE.yaw, marker_minus_model_car_mm)
        candidates = {}
        for sn in SERIALS:
            u, v = (float(c) for c in ht._project_raw(cameras[sn], true_mm))
            candidates[sn] = [[u, v, 100.0, 121, int(u) - 5, int(v) - 5, 11, 11]]
        frames.append({
            "frame_idx": 2 * len(frames),
            "exposure_pc": tracker_t0 + elapsed,  # tracker 帧时间 = 曝光中点
            "anchor_mm": fk_mm.tolist(),
            "candidates": candidates,
        })
        truth.append(true_mm / 1000.0)

    tracker_path, marker_path, arm_path, rk_path, tables_path = (
        tmp_path / name
        for name in ("tracker.json", "marker.json", "arm.json", "rk.json", "tables.json")
    )
    tracker_path.write_text(json.dumps({
        "config": {
            "car_config_path": "vehicle_v05.json",
            "calib_config_path": str(CALIB),
            "serials": SERIALS,
            "first_frame_exposure_pc": tracker_t0,
            "camera_settings": {sn: {"exposure_us": exposure_us} for sn in SERIALS},
        },
        "frames": [{"idx": frame["frame_idx"], "video_frame_idx": frame["frame_idx"]} for frame in frames],
        "car_locs": car_locs,
    }), encoding="utf-8")
    marker_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    arm_path.write_text(json.dumps(arm), encoding="utf-8")
    rk_path.write_text(json.dumps({"t0": rk_t0}), encoding="utf-8")
    tables_path.write_text(json.dumps({
        "align": {
            "auto": {"bias": 2.5, "err": 0.01, "n": 60, "flights": 4, "requiredFlights": 3,
                     "windowSource": "bridge"},
            "timeMap": {"scale": 1, "bias": 2.5},
        },
        "arm_contract": {
            "schema": "arm_final_ht/v4",
            "rkT0": rk_t0,
            "zPhasePolicy": {"maxAbsOffsetMs": 100, "appliesTo": "all_pc_sampling",
                             "rkUse": "global_baseline"},
            "rows": [{
                "reportRow": 1, "accepted": True, "finalMismatch": False, "side": 1,
                "finalHtRkAbs": 110.0, "finalHtPcBaselineElapsed": 12.5,
                "finalHtPcSampleElapsed": final_ht_elapsed,
                "zPhase": {"usable": True, "deltaS": 0.025},
            }],
        },
        "script_error": None,
        "tab_errors": [],
    }), encoding="utf-8")

    result = ht.measure(tracker_path, marker_path, arm_path, rk_path, tables_path)

    assert "reanchor" not in result["config"]
    assert result["config"]["model_anchor"]["status"] == "ok"
    groups = result["config"]["model_residual"]["groups"]
    assert set(groups) == {"forehand [-450,-100) ms", "forehand [-100,-20) ms", "forehand [-20,+20) ms",
                           "forehand [+20,+120) ms"}
    for group in groups.values():
        assert group["median_xyz_car_mm"] == pytest.approx(marker_minus_model_car_mm.tolist(), abs=0.5)
    observations = result["racket_observations"]
    assert [row["frame_idx"] for row in observations] == [f["frame_idx"] for f in frames]
    assert [row["video_frame_idx"] for row in observations] == [f["frame_idx"] for f in frames]
    for row, point, anchor in zip(observations, truth, model):
        assert [row["x"], row["y"], row["z"]] == pytest.approx(point.tolist(), abs=5e-4)
        assert row["side"] == 1
        assert row["model_marker_world"] == pytest.approx((anchor / 1000.0).tolist(), abs=1e-5)
        assert row["residual_car_mm"] == pytest.approx(marker_minus_model_car_mm.tolist(), abs=0.5)
