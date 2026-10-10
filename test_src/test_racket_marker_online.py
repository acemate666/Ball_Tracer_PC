"""黑标在线候选的核心 contract：按曝光时刻取到对应的关节角；FK 锚投影处的黑方块被截出来；
收场在这些候选里解出三维。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import extract_arm_bag as eab
import racket_ht_black_marker as ht
from src.racket_marker_online import JointStateBuffer, RacketMarkerLocator

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "src" / "config" / "four_camera_calib_18.json"
SERIALS = list(json.loads(CALIB.read_text(encoding="utf-8"))["cameras"])
# 1010 113329 场 ready 位的一条实测关节角（槽 4 是五轴臂的空槽）
READY_Q = [-1.56487, -0.15568, 2.48771, 2.42524, 0.0, 0.05177]
# 车在这里时 ready 位黑标四台相机都看得到
CAR_POSE = SimpleNamespace(x=-0.3, y=1.5, z=0.0, yaw=0.02)
PC_MINUS_RK = 50.0


def _feed(buffer: JointStateBuffer, q_of_t, *, latency_s=lambda k: 0.002) -> None:
    """200 Hz 的 /joint_states：RK 时刻 100.000…101.995 s，关节名倒序发（按名字取槽）。"""
    slots = eab.CAR_MODELS["v05"].slot_joint_names
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


def test_locator_crops_the_marker_square_at_the_fk_anchor():
    locator = _locator(lambda t: READY_Q)
    exposure_pc = 101.0 + PC_MINUS_RK
    blank = {sn: np.full((1536, 2048, 3), 120, np.uint8) for sn in SERIALS}
    anchor_mm = np.asarray(locator.process(CAR_POSE, [(0, exposure_pc, blank)])[0]["anchor_mm"])

    # 真黑标离 FK 锚 2–3 cm（本场常偏的量级），画成全分辨率下 11 px 的黑方块
    cameras = ht._load_cameras(CALIB, SERIALS)
    true_mm = anchor_mm + np.array([8.0, -5.0, -25.0])
    images, expected = {}, {}
    for sn in SERIALS:
        u, v = (int(round(float(c))) for c in ht._project_raw(cameras[sn], true_mm))
        assert 6 <= u < 2042 and 6 <= v < 1530, f"{sn} 看不到黑标"
        image = np.full((1536, 2048, 3), 120, np.uint8)
        image[v - 5 : v + 6, u - 5 : u + 6] = 15
        images[sn], expected[sn] = image, (u, v)

    result = locator.process(CAR_POSE, [(2, exposure_pc, images)])[0]
    for sn in SERIALS:
        assert result["candidates"][sn][0][:2] == pytest.approx(expected[sn], abs=1e-6)


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
    marker = eab.fk(q_middle, car="v05")["marker"]
    z_off = eab.standard_kinematics("v05").car_hit_pos_z_offset("v05")
    c, s = np.cos(middle.yaw), np.sin(middle.yaw)
    expected_mm = 1000.0 * np.array([
        middle.x + c * marker[0] - s * marker[1],
        middle.y + s * marker[0] + c * marker[1],
        middle.z + marker[2] - z_off,
    ])
    assert rows[0]["anchor_mm"] == pytest.approx(expected_mm.tolist(), abs=0.2)


def test_post_run_solves_the_marker_from_online_candidates(tmp_path: Path):
    tracker_t0, rk_t0, rk_to_pc_bias, z_off = 1000.0, 100.0, 2.525, -0.1471
    final_ht_elapsed = 12.525
    exposure_us = 1000.0
    marker_minus_fk_car_mm = np.array([6.0, 4.0, -22.0])
    arm = {"car": "v05", "states": [{"t": 109.0 + 0.005 * k, "position": READY_Q} for k in range(400)]}
    car_locs = [
        {"elapsed_s": 12.0 + 0.05 * k, "x": CAR_POSE.x, "y": CAR_POSE.y, "z": CAR_POSE.z,
         "yaw": CAR_POSE.yaw}
        for k in range(21)
    ]
    cameras = ht._load_cameras(CALIB, SERIALS)
    states = ht.arm_states_with_marker(arm)
    frames, truth = [], []
    for k in range(-8, 3):  # ht−400 … +100 ms，车定位帧 20 Hz
        elapsed = final_ht_elapsed + 0.05 * k
        fk_mm = ht._expected_world_mm(
            elapsed, states, car_locs, car_locs, rk_t0, rk_to_pc_bias, z_off
        )
        true_mm = fk_mm + ht._car_to_world(CAR_POSE.yaw, marker_minus_fk_car_mm)
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
            "calibration": {"zOff": z_off},
            "rows": [{
                "reportRow": 1, "accepted": True, "finalMismatch": False,
                "finalHtRkAbs": 110.0, "finalHtPcBaselineElapsed": 12.5,
                "finalHtPcSampleElapsed": final_ht_elapsed,
                "zPhase": {"usable": True, "deltaS": 0.025},
            }],
        },
        "script_error": None,
        "tab_errors": [],
    }), encoding="utf-8")

    result = ht.measure(tracker_path, marker_path, arm_path, rk_path, tables_path)

    assert result["config"]["reanchor"]["offset_car_mm"] == pytest.approx(
        marker_minus_fk_car_mm.tolist(), abs=0.05
    )
    observations = result["racket_observations"]
    assert [row["frame_idx"] for row in observations] == [frame["frame_idx"] for frame in frames]
    assert [row["video_frame_idx"] for row in observations] == [frame["frame_idx"] for frame in frames]
    for row, point in zip(observations, truth):
        assert [row["x"], row["y"], row["z"]] == pytest.approx(point.tolist(), abs=5e-4)
