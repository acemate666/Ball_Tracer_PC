# -*- coding: utf-8 -*-
"""swing_hs（100 fps 静态挥拍识别）的纯逻辑单测：ROI 规划、截块认黑标（YOLO）、快速三角化、车位、关节缓存、格号拼组。"""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from swing_hs import geometry, udp_msgs  # noqa: E402
from swing_hs.camera_proc import (  # noqa: E402
    PATCH_BORDER_PX, TagWindowAverager, bayer_code, detect_marker_in_roi, detect_tags, make_tag_detector, slot_of,
    tag_windows,
)
from swing_hs.joints import JointBuffer  # noqa: E402
from swing_hs.marker_solver import FastMarkerSolver, MarkerFusion, SessionOffset  # noqa: E402
from swing_hs.run_swing_hs import (  # noqa: E402
    CarPose, SeqReorder, SlotAssembler, merge_detect_stats, pc_car_loc_payload, pose_from_tag_lists, video_every_slots,
)
from swing_hs.trigger_proc import next_trigger_slot  # noqa: E402

CALIB = REPO / "src" / "config" / "four_camera_calib_18.json"
VEHICLE = REPO / "src" / "config" / "vehicle_v05.json"
ENVELOPE = REPO / "swing_hs" / "envelope_v05.json"
SERIALS = ["DB0260414", "DB0260405", "DB0260373", "DB0260378"]
#: 1010 113329 场车位（双 tag 中位）与 1010 23:01 静态抛球车位（离相机近）
POSE_MID = (-0.0227, 2.5052, math.radians(0.74))
POSE_NEAR = (-0.0498, 1.0281, 0.0255)


@pytest.fixture(scope="module")
def cameras():
    rbm = geometry.load_test_src("racket_ht_black_marker")
    return rbm._load_cameras(CALIB, SERIALS)


@pytest.fixture(scope="module")
def vehicle():
    return geometry.load_vehicle_reference(VEHICLE)


@pytest.fixture(scope="module")
def fore_marker():
    q = geometry.load_envelope(ENVELOPE, "fore", (-45.0, 45.0))
    return geometry.marker_points_car(q[::3], "v05")


# ── 帧率 / ROI ──

def test_fps_model_matches_measured_points():
    # 1006 实测：680 行传感器 156.84；405 1328x680 链路 130.07
    assert geometry.sensor_max_fps(680) == pytest.approx(156.8, abs=0.3)
    assert geometry.link_max_fps(1328, 680) == pytest.approx(130.1, abs=0.3)
    assert geometry.predicted_max_fps(2048, 1536) == pytest.approx(37.35, abs=0.1)


def test_fit_roi_aligns_and_clips():
    roi = geometry.fit_roi(101.3, 57.9, 777.2, 433.1, 20)
    ox, oy, w, h = roi
    assert ox % 16 == 0 and w % 16 == 0 and oy % 4 == 0 and h % 4 == 0
    assert ox <= 81.3 and oy <= 37.9 and ox + w >= 797.2 and oy + h >= 453.1
    roi = geometry.fit_roi(-50, -50, 5000, 5000, 20)
    assert roi == [0, 0, 2048, 1536]


def test_plan_mid_court_fits_100fps_with_both_tags(cameras, vehicle, fore_marker):
    z_off = geometry.marker_z_offset_m("v05")
    plans = geometry.plan_rois(cameras, POSE_MID, fore_marker, z_off, vehicle, geometry.RoiMargins(), fps=100.0)
    assert geometry.plan_problems(plans, 100.0) == []
    for p in plans.values():
        assert p.max_fps > 120.0
        assert p.megapixels < 0.75
        assert p.marker_in_roi == pytest.approx(1.0)
        assert all(p.tags_in_roi.values()) and not p.tags_dropped


def test_plan_near_cameras_flags_414_coverage(cameras, vehicle, fore_marker):
    z_off = geometry.marker_z_offset_m("v05")
    plans = geometry.plan_rois(cameras, POSE_NEAR, fore_marker, z_off, vehicle, geometry.RoiMargins(), fps=100.0)
    assert plans["DB0260414"].marker_in_sensor < 0.8
    assert any("黑标包络" in n for n in plans["DB0260414"].notes)
    for sn in ("DB0260405", "DB0260373", "DB0260378"):
        assert plans[sn].marker_in_sensor == pytest.approx(1.0)


# ── 截块认黑标（YOLO） ──

#: 真实录像里人工核对过的黑标（1011 复核页）：014335 夜场 2 张、115405 白天 2 张，原始 Bayer 160×160 块（ROI 内偶数对齐截的，
#: Bayer 相位不变）+ 人标框（块内像素、像素边缘坐标）
MARKER_SAMPLES = Path(__file__).resolve().parent / "fixtures" / "swing_hs_marker_samples.npz"


@pytest.fixture(scope="module")
def marker_detector():
    from swing_hs.marker_yolo import MarkerYolo

    return MarkerYolo()


def _roi_with_sample(roi, raw, at=(200, 120)):
    """假 ROI（填样本的灰度中位数）里在偶数位置 at 贴一块真实原始 Bayer。"""
    img = np.full((roi[3], roi[2]), int(np.median(raw)), dtype=np.uint8)
    img[at[1]:at[1] + raw.shape[0], at[0]:at[0] + raw.shape[1]] = raw
    return img


def test_detect_marker_in_roi_finds_real_markers(marker_detector):
    """锚点故意偏 (+15, −10) px（在线 FK 锚的典型误差）：候选 top-1 落在人标框心 1.5 px 内，全幅坐标对得上。"""
    samples = np.load(MARKER_SAMPLES)
    roi, at = [512, 300, 640, 480], (200, 120)
    for raw, box, pixel_type in zip(samples["raw"], samples["box"], samples["pixel_type"]):
        img = _roi_with_sample(roi, raw, at)
        cx = roi[0] + at[0] + 0.5 * (box[0] + box[2])
        cy = roi[1] + at[1] + 0.5 * (box[1] + box[3])
        rows, patch = detect_marker_in_roi(marker_detector, img, (roi[0], roi[1]), np.array([cx + 15.0, cy - 10.0]),
                                           bayer_code(int(pixel_type)), radius=48)
        assert patch is not None
        gx, gy, raw_patch = patch
        assert gx % 2 == 0 and gy % 2 == 0
        assert raw_patch.shape[0] <= 2 * (48 + PATCH_BORDER_PX) + 2
        assert 1 <= len(rows) <= 3
        u, v, score, area, bx, by, bw, bh = rows[0]
        assert 0.25 <= score <= 1.0
        assert math.hypot(u - cx, v - cy) < 1.5
        assert abs(bx + 0.5 * bw - u) <= 1.0 and abs(by + 0.5 * bh - v) <= 1.0 and area > 0


def test_detect_marker_nothing_on_blank(marker_detector):
    roi = [512, 300, 640, 480]
    img = np.full((roi[3], roi[2]), 90, dtype=np.uint8)
    rows, patch = detect_marker_in_roi(marker_detector, img, (roi[0], roi[1]), np.array([900.0, 520.0]),
                                       bayer_code(0x01080008), radius=48)
    assert patch is not None and rows == []


def test_detect_marker_out_of_roi(marker_detector):
    roi = [512, 300, 640, 480]
    img = np.full((roi[3], roi[2]), 90, dtype=np.uint8)
    rows, patch = detect_marker_in_roi(marker_detector, img, (roi[0], roi[1]), np.array([100.0, 100.0]),
                                       bayer_code(0x01080008), radius=48)
    assert rows == [] and patch is None


# ── 快速三角化 ──

def _cands_for(cameras, xyz_mm, rng, *, serials=SERIALS, noise=0.3, distractors=3):
    rbm = geometry.load_test_src("racket_ht_black_marker")
    out = {}
    for sn in serials:
        uv = rbm._project_raw(cameras[sn], xyz_mm) + rng.normal(0, noise, 2)
        rows = [(float(uv[0]), float(uv[1]), 100.0, 60, 0, 0, 0, 0)]
        for _ in range(distractors):
            d = rng.uniform(12, 40) * np.array([math.cos(a := rng.uniform(0, 2 * math.pi)), math.sin(a)])
            rows.append((float(uv[0] + d[0]), float(uv[1] + d[1]), 80.0, 50, 0, 0, 0, 0))
        rng.shuffle(rows)
        out[sn] = rows
    return out


def test_fast_solver_recovers_point_among_distractors(cameras):
    rng = np.random.default_rng(3)
    solver = FastMarkerSolver(cameras)
    for xyz in ([900.0, 2500.0, 1000.0], [700.0, 3200.0, 600.0], [1100.0, 1800.0, 1500.0]):
        truth = np.asarray(xyz)
        cands = _cands_for(cameras, truth, rng)
        sol = solver.solve(cands, truth + np.array([20.0, -15.0, 30.0]))
        assert sol is not None
        assert len(sol.serials) == 4
        assert np.linalg.norm(sol.xyz_mm - truth) < 5.0
        assert sol.max_px <= 2.5


def test_fast_solver_needs_three_cameras_and_anchor_gate(cameras):
    rng = np.random.default_rng(5)
    truth = np.array([900.0, 2500.0, 1000.0])
    two = _cands_for(cameras, truth, rng, serials=["DB0260405", "DB0260378"], distractors=0)
    assert FastMarkerSolver(cameras).solve(two, truth + np.array([10.0, 10.0, -10.0])) is None
    sol = FastMarkerSolver(cameras, min_cams=2).solve(two, truth + np.array([10.0, 10.0, -10.0]))
    assert sol is not None and sol.serials == ["DB0260378", "DB0260405"]
    assert np.linalg.norm(sol.xyz_mm - truth) < 15.0
    three = _cands_for(cameras, truth, rng, serials=["DB0260405", "DB0260378", "DB0260373"], distractors=2)
    solver = FastMarkerSolver(cameras)
    assert solver.solve(three, truth + np.array([20.0, 0.0, 0.0])) is not None
    assert solver.solve(three, truth + np.array([0.0, 0.0, 300.0])) is None


def _clutter(cameras, anchor, rng, n):
    rbm = geometry.load_test_src("racket_ht_black_marker")
    out = {}
    for sn in SERIALS:
        uv = rbm._project_raw(cameras[sn], anchor)
        out[sn] = [(float(uv[0] + dx), float(uv[1] + dy), 50.0, 40, 0, 0, 0, 0) for dx, dy in rng.uniform(-38, 38, (n, 2))]
    return out


def test_fast_solver_pure_clutter_rate(cameras):
    """真黑标不在候选里、锚点附近全是杂点（极端情形）。候选来自黑标 YOLO（每台每格 ≤ 3 个框），按每台 3 个杂点测。
    1011 门放宽到 5 / 5 px 后每台 1 / 2 / 3 个杂点误收 0.2 / 1.8 / 5.5 %（旧 3 / 2.5 px：0 / 0 / 1.0 %，各 400 次）；
    旧阈值候选器那种每台 8 个杂点会到 ~47 %（旧门 ~14 %）——放宽的门只配 YOLO 候选，别再接回阈值候选器。"""
    rng = np.random.default_rng(13)
    solver = FastMarkerSolver(cameras)
    false = 0
    for _ in range(200):
        anchor = np.array([rng.uniform(500, 1200), rng.uniform(1500, 3500), rng.uniform(500, 1600)])
        false += solver.solve(_clutter(cameras, anchor, rng, 3), anchor) is not None
    assert false <= 20


def test_fusion_flags_clutter_fits_as_not_smooth(cameras):
    """连续挥拍里偶尔整帧没有黑标（只剩杂点）：真解几乎全连续，杂点凑出的解标成不连续。"""
    rbm = geometry.load_test_src("racket_ht_black_marker")
    rng = np.random.default_rng(17)
    fusion = MarkerFusion(cameras, SERIALS)
    truth_fits = clutter_smooth = 0
    errors = []
    for i in range(300):
        t = 100.0 + 0.01 * i
        # ~3 m/s 的平滑轨迹（每格 10 ms 走 ~3 cm）
        truth = np.array([800.0 + 300.0 * math.sin(0.1 * i), 2500.0 + 300.0 * math.cos(0.1 * i), 1000.0 + 100.0 * math.sin(0.05 * i)])
        anchor = truth - np.array([5.0, -4.0, 20.0]) + rng.normal(0, 2.0, 3)
        cands = _clutter(cameras, anchor, rng, 4)
        marker_visible = i % 5 != 3
        if marker_visible:
            for sn in SERIALS:
                uv = rbm._project_raw(cameras[sn], truth) + rng.normal(0, 0.4, 2)
                cands[sn].append((float(uv[0]), float(uv[1]), 100.0, 70, 0, 0, 0, 0))
        sol, _ = fusion.solve_slot(anchor, 0.0, cands, t)
        if marker_visible:
            assert sol is not None
            errors.append(np.linalg.norm(sol.xyz_mm - truth))
            truth_fits += sol.smooth is True
        elif sol is not None and sol.smooth:
            clutter_smooth += 1
    assert np.max(errors) < 8.0
    assert truth_fits >= 0.9 * len(errors)
    assert clutter_smooth <= 2


def test_fusion_learns_session_offset(cameras):
    rng = np.random.default_rng(7)
    fusion = MarkerFusion(cameras, SERIALS)
    true_offset_car = np.array([-4.0, 5.0, -20.0])
    yaw = 0.02
    rbm = geometry.load_test_src("racket_ht_black_marker")
    for i in range(60):
        fk = np.array([600.0 + 10 * i, 2300.0 + 5 * i, 900.0 + 3 * i])
        truth = fk + rbm._car_to_world(yaw, true_offset_car)
        sol, _ = fusion.solve_slot(fk, yaw, _cands_for(cameras, truth, rng))
        assert sol is not None
    assert fusion.offset.value is not None
    assert np.linalg.norm(fusion.offset.value - true_offset_car) < 3.0
    assert sol.corrected


def test_session_offset_rejects_outliers():
    rng = np.random.default_rng(11)
    so = SessionOffset(refresh=5)
    for _ in range(40):
        so.add(np.array([3.0, -2.0, -18.0]) + rng.normal(0, 2.0, 3))
    for _ in range(8):
        so.add(np.array([90.0, 80.0, 70.0]))
    assert so.value is not None
    assert np.linalg.norm(so.value - np.array([3.0, -2.0, -18.0])) < 2.0


# ── 车位 ──

def test_pose_from_tag_lists_recovers_pose(cameras, vehicle):
    from src.car_localizer import CarLocalizer

    carloc = CarLocalizer(str(CALIB), str(VEHICLE))
    pose = (0.12, 2.72, math.radians(-0.74))
    corners = geometry.tag_corners_car(vehicle, 1.0)
    tag_lists = {}
    for sn in SERIALS:
        tag_lists[sn] = [(tid, geometry.project_m(cameras[sn], geometry.car_to_world(c, pose)).tolist())
                         for tid, c in corners.items()]
    loc = pose_from_tag_lists(carloc, tag_lists, 1.0)
    assert loc is not None
    assert loc.x == pytest.approx(pose[0], abs=1e-3)
    assert loc.y == pytest.approx(pose[1], abs=1e-3)
    assert loc.yaw == pytest.approx(pose[2], abs=math.radians(0.05))
    # 只剩一块 tag：不收（车 yaw 必须双 tag）
    single = {sn: rows[:1] for sn, rows in tag_lists.items()}
    assert pose_from_tag_lists(carloc, single, 1.0) is None


def test_pc_car_loc_payload_matches_tracker_contract():
    """与 tracker 主线 /pc_car_loc 逐字段相同（src/run_tracker.py 车定位线 publish_car_loc 那一处）。"""
    from types import SimpleNamespace

    loc = SimpleNamespace(x=0.123456, y=2.505049, z=0.0, yaw=0.0129361, yaw_valid=True, t=2123873.2685926,
                          tag_id=0, tag_ids=[0, 1])
    payload = pc_car_loc_payload(loc)
    assert payload == {"topic": "car_loc", "x": 0.1235, "y": 2.505, "z": 0.0, "yaw": 0.0129, "yaw_valid": True,
                       "t": 2123873.268593, "tag_id": 0, "tag_ids": [0, 1]}


def test_tag_windows_cover_tags(cameras, vehicle):
    corners = geometry.tag_corners_car(vehicle, 1.0)
    for sn in SERIALS:
        roi = [0, 0, 2048, 1536]
        wins = tag_windows(cameras[sn], POSE_MID, corners, roi, 60.0)
        assert wins
        for c in corners.values():
            uv = geometry.project_m(cameras[sn], geometry.car_to_world(c, POSE_MID))
            assert any(np.all((uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1))
                       for x0, y0, x1, y1 in wins)
            for x0, y0, _, _ in wins:
                assert x0 % 2 == 0 and y0 % 2 == 0


def test_tag_window_averager_blocks_and_averages():
    rng = np.random.default_rng(3)
    pattern = np.full((40, 60), 100.0)
    pattern[10:30, 20:40] = 20.0
    avg = TagWindowAverager(10)
    calls = []

    def windows():
        calls.append(1)
        return [(4, 6, 64, 46)]

    out = []
    for slot in [*range(0, 10), *range(12, 20), 20]:     # 第二块缺两帧
        frame = np.zeros((60, 80), np.uint8)
        frame[6:46, 4:64] = np.clip(pattern + rng.normal(0, 20, pattern.shape), 0, 255).astype(np.uint8)
        done = avg.add(slot, 100.0 + 0.01 * slot, frame, windows)
        if done is not None:
            out.append(done)
    out.append(avg.flush())
    assert [o[0] for o in out] == [0, 10, 20]
    assert out[0][1] == pytest.approx(100.045)
    assert len(calls) == 3                       # 每块只定一次窗
    x0, y0, img = out[0][2][0]
    assert (x0, y0) == (4, 6) and img.shape == (40, 60) and img.dtype == np.uint8
    assert np.abs(img.astype(float) - pattern).std() < 9.0      # 单帧噪声 20 → 10 帧平均 ~6
    assert avg.flush() is None


def test_detect_tags_finds_ids_and_offsets_corners():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = np.full((240, 400), 200, np.uint8)
    for tid, x in ((0, 40), (1, 230)):
        img[60:180, x:x + 120] = cv2.aruco.generateImageMarker(dictionary, tid, 120, borderBits=1)
    found = dict(detect_tags(make_tag_detector(full_frame=False), img, (1000, 500), {0, 1}))
    assert set(found) == {0, 1}
    c0 = np.asarray(found[0])
    assert c0[:, 0].min() == pytest.approx(1040, abs=2) and c0[:, 1].min() == pytest.approx(560, abs=2)
    # 同一 id 在一张图里出现两次：作废
    img[60:180, 230:350] = cv2.aruco.generateImageMarker(dictionary, 0, 120, borderBits=1)
    assert dict(detect_tags(make_tag_detector(full_frame=False), img, (0, 0), {0, 1})) == {}


def test_car_pose_median_and_shared():
    shared = mp.get_context("spawn").Array("d", 6)
    car = CarPose(shared, window=5)
    for x in (0.10, 0.11, 0.50, 0.12, 0.13):
        car.accept(x, 2.0, 0.01, 1.0)
    assert car.current[0] == pytest.approx(0.12)
    assert list(shared[:6]) == pytest.approx([0.12, 2.0, 0.0, 0.01, 1.0, 1.0])


# ── 关节缓存 / 消息 / 格号 ──

def test_joint_buffer_interpolates_on_pc_axis():
    names = ("joint1", "joint2", "joint3", "joint4", None, "joint5")
    buf = JointBuffer(names)
    offset = 2150340.0
    for i in range(100):
        stamp = 100.0 + 0.005 * i
        latency = 0.003 + (0.004 if i % 3 else 0.0)
        buf.add(stamp, ["joint1", "joint2", "joint3", "joint4", "joint5"], [0.01 * i, 0, 0, 0, 0.5], stamp + offset + latency)
    assert buf.offset() == pytest.approx(offset + 0.003, abs=1e-4)
    q = buf.at_pc(100.2525 + offset + 0.003)
    assert q is not None and q[0] == pytest.approx(0.505, abs=1e-6) and q[4] == 0.0 and q[5] == 0.5
    assert buf.latest_pc() == pytest.approx(100.495 + offset + 0.003, abs=1e-4)
    assert buf.at_pc(100.6 + offset) is None
    buf.add(101.0, ["a", "b"], [0, 0], 0.0)
    assert buf.name_mismatch == ["a", "b"]


def test_udp_roundtrip():
    data = udp_msgs.encode_joint_state(1.5, 2.5, ["joint1"], [0.25])
    assert udp_msgs.decode_joint_state(data) == (1.5, 2.5, ["joint1"], [0.25])
    topic, payload = udp_msgs.decode_publish(udp_msgs.encode_publish("/x", {"a": 1}))
    assert topic == "/x" and payload == '{"a":1}'
    assert udp_msgs.decode_any(udp_msgs.encode_status({"joint_rx": 3}))["kind"] == "bridge"


def test_slots_and_trigger_grid():
    assert slot_of(10.0 + 0.00006, 10.0, 0.01) == 0
    assert slot_of(10.0 + 0.03007, 10.0, 0.01) == 3
    assert next_trigger_slot(5, 0.0502, period=0.01, minimum_period=0.0098) == 6
    assert next_trigger_slot(5, 0.0565, period=0.01, minimum_period=0.0098) == 7


def test_slot_assembler_completes_and_expires():
    asm = SlotAssembler({"a", "b"}, timeout_slots=3)
    assert asm.add(1, "a", 1) == []
    assert asm.add(1, "b", 2) == [(1, {"a": 1, "b": 2})]
    assert asm.add(2, "a", 3) == []
    out = asm.add(6, "a", 4)
    assert out == [(2, {"a": 3})]
    assert asm.expire(force=True) == [(6, {"a": 4})]


def test_seq_reorder_restores_submit_order():
    r = SeqReorder(max_hold=5)
    assert r.push(1, "b") == []
    assert r.push(0, "a") == ["a", "b"]
    assert r.push(3, "d") == []
    assert r.push(2, "c") == ["c", "d"]
    assert r.flush() == [] and r.skipped == 0


def test_seq_reorder_skips_missing_after_max_hold():
    r = SeqReorder(max_hold=2)
    assert r.push(1, "b") == []
    assert r.push(2, "c") == []
    assert r.push(3, "d") == ["b", "c", "d"]     # 0 号一直没来：攒到 3 条 > max_hold 就跳过去
    assert r.skipped == 1
    assert r.push(0, "a") == ["a"]               # 跳号之后才到的直接放
    assert r.push(5, "f") == []
    assert r.flush() == ["f"]


def test_merge_detect_stats_sums_counts_and_keeps_peaks():
    merged = merge_detect_stats({
        0: {"worker": 0, "processed": 10, "with_candidates": 8, "proc_ms_sum": 80.0, "proc_ms_max": 12.0,
            "tag_runs": 1, "tag_hits": 1, "backlog": 3, "backlog_max": 7},
        1: {"worker": 1, "processed": 9, "with_candidates": 9, "proc_ms_sum": 70.0, "proc_ms_max": 15.0,
            "tag_runs": 0, "tag_hits": 0, "backlog": 2, "backlog_max": 9},
    })
    assert (merged["workers"], merged["processed"], merged["with_candidates"]) == (2, 19, 17)
    assert (merged["proc_ms_sum"], merged["proc_ms_max"]) == (150.0, 15.0)
    assert (merged["backlog"], merged["backlog_max"], merged["tag_runs"]) == (3, 9, 1)
    assert merge_detect_stats({})["processed"] == 0


def test_video_every_slots():
    assert video_every_slots(100.0, 5.0) == 20
    assert video_every_slots(100.0, 100.0) == 1
    assert video_every_slots(100.0, 0.0) == 0
    with pytest.raises(ValueError):
        video_every_slots(100.0, 3.0)
