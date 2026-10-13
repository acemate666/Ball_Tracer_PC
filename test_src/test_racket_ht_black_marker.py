from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from racket_ht_black_marker import (
    ANCHOR_MAX_DEPTH_MM,
    ANCHOR_MAX_LATERAL_MM,
    MARKER_LOO_MAX_MM,
    MARKER_ROI_RADIUS_PX,
    RESIDUAL_WINDOWS_MS,
    AnchorGate,
    CameraModel,
    MarkerCandidate,
    NothingToMeasure,
    _car_to_world,
    _component_candidates,
    _grid_panels,
    _measurement_context,
    _world_to_car,
    arm_states_with_marker,
    model_residual_summary,
    residual_lines,
)


SERIALS = ["cam0", "cam1", "cam2", "cam3"]


def _write_inputs(tmp_path: Path, *, align_err: float = 0.01, tab_errors=None):
    tracker = tmp_path / "tracker.json"
    arm = tmp_path / "arm.json"
    rk = tmp_path / "rk.json"
    tables = tmp_path / "tables.json"
    tracker.write_text(
        json.dumps(
            {
                "config": {
                    "car_config_path": "vehicle_v04.json",
                    "serials": SERIALS,
                    "camera_settings": {
                        serial: {"exposure_us": 9000.0} for serial in SERIALS
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    arm.write_text(json.dumps({"car": "v04"}), encoding="utf-8")
    rk.write_text(json.dumps({"t0": 100.0}), encoding="utf-8")
    tables.write_text(
        json.dumps(
            {
                "align": {
                    "auto": {
                        "bias": 2.5,
                        "err": align_err,
                        "n": 60,
                        "flights": 4,
                        "requiredFlights": 3,
                        "windowSource": "bridge",
                    },
                    "timeMap": {"scale": 1, "bias": 2.5},
                },
                "arm_contract": {
                    "schema": "arm_final_ht/v4",
                    "rkT0": 100.0,
                    "zPhasePolicy": {
                        "maxAbsOffsetMs": 100,
                        "appliesTo": "all_pc_sampling",
                        "rkUse": "global_baseline",
                    },
                    "rows": [
                        {
                            "reportRow": 1,
                            "accepted": True,
                            "finalMismatch": False,
                            "finalHtRkAbs": 110.0,
                            "finalHtPcBaselineElapsed": 12.5,
                            "finalHtPcSampleElapsed": 12.525,
                            "zPhase": {"usable": True, "deltaS": 0.025},
                        },
                        {
                            "reportRow": 2,
                            "accepted": False,
                            "finalMismatch": False,
                            "finalHtRkAbs": None,
                            "finalHtPcBaselineElapsed": None,
                        },
                    ],
                },
                "script_error": None,
                "tab_errors": tab_errors or [],
            }
        ),
        encoding="utf-8",
    )
    return tracker, arm, rk, tables


def test_measurement_context_uses_report_final_ht(tmp_path: Path):
    # 帧时间就是曝光中点（只在 tracker 入口补一次半曝光），上下文里不再有曝光补偿量；
    # 高度不再用报告的 zOff（模型黑标的离地高由臂端点模型给），合同里没有击球侧就是 None
    paths = _write_inputs(tmp_path)
    _, _, _, serials, targets = _measurement_context(*paths)
    assert serials == SERIALS
    assert targets == [{
        "report_row": 1,
        "final_ht_pc_elapsed_s": 12.525,
        "rk_to_pc_bias_s": 2.525,
        "side": None,
    }]


@pytest.mark.parametrize(("side", "expected"), [(1, 1), (-1, -1), (0, None), ("forehand", None), (None, None)])
def test_measurement_context_carries_the_contract_side(tmp_path: Path, side, expected):
    # 击球侧由报告合同给（臂 node::final_plan_side 的生产判决或臂当时自己记的 rl_swing done），这里只认 ±1
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["rows"][0]["side"] = side
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    *_, targets = _measurement_context(*paths)
    assert targets[0]["side"] == expected


def test_measurement_context_requires_usable_throw_phase(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    row = payload["arm_contract"]["rows"][0]
    row["zPhase"] = {"usable": False, "deltaS": None}
    row["finalHtPcSampleElapsed"] = None
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="usable zPhase"):
        _measurement_context(*paths)


def test_measurement_context_rejects_inconsistent_pc_sample(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["rows"][0]["finalHtPcSampleElapsed"] = 12.526
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline plus zPhase"):
        _measurement_context(*paths)


def test_measurement_context_rejects_old_visual_contract(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["schema"] = "arm_final_ht/v3"
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="arm_final_ht/v4"):
        _measurement_context(*paths)


def test_measurement_context_skips_a_session_without_throws(tmp_path: Path):
    """0 抛的场次（RK 没发过 /predict_hit_pos）只是没东西可测，不是报告不可信。

    报告页对空场次也发 contract（rows=[]），所以这里能和"合同缺失=报告代码太旧"分开：
    前者跳过让后处理继续，后者必须报错。对齐质量此时不参与判定。
    """
    paths = _write_inputs(tmp_path, align_err=0.5)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"]["rows"] = []
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NothingToMeasure):
        _measurement_context(*paths)


def test_measurement_context_still_rejects_a_missing_contract(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["arm_contract"] = None
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="arm_final_ht/v4"):
        _measurement_context(*paths)


@pytest.mark.parametrize(
    ("align_err", "tab_errors"),
    [(0.081, None), (0.01, ["sw(5): failed"])],
)
def test_measurement_context_rejects_untrusted_report(
    tmp_path: Path, align_err: float, tab_errors
):
    paths = _write_inputs(tmp_path, align_err=align_err, tab_errors=tab_errors)
    with pytest.raises(ValueError):
        _measurement_context(*paths)


def _camera(image_size: tuple[int, int]) -> CameraModel:
    return CameraModel(
        serial="cam",
        K=np.eye(3),
        D=np.zeros(5),
        R=np.eye(3),
        t=np.zeros(3),
        rvec=np.zeros(3),
        P=np.zeros((3, 4)),
        image_size=image_size,
    )


@pytest.mark.parametrize("frame_size", [(2048, 1536), (2048, 1304)])
def test_grid_panels_restore_the_captured_frame_not_the_calibrated_height(frame_size):
    """A bottom-cropped sensor ROI must not be stretched back to the calib height.

    camera_18.json roi_height shortened the capture to 1304 rows while the
    calibration still records 1536; restoring panels to the calibrated size
    scaled every row by 1.178 and shifted projected anchors ~0.18*v px.
    """
    width, height = frame_size
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    # one bright pixel per quadrant, at a known place inside each panel
    for index in range(4):
        x = (index % 2) * (width // 2) + 100
        y = (index // 2) * (height // 2) + 200
        grid[y, x] = (0, 0, 255)
    cameras = {serial: _camera((2048, 1536)) for serial in SERIALS}
    panels = _grid_panels(grid, SERIALS, cameras)
    for serial in SERIALS:
        panel = panels[serial]
        assert panel.shape[:2] == (height, width)
        ys, xs = np.nonzero(panel[:, :, 2])
        # half-scale panel restored by exactly 2x: (100, 200) -> (200, 400)
        assert int(round(xs.mean())) == pytest.approx(200, abs=2)
        assert int(round(ys.mean())) == pytest.approx(400, abs=2)


def test_grid_panels_reject_a_video_that_is_not_a_roi_of_the_calibrated_frame():
    grid = np.zeros((1304, 1600, 3), dtype=np.uint8)
    cameras = {serial: _camera((2048, 1536)) for serial in SERIALS}
    with pytest.raises(ValueError, match="bottom-cropped ROI"):
        _grid_panels(grid, SERIALS, cameras)


def _set_car(paths, car: str, layout: str) -> None:
    tracker, arm = paths[0], paths[1]
    payload = json.loads(tracker.read_text(encoding="utf-8"))
    payload["config"]["car_config_path"] = layout
    tracker.write_text(json.dumps(payload), encoding="utf-8")
    arm.write_text(json.dumps({"car": car}), encoding="utf-8")


def test_measurement_context_accepts_v05(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    _set_car(paths, "v05", r"D:\robot\Ball_Tracer_PC\src\config\vehicle_v05.json")
    _, arm, _, _, targets = _measurement_context(*paths)
    assert arm["car"] == "v05"
    assert [row["report_row"] for row in targets] == [1]


@pytest.mark.parametrize(
    ("car", "layout"),
    [("v05", "vehicle_v04.json"), ("v04", "vehicle_v05.json"), ("v03", "vehicle_v03.json")],
)
def test_measurement_context_rejects_car_layout_mismatch(tmp_path: Path, car, layout):
    paths = _write_inputs(tmp_path)
    _set_car(paths, car, layout)
    with pytest.raises(ValueError, match="V04/V05"):
        _measurement_context(*paths)


def test_anchor_is_the_model_marker_with_split_gates_and_no_loo_mm_gate():
    # 锚 = 端点动态模型黑标本身（不估本场偏差）；深度 / 横向分开设门，不设 LOO 毫米门（v05 黑方块 ~5 px）
    assert (ANCHOR_MAX_LATERAL_MM, ANCHOR_MAX_DEPTH_MM) == (35.0, 80.0)
    assert MARKER_LOO_MAX_MM is None
    source = Path(__file__).with_name("racket_ht_black_marker.py").read_text(encoding="utf-8")
    for gone in ("_session_marker_offset", "REANCHOR_", "CAR_MARKER_POLICY", "eab.fk(", "zOff"):
        assert gone not in source, gone


def _obs(side, dt_ms, residual):
    return {"side": side, "dt_ht_ms": dt_ms, "residual_car_mm": residual}


def test_model_residual_summary_groups_by_side_and_window():
    observations = [
        _obs(1, -300.0, [1.0, 2.0, 20.0]),
        _obs(1, -200.0, [3.0, 2.0, 30.0]),
        _obs(1, -50.0, [0.0, 20.0, 0.0]),
        _obs(-1, -300.0, [-10.0, 0.0, 0.0]),
        _obs(None, 50.0, [0.0, 0.0, 4.0]),
        _obs(1, -300.0, None),          # 缺车 yaw / 模型黑标：不进汇总
        _obs(1, 500.0, [9.0, 9.0, 9.0]),  # 窗外
    ]
    summary = model_residual_summary(observations)
    groups = summary["groups"]
    lo, hi = RESIDUAL_WINDOWS_MS[0]
    early = groups[f"forehand [{lo:+.0f},{hi:+.0f}) ms"]
    assert early["n"] == 2
    assert early["median_xyz_car_mm"] == [2.0, 2.0, 25.0]
    norms = [np.linalg.norm([1.0, 2.0, 20.0]), np.linalg.norm([3.0, 2.0, 30.0])]
    assert early["median_3d_mm"] == pytest.approx(round(float(np.median(norms)), 1))
    assert early["rms_3d_mm"] == pytest.approx(round(float(np.sqrt(np.mean(np.square(norms)))), 1))
    assert groups["forehand [-100,-20) ms"]["median_xyz_car_mm"] == [0.0, 20.0, 0.0]
    assert groups["backhand [-450,-100) ms"]["n"] == 1
    assert groups["unknown [+20,+120) ms"]["n"] == 1
    assert sum(g["n"] for g in groups.values()) == 5
    lines = residual_lines(summary)
    assert len(lines) == len(groups)
    assert lines[0].startswith("视觉 − 模型黑标 forehand [-450,-100) ms: n=2")


def test_marker_anchor_refuses_what_the_production_endpoint_cannot_compute():
    # 端点动态模型只覆盖五轴臂：v04 场次不出黑标观测（NothingToMeasure 带原因），不另用别的 FK 凑
    import arm_production as ap

    try:
        ap.arm()
    except ap.Unavailable as exc:
        pytest.skip(str(exc))
    arm = {"car": "v04", "states": [{"t": 1.0 + 0.01 * k, "position": [0.0] * 6, "effort": [0.0] * 6}
                                    for k in range(200)]}
    with pytest.raises(NothingToMeasure, match="五轴"):
        arm_states_with_marker(arm, "tracker_test")


def test_car_world_rotation_round_trips():
    offset = np.array([6.0, 9.0, -50.0])
    for yaw in (-2.5, -0.03, 0.0, 0.4, 3.0):
        world = _car_to_world(yaw, offset)
        assert np.linalg.norm(world[:2]) == pytest.approx(np.linalg.norm(offset[:2]))
        assert _world_to_car(yaw, world) == pytest.approx(offset)
    # yaw=+90°：车体 x（右）指向世界 +y
    assert _car_to_world(np.pi / 2, np.array([1.0, 0.0, 0.0])) == pytest.approx([0.0, 1.0, 0.0])


def test_anchor_gate_separates_view_depth_from_lateral():
    gate = AnchorGate(np.array([0.0, 1.0, 0.0]), max_lateral_mm=35.0, max_depth_mm=80.0)
    assert gate.split(np.array([20.0, -60.0, 10.0])) == pytest.approx(
        (np.hypot(20.0, 10.0), 60.0)
    )
    # 深度方向三角化噪声大：沿视线 70 mm 仍收，横向 40 mm（拍框/背景假点的量级）就拒
    assert gate.rank(np.array([0.0, 70.0, 0.0])) is not None
    assert gate.rank(np.array([40.0, 0.0, 0.0])) is None
    assert gate.rank(np.array([0.0, -90.0, 0.0])) is None
    near = gate.rank(np.array([5.0, 20.0, 5.0]))
    far = gate.rank(np.array([25.0, 20.0, 5.0]))
    assert near is not None and far is not None and near < far


def _component_candidates_full_roi(
    image, anchor_uv, *, radius, thresholds, anchor_limit_px, result_limit,
    prefer_near=False, dedupe_px=5.0,
):
    """改快之前的实现（每个连通域在整块 ROI 上求质心/对比度），只用来对拍。"""
    height, width = image.shape[:2]
    anchor_x, anchor_y = anchor_uv
    x0 = max(0, int(math.floor(anchor_x - radius)))
    y0 = max(0, int(math.floor(anchor_y - radius)))
    x1 = min(width, int(math.ceil(anchor_x + radius)))
    y1 = min(height, int(math.ceil(anchor_y + radius)))
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = float(np.median(gray))
    ys, xs = np.indices(gray.shape)
    found = []
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
            component = labels == label
            weights = np.where(component, np.maximum(background - gray, 1.0), 0.0)
            mass = float(weights.sum())
            if mass <= 0.0:
                continue
            u = x0 + float((weights * xs).sum() / mass)
            v = y0 + float((weights * ys).sum() / mass)
            distance = math.hypot(u - anchor_x, v - anchor_y)
            if anchor_limit_px is not None and distance > anchor_limit_px:
                continue
            contrast = float(np.median(background - gray[component]))
            scale = 25.0 if anchor_limit_px is not None else 140.0
            found.append(
                MarkerCandidate(
                    uv=(u, v),
                    score=max(contrast, 1.0) * fill * min(area, 250)
                    / (1.0 + (distance / scale) ** 2),
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
    deduped = []
    for item in found:
        if all(np.linalg.norm(np.subtract(item.uv, old.uv)) > dedupe_px for old in deduped):
            deduped.append(item)
        if len(deduped) == result_limit:
            break
    return deduped


@pytest.mark.parametrize("seed", range(4))
def test_component_candidates_match_the_full_roi_implementation(seed):
    # 包围盒内求质心是纯提速：同一批连通域、同样的门与排序，坐标只差浮点求和顺序
    rng = np.random.default_rng(seed)
    image = (150 + rng.integers(-12, 13, (640, 760, 3))).astype(np.uint8)
    anchor = (380.4, 317.6)
    for _ in range(60):
        near = rng.random() < 0.4
        cx = anchor[0] + rng.uniform(-40, 40) if near else rng.uniform(0, 760)
        cy = anchor[1] + rng.uniform(-40, 40) if near else rng.uniform(0, 640)
        axes = (int(rng.integers(2, 16)), int(rng.integers(2, 16)))
        level = int(rng.integers(0, 50))
        cv2.ellipse(
            image, (int(cx), int(cy)), axes, float(rng.uniform(0, 180)), 0, 360,
            (level, level, level), -1,
        )
    searches = (
        dict(radius=55, thresholds=[8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 56.0],
             anchor_limit_px=45.0, result_limit=12, prefer_near=True, dedupe_px=3.0),
        dict(radius=MARKER_ROI_RADIUS_PX, thresholds=[8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
             anchor_limit_px=None, result_limit=32),
    )
    for kwargs in searches:
        fast = _component_candidates(image, anchor, **kwargs)
        reference = _component_candidates_full_roi(image, anchor, **kwargs)
        assert len(fast) == len(reference) > 0
        for got, want in zip(fast, reference):
            assert (got.area, got.bbox_xywh) == (want.area, want.bbox_xywh)
            assert got.uv == pytest.approx(want.uv, abs=1e-6)
            assert got.score == pytest.approx(want.score, rel=1e-9)
