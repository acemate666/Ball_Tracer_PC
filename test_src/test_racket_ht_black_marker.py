from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from racket_ht_black_marker import (
    CAR_MARKER_POLICY,
    REANCHOR_INLIER_MM,
    REANCHOR_MIN_INLIERS,
    AnchorGate,
    CameraModel,
    NothingToMeasure,
    _car_to_world,
    _grid_panels,
    _measurement_context,
    _session_marker_offset,
    _world_to_car,
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
                    "video_frame_mapping_exact": True,
                    "video_output": {
                        "layout": "grid",
                        "grid_cols": 2,
                        "grid_rows": 2,
                        "serial_order": SERIALS,
                    },
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
                    "calibration": {"zOff": -0.1471},
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


def test_measurement_context_uses_report_final_ht_and_exposure_center(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    _, _, _, serials, exposure_offset, z_offset, targets = (
        _measurement_context(*paths)
    )
    assert serials == SERIALS
    assert exposure_offset == pytest.approx(0.0045)
    assert z_offset == pytest.approx(-0.1471)
    assert targets == [{
        "report_row": 1,
        "final_ht_pc_elapsed_s": 12.525,
        "rk_to_pc_bias_s": 2.525,
    }]


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
    _, arm, _, _, _, _, targets = _measurement_context(*paths)
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


def test_car_policy_keeps_v04_gates_and_reanchors_v05():
    # v04 已验证的解算（10 mm LOO、按原始 FK 排名）不动；v05 换成校正锚点 + 无 LOO 毫米门。
    assert CAR_MARKER_POLICY["v04"].loo_max_mm == 10.0
    assert CAR_MARKER_POLICY["v04"].reanchor is False
    assert CAR_MARKER_POLICY["v05"].loo_max_mm is None
    assert CAR_MARKER_POLICY["v05"].reanchor is True


def test_car_world_rotation_round_trips():
    offset = np.array([6.0, 9.0, -50.0])
    for yaw in (-2.5, -0.03, 0.0, 0.4, 3.0):
        world = _car_to_world(yaw, offset)
        assert np.linalg.norm(world[:2]) == pytest.approx(np.linalg.norm(offset[:2]))
        assert _world_to_car(yaw, world) == pytest.approx(offset)
    # yaw=+90°：车体 x（右）指向世界 +y
    assert _car_to_world(np.pi / 2, np.array([1.0, 0.0, 0.0])) == pytest.approx([0.0, 1.0, 0.0])


def test_session_marker_offset_is_the_inlier_median():
    rng = np.random.default_rng(7)
    truth = np.array([6.0, 9.0, -50.0])
    inliers = [truth + rng.normal(0.0, [5.0, 15.0, 5.0]) for _ in range(20)]
    # 拍框/背景假点：离真值 60–150 mm（0929 v05 实场 14 个假点的量级）
    outliers = [truth + np.array(v) for v in (
        [30, -40, 80], [-35, 5, 95], [-110, 20, 60], [40, -10, 50],
        [-5, 110, -120], [30, 20, 130], [-60, -40, 10], [45, -50, 65],
    )]
    offset, count = _session_marker_offset(inliers + outliers)
    assert count == 20
    assert offset == pytest.approx(np.median(np.array(inliers), axis=0))
    assert np.linalg.norm(offset - truth) < 10.0


def test_session_marker_offset_refuses_weak_or_minority_clusters():
    truth = np.array([6.0, 9.0, -50.0])
    few = [truth + np.array([k, 0.0, 0.0]) for k in range(REANCHOR_MIN_INLIERS - 1)]
    assert _session_marker_offset(few) is None
    # 聚成一团的不过半：宁可整场不出，也不拿假点定锚
    cluster = [truth + np.array([k, 0.0, 0.0]) for k in range(REANCHOR_MIN_INLIERS)]
    scattered = [
        truth + (3.0 * REANCHOR_INLIER_MM) * np.array([np.cos(a), np.sin(a), 0.5])
        for a in np.linspace(0.0, 2.0 * np.pi, REANCHOR_MIN_INLIERS + 1, endpoint=False)
    ]
    assert _session_marker_offset(cluster + scattered) is None


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
