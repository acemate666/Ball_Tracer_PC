"""臂中心前向登记：marker 随车 yaw 转一次，tag 坐标与历史场参数不混用。"""

import json
import math

import numpy as np
import pytest

import arm_production as ap
from racket_ht_black_marker import _expected_world_mm
from swing_hs.geometry import arm_to_world, car_to_world


@pytest.mark.parametrize("yaw,shift", [(0.0, [0, 1, 0]), (math.pi / 2, [-1, 0, 0]), (math.pi, [0, -1, 0])])
@pytest.mark.parametrize("forward", [0.0, 0.006516706747101281, 0.023])
def test_arm_marker_translation_rotates_once_and_leaves_tag_frame(yaw, shift, forward):
    point = np.array([[0.3, 0.2, 0.8]])
    pose = (2.0, 3.0, yaw)
    tag_world = car_to_world(point, pose, -0.186)
    actual = arm_to_world(point, pose, -0.186, forward)
    np.testing.assert_allclose(actual - tag_world, np.asarray([shift]) * forward, atol=1e-14)
    np.testing.assert_array_equal(point, [[0.3, 0.2, 0.8]])
    np.testing.assert_array_equal(car_to_world(point, pose, -0.186), tag_world)

    # 离线入口接收同场值；黑标已经是离地高，车高只加一次。
    states = [{"t": 100.0 + dt, "marker": point[0].tolist(), "marker_h": 0.986} for dt in [-0.01, 0.01]]
    cars = [{"elapsed_s": dt, "x": 2.0, "y": 3.0, "z": 0.01, "yaw": yaw} for dt in [-0.01, 0.01]]
    world_mm = _expected_world_mm(0.0, states, cars, cars, 100.0, 0.0, forward)
    np.testing.assert_allclose(world_mm, (actual[0] + [0, 0, 0.01]) * 1000, atol=1e-10)


@pytest.mark.parametrize("value", [0.0, 0.006516706747101281])
def test_online_offset_reads_the_sibling_bot_car_config(tmp_path, monkeypatch, value):
    monkeypatch.setattr(ap, "ARM_CONTROLLER_ROOT", tmp_path / "arm_controller")
    path = tmp_path / "arm_bot_center/config/cars/v05.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"/**": {"ros__parameters": {
        "car_name": "v05", "arm_forward_offset_m": value,
    }}}), encoding="utf-8")
    assert ap.arm_forward_offset_m("v05") == value


@pytest.mark.parametrize("params", [{}, {"car_name": "v05"}, {"car_name": "v04", "arm_forward_offset_m": 0},
                                   {"car_name": "v05", "arm_forward_offset_m": True},
                                   {"car_name": "v05", "arm_forward_offset_m": float("nan")}])
def test_online_offset_never_guesses_zero(tmp_path, monkeypatch, params):
    monkeypatch.setattr(ap, "ARM_CONTROLLER_ROOT", tmp_path / "arm_controller")
    path = tmp_path / "arm_bot_center/config/cars/v05.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"/**": {"ros__parameters": params}}), encoding="utf-8")
    with pytest.raises(ap.Unavailable, match="arm_forward_offset_m"):
        ap.arm_forward_offset_m("v05")
