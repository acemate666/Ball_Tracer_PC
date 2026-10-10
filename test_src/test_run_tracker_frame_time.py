from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.run_tracker import _frame_exposure_center_pc


def test_frame_time_is_the_mean_exposure_center():
    # 2026-10-10 用户定：PC 帧时间 = 曝光开始 + 曝光 / 2（800 µs 曝光 ⇒ 比曝光开始晚 0.4 ms）
    frames = {
        "a": SimpleNamespace(exposure_start_pc=100.000000),
        "b": SimpleNamespace(exposure_start_pc=100.000020),
    }
    half = {"a": 0.0004, "b": 0.0004}
    assert _frame_exposure_center_pc(frames, ["a", "b"], half) == pytest.approx(100.000410, abs=1e-9)


def test_each_camera_uses_its_own_exposure_and_absent_serials_are_skipped():
    frames = {
        "a": SimpleNamespace(exposure_start_pc=10.0),
        "b": SimpleNamespace(exposure_start_pc=10.0),
        "hs": SimpleNamespace(exposure_start_pc=10.0),
    }
    half = {"a": 0.0004, "b": 0.0010}
    # 高速相机不在 serials 里：它的 exposure_start_pc 是同步组均值的复制，不参与
    assert _frame_exposure_center_pc(frames, ["a", "b", "c"], half) == pytest.approx(10.0007, abs=1e-12)
