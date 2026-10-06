"""高速相机（一台小 ROI 按 N 倍帧率跑、每 N 格与其余相机同拍）的单测。

现场配置见 src/config/camera_18_hs405.json：405 裁 1328x680@(720,268)、120 fps = 3x40，
相机实测上限 130.07 fps（单台 1GbE），其余三台 ROI 1392 行上限 41.21 fps。
"""
from __future__ import annotations

import json
import struct
import threading
import time
import types

import numpy as np
import pytest

from src import ball_grabber
from src.ball_grabber import (
    HIGH_SPEED_GROUP_BIT,
    HIGH_SPEED_INDEX_FORMAT,
    ActionTriggerLoop,
    ActionTriggerProcess,
    Frame,
    HighSpeedConfig,
    HighSpeedRecorder,
    HighSpeedStream,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_RGB8_Packed,
    SyncCapture,
    frame_to_numpy,
    pad_roi_frame,
)

SLOW_CEILING = 41.21      # ROI 1392 行，117.5e6/(2048*1392)
FAST_CEILING = 130.07     # 1328x680，117.5e6/(1328*680)
MARGIN = 0.0001


def _hs_trigger(**overrides) -> ActionTriggerLoop:
    kwargs = dict(
        stop_event=threading.Event(),
        fps=120.0,
        acquisition_frame_rate=FAST_CEILING,
        device_key=1,
        group_key=1,
        group_mask=1,
        broadcast_address="255.255.255.255",
        timeout_ms=0,
        sync_every=3,
        fast_group_mask=HIGH_SPEED_GROUP_BIT,
        sync_minimum_period_s=1.0 / SLOW_CEILING + MARGIN,
        minimum_period_margin_s=MARGIN,
    )
    kwargs.update(overrides)
    return ActionTriggerLoop(**kwargs)


def test_every_third_slot_fires_every_camera():
    trigger = _hs_trigger()
    period = 1.0 / 120.0
    kinds = []
    last_sync = -np.inf
    for slot in range(1, 10):
        kind = trigger._slot_kind(slot, slot * period, last_sync)
        if kind == "sync":
            last_sync = slot * period
        kinds.append(kind)
    assert kinds == ["fast", "fast", "sync"] * 3


def test_late_all_camera_fire_downgrades_the_next_one():
    """上一个全组触发迟到 1 ms，下一个准点的全组触发离它只有 24.0 ms，
    低于普通相机 24.37 ms 的下限——发出去会被它们静默吞掉，所以降级成只打高速相机。"""
    trigger = _hs_trigger()
    period = 1.0 / 120.0
    late_sync = 3 * period + 0.001
    assert trigger._slot_kind(6, 6 * period, late_sync) == "downgraded"
    # 迟到 0.5 ms 还在 0.63 ms 的容忍内
    assert trigger._slot_kind(6, 6 * period, 3 * period + 0.0005) == "sync"


def test_single_rate_loop_is_unchanged():
    trigger = ActionTriggerLoop(
        stop_event=threading.Event(), fps=40.0, acquisition_frame_rate=48.0,
        device_key=1, group_key=1, group_mask=1,
        broadcast_address="255.255.255.255", timeout_ms=0,
    )
    assert all(trigger._slot_kind(s, s / 40.0, -np.inf) == "sync" for s in range(1, 20))
    assert "issued_sync" not in trigger.stats()


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"fast_group_mask": 0}, "fast_group_mask"),
        ({"fast_group_mask": 0x1}, "fast_group_mask"),          # 和 group_mask 重叠
        ({"sync_every": 2}, "slow cameras"),                     # 2/120 = 16.7 ms < 24.4 ms
        ({"acquisition_frame_rate": 118.0}, "acquisition_frame_rate"),
    ],
)
def test_high_speed_trigger_refuses_unworkable_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        _hs_trigger(**overrides)


class _FakeClock:
    """perf_counter 每读一次走 20 µs（模拟自旋），sleep 直接拨表。"""

    def __init__(self) -> None:
        self.t = 100.0

    def perf_counter(self) -> float:
        self.t += 20e-6
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _run_trigger(monkeypatch, trigger: ActionTriggerLoop, *, n_issues: int, stall_at=None):
    """在假时钟上跑真的 run()，返回 [(slot 序号推算, mask)]。stall_at={issue序号: 秒}。"""
    clock = _FakeClock()
    monkeypatch.setattr(ball_grabber, "time", types.SimpleNamespace(
        perf_counter=clock.perf_counter, sleep=clock.sleep))
    issued: list[tuple[float, int]] = []
    stall_at = stall_at or {}

    def fake_issue(mask: int) -> None:
        issued.append((clock.t, mask))
        clock.t += stall_at.get(len(issued), 0.0)
        if len(issued) >= n_issues:
            trigger._stop_event.set()

    monkeypatch.setattr(trigger, "_issue", fake_issue)
    trigger.run()
    return issued


def test_trigger_loop_emits_fast_fast_all_pattern(monkeypatch):
    trigger = _hs_trigger()
    issued = _run_trigger(monkeypatch, trigger, n_issues=12)
    masks = [m for _t, m in issued]
    assert masks == [HIGH_SPEED_GROUP_BIT, HIGH_SPEED_GROUP_BIT, 1] * 4
    gaps = np.diff([t for t, _m in issued])
    np.testing.assert_allclose(gaps, 1.0 / 120.0, atol=1e-4)
    stats = trigger.stats()
    assert stats["issued"] == 12 and stats["issued_sync"] == 4
    assert stats["skipped_slots"] == 0 and stats["sync_downgraded"] == 0


def test_trigger_loop_stall_skips_slots_but_keeps_the_grid(monkeypatch):
    """第 3 次触发（全组）卡住 20 ms：之后最早能发的是 25+20+7.79=52.8 ms → slot 7（50 ms 的
    slot 6 太近）。跳过的 4/5/6 计数（其中 6 是全组格），网格不平移，之后仍按 slot%3 发全组。"""
    trigger = _hs_trigger()
    issued = _run_trigger(monkeypatch, trigger, n_issues=8, stall_at={3: 0.020})
    t0 = issued[0][0] - 1.0 / 120.0           # slot 1 在 t0 + period
    slots = [round((t - t0) * 120.0) for t, _m in issued]
    assert slots == [1, 2, 3, 7, 8, 9, 10, 11]
    for slot, (_t, mask) in zip(slots, issued):
        assert mask == (1 if slot % 3 == 0 else HIGH_SPEED_GROUP_BIT)
    stats = trigger.stats()
    assert stats["skipped_slots"] == 3
    assert stats["skipped_sync_slots"] == 1
    assert stats["issued_sync"] == 2
    assert stats["late_over_2ms"] >= 1


def test_config_requires_every_key_and_sane_values():
    good = dict(serial="DB0260405", rate_multiplier=3, roi_offset_x=720, roi_offset_y=268,
                roi_width=1328, roi_height=680, acquisition_frame_rate=160.0)
    assert HighSpeedConfig.from_dict(good).roi_width == 1328
    for key in good:
        with pytest.raises(ValueError, match="缺"):
            HighSpeedConfig.from_dict({k: v for k, v in good.items() if k != key})
    with pytest.raises(ValueError, match="多"):
        HighSpeedConfig.from_dict({**good, "fps": 120})
    with pytest.raises(ValueError, match="rate_multiplier"):
        HighSpeedConfig.from_dict({**good, "rate_multiplier": 1})
    with pytest.raises(ValueError, match="偶数"):
        HighSpeedConfig.from_dict({**good, "roi_offset_x": 721})


def _bayer_frame(width: int, height: int, *, seed: int = 3, frame_num: int = 1,
                 exposure: float = 0.0) -> Frame:
    raw = np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)
    return Frame(data=raw.tobytes(), width=width, height=height, frame_num=frame_num,
                 pixel_type=PixelType_Gvsp_BayerRG8, exposure_start_pc=exposure)


def test_padded_frame_keeps_full_frame_pixel_coordinates():
    roi = _bayer_frame(16, 10)
    padded = pad_roi_frame(roi, canvas_width=40, canvas_height=30, offset_x=12, offset_y=6)
    assert (padded.width, padded.height) == (40, 30)
    canvas = np.frombuffer(padded.data, dtype=np.uint8).reshape(30, 40)
    np.testing.assert_array_equal(
        canvas[6:16, 12:28], np.frombuffer(roi.data, dtype=np.uint8).reshape(10, 16))
    canvas_outside = canvas.copy()
    canvas_outside[6:16, 12:28] = 0
    assert not canvas_outside.any()
    # 偶数偏移保住 Bayer 相位：贴回后解码的内部像素 = ROI 自己解码
    np.testing.assert_array_equal(
        frame_to_numpy(padded)[8:14, 14:26], frame_to_numpy(roi)[2:8, 2:14])


def test_padding_refuses_odd_offsets_overflow_and_colour_frames():
    roi = _bayer_frame(16, 10)
    with pytest.raises(ValueError, match="偶数"):
        pad_roi_frame(roi, canvas_width=40, canvas_height=30, offset_x=11, offset_y=6)
    with pytest.raises(ValueError, match="超出画布"):
        pad_roi_frame(roi, canvas_width=40, canvas_height=30, offset_x=30, offset_y=6)
    rgb = Frame(data=bytes(16 * 10 * 3), width=16, height=10, frame_num=1,
                pixel_type=PixelType_Gvsp_RGB8_Packed)
    with pytest.raises(ValueError, match="8-bit"):
        pad_roi_frame(rgb, canvas_width=40, canvas_height=30, offset_x=12, offset_y=6)


PERIOD = 1.0 / 120.0


def _stream_with(exposures) -> HighSpeedStream:
    stream = HighSpeedStream("DB0260405", slot_period_s=PERIOD)
    for i, t in enumerate(exposures):
        stream.push(_bayer_frame(8, 4, frame_num=100 + i, exposure=t))
    return stream


def test_stream_matches_the_same_trigger_not_the_neighbour():
    stream = _stream_with([10.0 - PERIOD, 10.0003, 10.0 + PERIOD])
    frame = stream.match(10.0, wait_s=0.0)
    assert frame is not None and frame.frame_num == 101
    assert stream.stats()["sync_matched"] == 1
    assert stream.sync_frame_nums == [101]


def test_stream_gives_up_at_once_when_the_sync_frame_was_lost():
    stream = _stream_with([10.0 - PERIOD, 10.0 + PERIOD])     # 同拍那帧丢了
    t0 = time.perf_counter()
    assert stream.match(10.0, wait_s=1.0) is None
    assert time.perf_counter() - t0 < 0.2                    # 有更晚的帧就不必等
    assert stream.stats()["sync_missed"] == 1


def test_stream_waits_for_a_frame_still_on_the_wire():
    stream = _stream_with([10.0 - PERIOD])
    pusher = threading.Timer(0.02, lambda: stream.push(_bayer_frame(8, 4, frame_num=7, exposure=10.0001)))
    pusher.start()
    frame = stream.match(10.0, wait_s=1.0)
    pusher.join()
    assert frame is not None and frame.frame_num == 7


def test_recorder_writes_raw_frames_index_and_metadata(tmp_path):
    recorder = HighSpeedRecorder(tmp_path / "run_hs_DB0260405", meta={"serial": "DB0260405"})
    assert json.loads(recorder.json_path.read_text(encoding="utf-8"))["status"] == "recording"
    recorder.start()
    frames = [_bayer_frame(8, 4, seed=i, frame_num=50 + i, exposure=1.0 + i * PERIOD) for i in range(5)]
    for f in frames:
        recorder.submit(f)
    stats = recorder.close(extra={"sync_frame_nums": [52]})
    assert stats["written"] == 5 and stats["dropped"] == 0 and stats["error"] is None
    raw = recorder.raw_path.read_bytes()
    assert raw == b"".join(f.data for f in frames)
    rec = struct.Struct(HIGH_SPEED_INDEX_FORMAT)
    index = recorder.index_path.read_bytes()
    assert len(index) == 5 * rec.size
    rows = [rec.unpack_from(index, i * rec.size) for i in range(5)]
    assert [r[0] for r in rows] == [50, 51, 52, 53, 54]
    np.testing.assert_allclose([r[2] for r in rows], [f.exposure_start_pc for f in frames])
    meta = json.loads(recorder.json_path.read_text(encoding="utf-8"))
    assert meta["status"] == "complete" and meta["recording"]["written"] == 5
    assert meta["sync_frame_nums"] == [52] and meta["files"]["raw"] == recorder.raw_path.name


def test_recorder_drops_instead_of_blocking_the_grabber(tmp_path):
    recorder = HighSpeedRecorder(tmp_path / "x", meta={}, queue_max=2)   # 不 start：没人消费
    for i in range(5):
        recorder.submit(_bayer_frame(8, 4, frame_num=i))
    assert recorder.dropped == 3


class _FakeGrabber:
    def __init__(self, frames):
        self._frames = list(frames)

    def _drain_keep_latest(self):
        return self._frames.pop(0) if self._frames else None

    def stop(self):
        pass

    def join(self, timeout=None):
        pass

    def stats(self):
        return {"received": 0, "missing": 0, "lost_packets": 0, "overwritten": 0}


def _capture_with_high_speed(slow_frames: dict[str, Frame], stream: HighSpeedStream) -> SyncCapture:
    """绕过 __init__（不碰相机）拼一个配了 405 高速的 SyncCapture。"""
    cap = object.__new__(SyncCapture)
    serials = list(slow_frames) + ["DB0260405"]
    cap._sync_serials = serials
    cap._sync_set = set(serials)
    cap._assembly_serials = list(slow_frames)
    cap._grabbers = {sn: _FakeGrabber([f]) for sn, f in slow_frames.items()}
    cap._grabbers["DB0260405"] = _FakeGrabber([])
    cap._sync_threshold_s = 0.2
    cap._exposure_spread_discards = 0
    cap._arrival_spread_discards = 0
    cap._assembly_timeouts = 0
    cap._high_speed = HighSpeedConfig.from_dict(dict(
        serial="DB0260405", rate_multiplier=3, roi_offset_x=4, roi_offset_y=2,
        roi_width=8, roi_height=4, acquisition_frame_rate=160.0))
    cap._hs_stream = stream
    cap._hs_info = {"canvas": {"width": 20, "height": 12},
                    "roi": {"offset_x": 4, "offset_y": 2, "width": 8, "height": 4}}
    return cap


def _slow_set(exposure: float) -> dict[str, Frame]:
    return {
        sn: Frame(data=bytes(20 * 12), width=20, height=12, frame_num=1,
                  pixel_type=PixelType_Gvsp_BayerRG8, exposure_start_pc=exposure + d,
                  arrival_perf=time.perf_counter())
        for sn, d in (("DB0260414", 0.0), ("DB0260373", 0.0002), ("DB0260378", -0.0001))
    }


def test_get_frames_attaches_the_synchronous_high_speed_frame_on_a_full_canvas():
    stream = _stream_with([10.0 - PERIOD, 10.00005, 10.0 + PERIOD])
    cap = _capture_with_high_speed(_slow_set(10.0), stream)
    frames = cap.get_frames(timeout_s=0.5)
    assert frames is not None and set(frames) == {"DB0260414", "DB0260373", "DB0260378", "DB0260405"}
    hs = frames["DB0260405"]
    assert hs.frame_num == 101 and (hs.width, hs.height) == (20, 12)
    canvas = np.frombuffer(hs.data, dtype=np.uint8).reshape(12, 20)
    expected_roi = np.frombuffer(_bayer_frame(8, 4, frame_num=101).data, dtype=np.uint8).reshape(4, 8)
    np.testing.assert_array_equal(canvas[2:6, 4:12], expected_roi)


def test_get_frames_puts_a_black_canvas_when_the_sync_frame_is_missing():
    stream = _stream_with([10.0 - PERIOD, 10.0 + PERIOD])
    cap = _capture_with_high_speed(_slow_set(10.0), stream)
    frames = cap.get_frames(timeout_s=0.5)
    hs = frames["DB0260405"]
    assert hs.frame_num == -1 and (hs.width, hs.height) == (20, 12)
    assert not any(hs.data)
    assert abs(hs.exposure_start_pc - 10.0) < 0.001


def test_close_keeps_the_final_stats_after_everything_is_torn_down(monkeypatch):
    cap = _capture_with_high_speed(_slow_set(10.0), _stream_with([]))
    cap._stop = threading.Event()
    cap._action_trigger = None
    cap._cameras = {}
    cap._hs_recorder = None
    cap._final_stats = None
    monkeypatch.setattr(ball_grabber, "MvCamera", types.SimpleNamespace(MV_CC_Finalize=lambda: 0))
    cap.close()
    cap.close()                                    # 第二次 close 不能把快照冲掉
    assert cap._grabbers == {}
    assert cap.stats()["high_speed"]["serial"] == "DB0260405"
    assert set(cap.stats()["per_camera"]) == {"DB0260414", "DB0260373", "DB0260378", "DB0260405"}


def _hs_trigger_params() -> dict:
    return dict(
        fps=120.0, acquisition_frame_rate=FAST_CEILING, device_key=1, group_key=1, group_mask=1,
        broadcast_address="255.255.255.255", timeout_ms=0, sync_every=3,
        fast_group_mask=HIGH_SPEED_GROUP_BIT, sync_minimum_period_s=1.0 / SLOW_CEILING + MARGIN,
        minimum_period_margin_s=MARGIN,
    )


def test_trigger_subprocess_runs_the_grid_and_reports_back():
    """子进程里跑真的 ActionTriggerLoop（dry_run：不往网上发 action 包），stdin 停、stdout 回统计。"""
    proc = ActionTriggerProcess(params=_hs_trigger_params(), dry_run=True)
    proc.start()
    assert proc.is_alive()
    time.sleep(0.5)
    proc.join(timeout=10.0)
    stats = proc.stats()
    assert not proc.is_alive()
    assert stats["process"] == "subprocess"
    assert stats["issued"] >= 40                        # 0.5 s × 120 Hz ≈ 60
    assert abs(stats["issued_sync"] - stats["issued"] / 3) <= 2
    assert stats["slot_fps"] == 120.0


def test_trigger_subprocess_rejects_bad_params_before_spawning():
    with pytest.raises(ValueError, match="slow cameras"):
        ActionTriggerProcess(params={**_hs_trigger_params(), "sync_every": 2})


class _ModelCamera:
    """按 1006 实测模型报 ResultingFrameRate：min(117.5e6/(W·H), 1/(H·8.0µs+0.94ms))。"""

    def __init__(self, **roi):
        self.nodes = {"OffsetX": 720, "OffsetY": 268, "Width": 1328, "Height": 680, **roi}
        self.width_min = 32

    def MV_CC_GetIntValueEx(self, node, val):
        val.nCurValue = self.nodes[node]
        val.nMin = self.width_min if node == "Width" else 0
        return 0

    def MV_CC_SetIntValue(self, node, value):
        self.nodes[node] = int(value)
        return 0

    def MV_CC_GetFloatValue(self, node, val):
        assert node == "ResultingFrameRate"
        w, h = self.nodes["Width"], self.nodes["Height"]
        val.fCurValue = min(117.5e6 / (w * h), 1.0 / (h * 8.0e-6 + 0.94e-3))
        return 0


def test_sensor_frame_rate_is_read_with_the_link_out_of_the_way_and_roi_restored():
    cam = _ModelCamera()
    assert ball_grabber.read_resulting_frame_rate(cam, "405") == pytest.approx(117.5e6 / (1328 * 680), rel=1e-5)
    assert ball_grabber.read_sensor_frame_rate(cam, "405") == pytest.approx(1 / (680 * 8.0e-6 + 0.94e-3), rel=1e-5)
    assert cam.nodes == {"OffsetX": 720, "OffsetY": 268, "Width": 1328, "Height": 680}


def test_sensor_frame_rate_refuses_to_leave_the_roi_changed():
    class Stuck(_ModelCamera):
        def MV_CC_SetIntValue(self, node, value):
            if node == "OffsetX" and value == 720:     # 还原偏移被相机拒
                return 0x80000102
            return super().MV_CC_SetIntValue(node, value)

    with pytest.raises(RuntimeError, match="没还原"):
        ball_grabber.read_sensor_frame_rate(Stuck(), "405")


def test_open_camera_moves_offset_x_after_shrinking_width(monkeypatch):
    calls = []

    class FakeCamera:
        def MV_CC_CreateHandle(self, _info):
            return 0

        def MV_CC_OpenDevice(self, *_args):
            return 0

        def MV_CC_SetEnumValueByString(self, node, value):
            return 0

        def MV_CC_SetBoolValue(self, node, value):
            return 0

        def MV_CC_SetFloatValue(self, node, value):
            return 0

        def MV_CC_SetIntValue(self, node, value):
            calls.append((node, value))
            return 0

        def MV_CC_SetImageNodeNum(self, n):
            calls.append(("ImageNodeNum", n))
            return 0

        def MV_CC_StartGrabbing(self):
            calls.append(("StartGrabbing", None))
            return 0

    device_info = ball_grabber.MV_CC_DEVICE_INFO()
    device_info.nTLayerType = ball_grabber.MV_USB_DEVICE
    device_list = ball_grabber.MV_CC_DEVICE_INFO_LIST()
    device_list.nDeviceNum = 1
    device_list.pDeviceInfo[0] = ball_grabber.ctypes.pointer(device_info)
    monkeypatch.setattr(ball_grabber, "MvCamera", FakeCamera)
    monkeypatch.setattr(ball_grabber, "_serial_of", lambda *_args: "camera")
    ball_grabber.open_camera("camera", trigger_source=None, _st_dev_list=device_list,
                             roi_offset_x=720, roi_width=1328, roi_offset_y=268, roi_height=680,
                             image_node_num=32)
    x_calls = [c for c in calls if c[0] in ("OffsetX", "Width")]
    y_calls = [c for c in calls if c[0] in ("OffsetY", "Height")]
    assert x_calls == [("OffsetX", 0), ("Width", 1328), ("OffsetX", 720)]
    assert y_calls == [("OffsetY", 0), ("Height", 680), ("OffsetY", 268)]
    # SDK 缓存节点数必须在起流之前设
    assert calls[-2:] == [("ImageNodeNum", 32), ("StartGrabbing", None)]
