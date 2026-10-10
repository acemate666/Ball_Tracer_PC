from __future__ import annotations

import pytest

from src import ball_grabber


def _open_with_fake_camera(monkeypatch, **kwargs):
    """用假相机跑一遍 open_camera，返回它写下去的节点调用序列。"""
    calls = []

    class FakeCamera:
        def MV_CC_CreateHandle(self, _device_info):
            return 0

        def MV_CC_OpenDevice(self, _access_mode, _switchover_key):
            return 0

        def MV_CC_SetEnumValueByString(self, node, value):
            calls.append(("enum", node, value))
            return 0

        def MV_CC_SetBoolValue(self, node, value):
            calls.append(("bool", node, value))
            return 0

        def MV_CC_SetFloatValue(self, node, value):
            calls.append(("float", node, value))
            return 0

        def MV_CC_StartGrabbing(self):
            return 0

    device_info = ball_grabber.MV_CC_DEVICE_INFO()
    device_info.nTLayerType = ball_grabber.MV_USB_DEVICE
    device_list = ball_grabber.MV_CC_DEVICE_INFO_LIST()
    device_list.nDeviceNum = 1
    device_list.pDeviceInfo[0] = ball_grabber.ctypes.pointer(device_info)
    monkeypatch.setattr(ball_grabber, "MvCamera", FakeCamera)
    monkeypatch.setattr(ball_grabber, "_serial_of", lambda *_args: "camera")

    ball_grabber.open_camera(
        "camera",
        trigger_source=None,
        _st_dev_list=device_list,
        **kwargs,
    )
    return calls


def test_reverse_is_always_written_and_defaults_to_18f(monkeypatch):
    """ReverseX/Y 是**相机里的持久状态**：调用方不传也必须写成 18F 的 False。

    以前不传就跳过写入，且环境变量默认还是 16F 的 True —— 直接开相机的脚本会继承
    上一个进程留下的翻转，tag 照常解码但位置整体转 180°（0906 实测车位姿飞到
    y=50m / 重投影 620px）。这条断言就是钉住「无条件写」这个契约。
    """
    assert ball_grabber._CAMERA_REVERSE_X is False
    assert ball_grabber._CAMERA_REVERSE_Y is False
    assert ball_grabber._ENV_SOFTWARE_ROTATE_180 is False

    calls = _open_with_fake_camera(monkeypatch)
    assert ("bool", "ReverseX", False) in calls
    assert ("bool", "ReverseY", False) in calls


def test_open_camera_enables_digital_shift_before_setting_value(monkeypatch):
    calls = _open_with_fake_camera(
        monkeypatch, digital_shift=-2.5, reverse_x=False, reverse_y=False)

    enable_call = ("bool", "DigitalShiftEnable", True)
    value_call = ("float", "DigitalShift", -2.5)
    assert calls.index(enable_call) < calls.index(value_call)


def test_camera_settings_reads_digital_shift_state_and_value():
    class FakeCamera:
        _values = {
            "ExposureTime": 9000.0,
            "Gain": 12.7,
            "DigitalShift": 2.0,
        }

        def MV_CC_GetBoolValue(self, node, value):
            assert node == "DigitalShiftEnable"
            value.value = True
            return 0

        def MV_CC_GetFloatValue(self, node, value):
            value.fCurValue = self._values[node]
            return 0

    capture = object.__new__(ball_grabber.SyncCapture)
    capture._cameras = {"camera": FakeCamera()}

    assert capture.camera_settings() == {
        "camera": {
            "digital_shift_enabled": True,
            "exposure_us": 9000.0,
            "gain_db": 12.7,
            "digital_shift": 2.0,
        }
    }


def test_auto_gain_fills_analog_first_then_digital_and_hits_target(capsys):
    """启动自动增益的契约：每台相机按自己画面亮度把灰度中位数拉到目标；
    总增益先用模拟 Gain（上限 12.78dB），超出部分给 DigitalShift，比 0dB 还亮就让 DigitalShift 走负；
    到了增益上限只告警不抛，且写下去的值不碰相机报的 fMax（正好写 fMax 会被拒）；
    高速相机贴回画布的帧只量 ROI（画布其余是 0）。
    """
    import math
    import time

    import numpy as np

    target = ball_grabber.AUTO_GAIN_TARGET_MEDIAN
    # 0dB 总增益下的画面灰度中位数 → 需要的总增益 20·log10(target/L0)
    scene_l0 = {
        "start_ok": target / 10 ** ((12.78 + 16.5) / 20),  # 正好是配置起点 29.28dB
        "dark": target / 10 ** (20.0 / 20),                 # 要 20dB：模拟顶满 + 数字 7.22
        "bright": target / 10 ** (6.0 / 20),                # 要 6dB：只用模拟
        "glare": target * 10,                               # 要 −20dB：模拟 0 + 数字 −20
        "too_dark": target / 10 ** (45.0 / 20),             # 要 45dB > 上限 12.78+23.99
        "hs": target / 10 ** (20.0 / 20),                   # 高速相机 ROI 内同 dark
    }
    size = 64
    rng = np.random.default_rng(0)
    base = {sn: rng.uniform(0.5, 1.5, (size, size)) * l0 for sn, l0 in scene_l0.items()}
    roi = {"offset_x": 16, "offset_y": 8, "width": 32, "height": 24}

    class FakeCamera:
        def __init__(self):
            self.values = {"Gain": 12.78, "DigitalShift": 16.5}

        def MV_CC_SetEnumValueByString(self, node, value):
            assert (node, value) == ("GainAuto", "Off")
            return 0

        def MV_CC_SetBoolValue(self, node, value):
            assert (node, value) == ("DigitalShiftEnable", True)
            return 0

        def MV_CC_GetFloatValue(self, node, value):
            value.fCurValue = self.values[node]
            # CS032 实测的上下限（float32）；写正好等于 fMax 的值相机会拒，auto_gain 得往里收
            value.fMin, value.fMax = {"Gain": (0.0, 12.7804), "DigitalShift": (-24.0824, 23.9998)}[node]
            return 0

        def MV_CC_SetFloatValue(self, node, value):
            self.values[node] = float(value)
            return 0

    cameras = {sn: FakeCamera() for sn in scene_l0}

    def fake_get_frames(timeout_s=1.0):
        frames = {}
        for sn, cam in cameras.items():
            scale = 10 ** ((cam.values["Gain"] + cam.values["DigitalShift"]) / 20)
            img = np.clip(base[sn] * scale, 0, 255).astype(np.uint8)
            if sn == "hs":
                canvas = np.zeros_like(img)
                canvas[roi["offset_y"]:roi["offset_y"] + roi["height"],
                       roi["offset_x"]:roi["offset_x"] + roi["width"]] = \
                    img[:roi["height"], :roi["width"]]
                img = canvas
            frames[sn] = ball_grabber.Frame(
                data=img.tobytes(), width=size, height=size, frame_num=1, pixel_type=0,
                exposure_start_pc=time.perf_counter() + 10.0,
            )
        return frames

    capture = object.__new__(ball_grabber.SyncCapture)
    capture._fps = 40.0
    capture._sync_serials = list(cameras)
    capture._cameras = cameras
    capture._high_speed = type("HS", (), {"serial": "hs"})()
    capture._hs_info = {"roi": roi}
    capture.get_frames = fake_get_frames

    capture.auto_gain()

    def total(sn):
        return cameras[sn].values["Gain"] + cameras[sn].values["DigitalShift"]

    def final_median(sn):
        img = np.frombuffer(fake_get_frames()[sn].data, dtype=np.uint8).reshape(size, size)
        if sn == "hs":
            img = img[roi["offset_y"]:roi["offset_y"] + roi["height"],
                      roi["offset_x"]:roi["offset_x"] + roi["width"]]
        return float(np.median(img[::4, ::4]))

    for sn in ("start_ok", "dark", "bright", "glare", "hs"):
        err_db = 20 * math.log10(target / final_median(sn))
        assert abs(err_db) <= 1.0, (sn, final_median(sn))
        gain = cameras[sn].values["Gain"]
        assert gain == pytest.approx(min(max(total(sn), 0.0), 12.78), abs=1e-9), (sn, cameras[sn].values)

    assert cameras["start_ok"].values == {"Gain": 12.78, "DigitalShift": 16.5}  # 起点已到位不动
    assert cameras["bright"].values["DigitalShift"] == 0.0
    assert cameras["glare"].values["Gain"] == 0.0 and cameras["glare"].values["DigitalShift"] < -18.0
    assert cameras["dark"].values["Gain"] == pytest.approx(12.78, abs=1e-9)
    assert cameras["hs"].values["Gain"] == pytest.approx(12.78, abs=1e-9)
    assert cameras["too_dark"].values == pytest.approx({"Gain": 12.78, "DigitalShift": 23.99}, abs=1e-9)

    out = capsys.readouterr().out
    warned = [line for line in out.splitlines() if "⚠" in line]
    assert len(warned) == 1 and "too_dark" in warned[0]
