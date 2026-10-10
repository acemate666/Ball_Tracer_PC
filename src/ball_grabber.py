# -*- coding: utf-8 -*-
"""
海康威视多相机同步采集模块 (ball_grabber)。

═══════════════════════════  系统方案概述  ═══════════════════════════

硬件（18F，唯一在用的场地）：
  - 相机型号：海康 MV-CS032-60GC (GigE Vision, 320万像素, BayerRG8)
  - GigE Action Command 广播同步触发四台相机
  - 相机正装，ReverseX/Y 恒 False（见 _CAMERA_REVERSE_X/Y）

相机参数：
  - config/camera_18.json —— 曝光、增益、帧率、触发、底部裁剪只在这里维护
  - tracker 开相机后调 SyncCapture.auto_gain() 按画面亮度把增益定一次（配置里的增益只是起点）

时间同步：
  - 每台相机的 ImageGrabber 启动时校准 PC↔设备时钟偏移 (GevTimestampControlLatch)
  - 后台线程按相机配置周期性重校准，偏移跳变 >5ms 自动忽略
  - 每帧按相机报告的 tick frequency 计算 exposure_start_pc (perf_counter 时间轴)

同步取图流程 (SyncCapture.get_frames)：
  1. 各 ImageGrabber 后台线程持续取帧入队（bounded deque, max=10）
  2. get_frames() 排空所有 grabber 队列，仅保留各相机最新帧
  3. 检查 arrival_perf spread ≤ 200ms（粗同步）
  4. 硬触发模式额外检查 exposure_start_pc spread ≤ 10ms（确保同一触发周期）
  5. 返回 {serial: Frame} 字典，或超时返回 None

高速相机（相机配置 high_speed_camera，可选）：
  - 一台相机裁小 ROI、按 N 倍帧率跑（如 405 1328×680 @120 = 3×40），其余相机不动
  - 靠 action 组掩码分频：高速相机认 group_mask|0x2，触发环每格发 0x2、每 N 格发全组；
    触发环在独立子进程里跑（ActionTriggerProcess），不跟 tracker 的线程抢 GIL
  - 它的每一帧经 ImageGrabber.frame_sink 进 HighSpeedStream（不走"只留最新"的排空），
    原样写盘（HighSpeedRecorder）；与其余相机同拍的那一帧按曝光时刻配进同步组，
    贴回全幅画布（pad_roi_frame），下游像素坐标与标定逐点一致

像素格式：
  - 海康 Bayer 命名与 OpenCV 相反：BayerRG8 → cv2.COLOR_BayerBG2BGR
  - frame_to_numpy() 自动处理 Mono8 / BayerXX8 / RGB8 / BGR8

模块结构：
  - open_camera()    打开单台相机并 StartGrabbing
  - close_camera()   停止取流、关闭设备、销毁句柄
  - frame_to_numpy() Frame 原始数据 → numpy BGR/Mono（旋转由启动入口设置）
  - ImageGrabber     后台取帧线程（带时间偏移校准）
  - SyncCapture      多相机同步采集上下文管理器
  - SyncCapture.from_config()  从 config/camera_18.json 创建
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

# 环境初始化（MvImport + DLL 路径），必须在 import SDK 之前
import src.mvs_env as _env  # noqa: F401

from MvCameraControl_class import MvCamera
from CameraParams_header import (
    MV_ACTION_CMD_INFO,
    MV_ACTION_CMD_RESULT_LIST,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_DEVICE_INFO,
    MV_FRAME_OUT,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE_EX,
)
from PixelType_header import (
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_RGB8_Packed,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerGB8,
)
from ctypes import cast, POINTER, byref, sizeof, memset


# ───────────────── 工具函数 ─────────────────


def _env_optional_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} 不是合法布尔值")


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_optional_bool(name)
    return default if raw is None else raw


# 18F 四相机正装，ReverseX/Y 恒为 False。这两个是**相机里的持久状态**，之前只有显式
# 传值或设了环境变量才写，且环境变量的默认值还是 16F 时代的 True —— 任何不经启动脚本
# 直接开相机的工具（标定采集、临时脚本）都会把四台相机翻成 180°，而且留在相机里。
# 0906 实测这种状态下 tag 照常解码（AprilTag 旋转不变）但位置整体转半圈，车位姿飞到
# y=50m / 重投影 620px。现在无条件按这里写死，不再留环境变量后门。
_CAMERA_REVERSE_X = False
_CAMERA_REVERSE_Y = False
_ENV_SOFTWARE_ROTATE_180 = _env_bool("BALL_TRACER_SOFTWARE_ROTATE_180", False)


def _decode(buf) -> str:
    try:
        b = memoryview(buf).tobytes()
        i = b.find(b"\x00")
        if i >= 0:
            b = b[:i]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _serial_of(dev_info, tlayer: int) -> str:
    if tlayer == MV_GIGE_DEVICE:
        return _decode(dev_info.SpecialInfo.stGigEInfo.chSerialNumber)
    if tlayer == MV_USB_DEVICE:
        return _decode(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
    return ""


def _model_of(dev_info, tlayer: int) -> str:
    if tlayer == MV_GIGE_DEVICE:
        return _decode(dev_info.SpecialInfo.stGigEInfo.chModelName)
    if tlayer == MV_USB_DEVICE:
        return _decode(dev_info.SpecialInfo.stUsb3VInfo.chModelName)
    return "Unknown"


def _check(ret: int, what: str) -> None:
    if int(ret) != 0:
        raise RuntimeError(f"{what} failed, ret=0x{int(ret):08X}")


# ───────────────── 数据结构 ─────────────────


@dataclass
class Frame:
    """单帧图像数据。"""
    data: bytes
    width: int
    height: int
    frame_num: int
    pixel_type: int
    dev_timestamp: int = 0       # 设备硬件时间戳 tick（nDevTimeStampHigh<<32 | nDevTimeStampLow）
    host_timestamp: int = 0      # SDK 主机时间戳（nHostTimeStamp, ms）
    exposure_time: float = 0.0   # 曝光时间 μs（fExposureTime）
    lost_packet: int = 0         # 本帧丢包数（nLostPacket）
    arrival_perf: float = 0.0    # time.perf_counter() 到达时刻
    exposure_start_pc: float = 0.0  # 曝光开始时刻（PC perf_counter 时间轴，秒）


# ───────────────── 时间校准 ─────────────────


# GevTimestampTickFrequency，MV-CS050-10GC 为 100 MHz
def read_tick_frequency(cam: Any, serial: str) -> int:
    val = MVCC_INTVALUE_EX()
    ret = cam.MV_CC_GetIntValueEx("GevTimestampTickFrequency", val)
    _check(ret, f"GevTimestampTickFrequency({serial})")
    freq = int(val.nCurValue)
    if freq <= 0:
        raise RuntimeError(f"GevTimestampTickFrequency({serial}) returned {freq}")
    return freq


def calibrate_time_offset(cam: Any, tick_frequency_hz: int) -> float:
    """
    校准 PC perf_counter 时间与相机设备时间的偏移量。

    通过 GevTimestampControlLatch 锁存设备时间戳并与 PC 时间比对。
    返回 offset（秒），使得:
        曝光时刻_perf = dev_timestamp / TICK_FREQ + offset

    为减少网络延迟误差，取 3 次中往返最短的一次。
    """
    best_offset = 0.0
    best_rtt = float("inf")

    for _ in range(3):
        t_before = time.perf_counter()
        cam.MV_CC_SetCommandValue("GevTimestampControlLatch")
        t_after = time.perf_counter()

        val = MVCC_INTVALUE_EX()
        ret = cam.MV_CC_GetIntValueEx("GevTimestampValue", val)
        if ret != 0:
            continue

        rtt = t_after - t_before
        pc_time = (t_before + t_after) / 2.0
        dev_time = int(val.nCurValue) / tick_frequency_hz
        offset = pc_time - dev_time

        if rtt < best_rtt:
            best_rtt = rtt
            best_offset = offset

    return best_offset


# ───────────────── 像素转换 ─────────────────


def frame_to_numpy(frame: Frame, *, rotate_180: Optional[bool] = None):
    """将 Frame 转为 numpy BGR 图像。默认做 180° 旋转，支持 Mono8、BayerXX8、RGB8、BGR8。"""
    import cv2
    import numpy as np

    if rotate_180 is None:
        rotate_180 = _ENV_SOFTWARE_ROTATE_180

    pt = frame.pixel_type
    w, h = frame.width, frame.height

    # Mono8
    if pt == PixelType_Gvsp_Mono8:
        img = np.frombuffer(frame.data, dtype=np.uint8, count=w * h).reshape(h, w)
        if rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # BGR8 — 直接 reshape
    if pt == PixelType_Gvsp_BGR8_Packed:
        img = np.frombuffer(frame.data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        return cv2.rotate(img, cv2.ROTATE_180) if rotate_180 else img

    # RGB8 → BGR
    if pt == PixelType_Gvsp_RGB8_Packed:
        rgb = np.frombuffer(frame.data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.rotate(img, cv2.ROTATE_180) if rotate_180 else img

    # Bayer 8-bit → BGR
    # 海康 GigE Vision 的 Bayer 命名与 OpenCV 相反（RG↔BG, GR↔GB）。
    # 若最终需要 180° 旋转，先旋转 raw Bayer 再解码，可避免对 3 通道 BGR 再做一次大内存旋转。
    bayer_codes = {
        PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
        PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
        PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
        PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
    }
    rotated_bayer_codes = {
        PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerRG2BGR,
        PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerBG2BGR,
        PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGR2BGR,
        PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGB2BGR,
    }
    if pt in bayer_codes:
        raw = np.frombuffer(frame.data, dtype=np.uint8, count=w * h).reshape(h, w)
        if rotate_180:
            raw = cv2.rotate(raw, cv2.ROTATE_180)
            return cv2.cvtColor(raw, rotated_bayer_codes[pt])
        return cv2.cvtColor(raw, bayer_codes[pt])

    raise RuntimeError(f"不支持的像素格式 0x{pt:08X}，请在 MVS Client 中切换为 Mono8 或 BayerXX8")


def frame_bayer_roi_to_numpy(
    frame: Frame,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    resize_to: int,
):
    """直接从 Bayer 原始帧裁剪 ROI，解码并缩放为 YOLO 输入。"""
    import cv2
    import numpy as np

    if _ENV_SOFTWARE_ROTATE_180:
        raise RuntimeError("Bayer ROI fast path requires camera-side orientation")

    bayer_codes = {
        PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
        PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
        PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
        PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
    }
    if frame.pixel_type not in bayer_codes:
        raise RuntimeError(
            f"YOLO ROI fast path requires Bayer8, got 0x{frame.pixel_type:08X}"
        )

    x0 = x & ~1
    y0 = y & ~1
    dx = x - x0
    dy = y - y0
    raw = np.frombuffer(
        frame.data,
        dtype=np.uint8,
        count=frame.width * frame.height,
    ).reshape(frame.height, frame.width)
    raw_roi = raw[y0:y + height, x0:x + width]
    bgr_roi = cv2.cvtColor(raw_roi, bayer_codes[frame.pixel_type])
    bgr_roi = bgr_roi[dy:dy + height, dx:dx + width]
    return cv2.resize(bgr_roi, (resize_to, resize_to))


# ───────────────── 设备枚举 ─────────────────


def _enum_devices():
    """返回 (st_dev_list, [(index, serial, model, tlayer), ...])。不做 Initialize/Finalize。"""
    st_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_list)
    if ret != 0:
        raise RuntimeError(f"枚举设备失败 ret=0x{ret:08X}")
    devs = []
    for i in range(int(st_list.nDeviceNum)):
        info = cast(st_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        t = int(info.nTLayerType)
        devs.append((i, _serial_of(info, t), _model_of(info, t), t))
    return st_list, devs


def list_devices() -> list[tuple[int, str, str]]:
    """枚举设备，返回 [(index, serial, model), ...]。"""
    MvCamera.MV_CC_Initialize()
    try:
        _, devs = _enum_devices()
        return [(i, s, m) for i, s, m, _t in devs]
    finally:
        MvCamera.MV_CC_Finalize()


# ───────────────── 相机打开 / 关闭 ─────────────────


def open_camera(
    serial: str,
    *,
    trigger_source: Optional[str] = "Software",
    trigger_activation: str = "FallingEdge",
    action_device_key: int = 1,
    action_group_key: int = 1,
    action_group_mask: int = 1,
    line_output: str = "",
    line_source: str = "",
    frame_rate: float = 0.0,
    full_frame: bool = False,
    exposure_us: float = 0.0,
    gain_db: float = -1.0,
    digital_shift: Optional[float] = None,
    pixel_format: str = "",
    roi_offset_x: int = 0,
    roi_offset_y: int = 0,
    roi_height: int = 0,
    roi_width: int = 0,
    binning: int = 1,
    gev_scpd: int = -1,
    reverse_x: Optional[bool] = None,
    reverse_y: Optional[bool] = None,
    # 广播触发的频率（不是 AcquisitionFrameRate）。给了就在开场校验传感器给得起。
    trigger_fps: float = 0.0,
    # SDK 内部取流缓存节点数（0 = SDK 默认）。取流线程抢不到 GIL 时靠它攒几帧不丢
    image_node_num: int = 0,
    _st_dev_list=None,
) -> Any:
    """
    打开指定序列号的相机并 StartGrabbing。

    Args:
        serial: 相机序列号。
        trigger_source: 触发源 — "Software"/"Line0" 等；None 表示 free-run 连续采集。
        trigger_activation: 硬触发沿 — "FallingEdge" / "RisingEdge"（软触发时忽略）。
        line_output: 主相机输出线（如 "Line1"），用于输出曝光信号触发从机。空字符串跳过。
        line_source: 输出源（如 "ExposureStartActive"）。与 line_output 配合使用。
        frame_rate: 帧率（fps），>0 时设置 AcquisitionFrameRate（仅 free-run 有效）。
        full_frame: True 时重置 ROI 为传感器最大分辨率。
        _st_dev_list: 内部复用，避免重复枚举。

    Returns:
        MvCamera 实例（已 StartGrabbing）。
    """
    serial = str(serial).strip()

    if reverse_x is None:
        reverse_x = _CAMERA_REVERSE_X
    if reverse_y is None:
        reverse_y = _CAMERA_REVERSE_Y

    if _st_dev_list is None:
        st_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_list)
        _check(ret, "EnumDevices")
    else:
        st_list = _st_dev_list

    for i in range(int(st_list.nDeviceNum)):
        dev_info = cast(st_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        tlayer = int(dev_info.nTLayerType)
        if _serial_of(dev_info, tlayer) != serial:
            continue

        cam = MvCamera()
        _check(cam.MV_CC_CreateHandle(dev_info), f"CreateHandle({serial})")

        ret = cam.MV_CC_OpenDevice(3, 0)  # MV_ACCESS_Control
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"OpenDevice({serial}) 失败 ret=0x{ret:08X}")

        # GigE 网络优化
        if tlayer == MV_GIGE_DEVICE:
            try:
                ps = cam.MV_CC_GetOptimalPacketSize()
                if int(ps) > 0:
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", ps)
            except Exception:
                pass
            # 丢包重传（参考 mvs-dev/camera.py _best_effort_gige_network_tuning）
            try:
                cam.MV_GIGE_SetResend(True, 100, 50)
            except Exception:
                pass
            if int(gev_scpd) >= 0:
                _check(cam.MV_CC_SetIntValueEx("GevSCPD", int(gev_scpd)), f"GevSCPD={gev_scpd}")

        # ── Binning（像素合并，降低分辨率但保持 FOV）──
        # 必须在 full_frame 之前设置，因为 binning 会改变 WidthMax/HeightMax
        if binning > 1:
            # 先缩小 Width/Height 以免超出 binning 后的最大值
            try:
                cam.MV_CC_SetIntValue("OffsetX", 0)
                cam.MV_CC_SetIntValue("OffsetY", 0)
                cam.MV_CC_SetIntValue("Width", 16)
                cam.MV_CC_SetIntValue("Height", 16)
            except Exception:
                pass
            try:
                cam.MV_CC_SetEnumValueByString("BinningHorizontalMode", "Average")
            except Exception:
                pass
            try:
                cam.MV_CC_SetEnumValueByString("BinningVerticalMode", "Average")
            except Exception:
                pass
            try:
                cam.MV_CC_SetIntValue("BinningHorizontal", binning)
            except Exception:
                pass
            try:
                cam.MV_CC_SetIntValue("BinningVertical", binning)
            except Exception:
                pass

        # ── 全画幅 ROI 重置（binning 后的最大分辨率）──
        if full_frame:
            try:
                cam.MV_CC_SetIntValue("OffsetX", 0)
                cam.MV_CC_SetIntValue("OffsetY", 0)
                from CameraParams_header import MVCC_INTVALUE
                w_param = MVCC_INTVALUE()
                cam.MV_CC_GetIntValue("WidthMax", w_param)
                cam.MV_CC_SetIntValue("Width", int(w_param.nCurValue))
                h_param = MVCC_INTVALUE()
                cam.MV_CC_GetIntValue("HeightMax", h_param)
                cam.MV_CC_SetIntValue("Height", int(h_param.nCurValue))
            except Exception:
                pass

        # ── 触发配置 ──
        if trigger_source is None:
            # Free-run 连续采集（关闭触发模式）
            _check(cam.MV_CC_SetEnumValueByString("TriggerMode", "Off"), "TriggerMode=Off")
        else:
            # 触发模式（Software / Line0 等）
            _check(cam.MV_CC_SetEnumValueByString("TriggerMode", "On"), "TriggerMode")
            _check(cam.MV_CC_SetEnumValueByString("TriggerSource", trigger_source), f"TriggerSource={trigger_source}")
            trigger_source_lower = trigger_source.lower()
            if trigger_source_lower.startswith("action"):
                _check(
                    cam.MV_CC_SetIntValueEx("ActionDeviceKey", int(action_device_key)),
                    f"ActionDeviceKey={action_device_key}",
                )
                _check(
                    cam.MV_CC_SetIntValueEx("ActionGroupKey", int(action_group_key)),
                    f"ActionGroupKey={action_group_key}",
                )
                _check(
                    cam.MV_CC_SetIntValueEx("ActionGroupMask", int(action_group_mask)),
                    f"ActionGroupMask={action_group_mask}",
                )

            # 硬触发才需要设置触发沿 + 输入线配置
            if trigger_source_lower not in ("software", "triggersoftware") and not trigger_source_lower.startswith("action"):
                try:
                    cam.MV_CC_SetEnumValueByString("TriggerActivation", trigger_activation)
                except Exception:
                    pass
                # 选择输入线并设置滤波时间（防误触发）
                try:
                    cam.MV_CC_SetEnumValueByString("LineSelector", trigger_source)
                    cam.MV_CC_SetIntValueEx("LineDebouncerTime", 50)
                except Exception:
                    pass

            try:
                cam.MV_CC_SetFloatValue("TriggerDelay", 0.0)
            except Exception:
                pass
            try:
                cam.MV_CC_SetBoolValue("TriggerCacheEnable", False)
            except Exception:
                pass

        # ── 主相机 Line 输出（参考 MVS SDK ParametrizeCamera_AreaScanIOSettings）──
        if frame_rate > 0:
            _check(cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True), "AcquisitionFrameRateEnable=True")
            _check(cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate)), f"AcquisitionFrameRate={frame_rate}")

        if line_output and line_source:
            try:
                cam.MV_CC_SetEnumValueByString("LineSelector", line_output)
                # 尝试 Output / Strobe
                ret2 = cam.MV_CC_SetEnumValueByString("LineMode", "Output")
                if int(ret2) != 0:
                    cam.MV_CC_SetEnumValueByString("LineMode", "Strobe")
                # 显式清零残留 strobe 参数，避免机内持久化状态引入不确定输出。
                try:
                    cam.MV_CC_SetBoolValue("LineInverter", False)
                except Exception:
                    pass
                cam.MV_CC_SetEnumValueByString("LineSource", line_source)
                for node_name in ("StrobeLineDuration", "StrobeLineDelay", "StrobeLinePreDelay"):
                    try:
                        cam.MV_CC_SetIntValueEx(node_name, 0)
                    except Exception:
                        pass
                # 开启输出使能（StrobeEnable）
                cam.MV_CC_SetBoolValue("StrobeEnable", True)
            except Exception:
                pass

        if pixel_format:
            _check(
                cam.MV_CC_SetEnumValueByString("PixelFormat", pixel_format),
                f"PixelFormat={pixel_format}",
            )

        # ── 曝光 / 模拟增益 / 数字增益 ──
        # SetFloatValue 越界不抛异常、只返回错误码（如 0x80000102）。以前这里连返回值都不看，
        # 于是"配置里写了但相机没接受"完全无声：MV-CS050 的 Gain 上限只有 12.78dB，
        # 写 20 会被静默丢弃、实际仍跑在旧增益上。超界只告警不抛，避免因一个参数写不进去
        # 就废掉整场采集，但必须让操作者看见。
        def _try_set_float(node: str, value: float, unit: str) -> None:
            ret = cam.MV_CC_SetFloatValue(node, float(value))
            if int(ret) == 0:
                return
            rng = ""
            try:
                cur = MVCC_FLOATVALUE()
                if int(cam.MV_CC_GetFloatValue(node, cur)) == 0:
                    rng = (f"；相机允许 [{cur.fMin:.2f}, {cur.fMax:.2f}]{unit}"
                           f"，当前仍为 {cur.fCurValue:.2f}{unit}")
            except Exception:
                pass
            print(f"[相机 {serial}] 警告: {node}={value:g}{unit} 未被接受 "
                  f"(ret=0x{int(ret) & 0xFFFFFFFF:08x}){rng}")

        if exposure_us > 0:
            try:
                cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            except Exception:
                pass
            _try_set_float("ExposureTime", exposure_us, "μs")

        if gain_db >= 0:
            try:
                cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            except Exception:
                pass
            _try_set_float("Gain", gain_db, "dB")

        if digital_shift is not None:
            ret = cam.MV_CC_SetBoolValue("DigitalShiftEnable", True)
            if int(ret) != 0:
                print(f"[相机 {serial}] 警告: DigitalShiftEnable=True 未被接受 "
                      f"(ret=0x{int(ret) & 0xFFFFFFFF:08x})")
            _try_set_float("DigitalShift", digital_shift, "")

        # ── 自定义 ROI（X 轴）── 先归零 OffsetX 再缩 Width，最后才挪 OffsetX（同 Y 轴）
        if roi_offset_x > 0 or roi_width > 0:
            try:
                cam.MV_CC_SetIntValue("OffsetX", 0)
                if roi_width > 0:
                    cam.MV_CC_SetIntValue("Width", roi_width)
                if roi_offset_x > 0:
                    cam.MV_CC_SetIntValue("OffsetX", roi_offset_x)
            except Exception:
                pass

        # ── 自定义 ROI（Y 轴）──
        if roi_offset_y > 0 or roi_height > 0:
            try:
                cam.MV_CC_SetIntValue("OffsetY", 0)
                if roi_height > 0:
                    cam.MV_CC_SetIntValue("Height", roi_height)
                elif roi_offset_y > 0:
                    from CameraParams_header import MVCC_INTVALUE as _IV
                    h_max = _IV()
                    cam.MV_CC_GetIntValue("HeightMax", h_max)
                    cam.MV_CC_SetIntValue("Height", int(h_max.nCurValue) - roi_offset_y)
                if roi_offset_y > 0:
                    cam.MV_CC_SetIntValue("OffsetY", roi_offset_y)
            except Exception:
                pass

        # ROI 定下来之后读回传感器真正能给的帧率上限。压 ROI 冲帧率时这是唯一真值：
        # 触发再准，ResultingFrameRate 低于触发频率就一定按比例丢帧。
        if frame_rate > 0:
            resulting = MVCC_FLOATVALUE()
            if int(cam.MV_CC_GetFloatValue("ResultingFrameRate", resulting)) == 0:
                ceiling = float(resulting.fCurValue)
                note = "" if ceiling >= frame_rate - 0.5 else "  ← 低于 AcquisitionFrameRate"
                print(
                    f"[camera {serial}] ROI 高 {roi_height or 'full'} 行，"
                    f"AcquisitionFrameRate={frame_rate:g} → "
                    f"ResultingFrameRate={ceiling:.2f} fps{note}"
                )
                # 触发频率高于传感器上限是**负收益**，不是「拿到上限那么多」：0906 实测
                # ROI1304（上限 44.0）下按 60Hz 触发，相机只吐 30.6fps/台，比按 40Hz 触发
                # 拿到的 39.9 还差。落在读出窗口里的触发被整个吞掉，且不进任何计数器。
                # 这种「配得越高跑得越慢」的坑必须在开场就拦，不能等跑完看帧率。
                if trigger_fps > 0 and ceiling > 0 and trigger_fps > ceiling + 0.5:
                    raise RuntimeError(
                        f"相机 {serial} 在 ROI 高 {roi_height or 'full'} 行下最高只能给 "
                        f"{ceiling:.2f} fps，而触发频率配的是 {trigger_fps:g} fps。"
                        f"超上限触发只会更慢：把 camera 配置的 fps 降到 ≤{ceiling:.1f}，"
                        f"或把 roi_height 压到 ≤{int(117.5e6 / (trigger_fps * 2048))} 行。"
                    )

        # 相机侧 180° 反转必须在 StartGrabbing 之前设置。
        _check(cam.MV_CC_SetBoolValue("ReverseX", bool(reverse_x)), f"ReverseX={reverse_x}")
        _check(cam.MV_CC_SetBoolValue("ReverseY", bool(reverse_y)), f"ReverseY={reverse_y}")

        if image_node_num > 0:
            _check(cam.MV_CC_SetImageNodeNum(int(image_node_num)), f"SetImageNodeNum({image_node_num})")

        _check(cam.MV_CC_StartGrabbing(), f"StartGrabbing({serial})")
        return cam

    raise RuntimeError(f"未找到序列号为 {serial} 的相机")


def close_camera(cam: Any) -> None:
    """停止取流、关闭设备、销毁句柄。"""
    try:
        cam.MV_CC_StopGrabbing()
    except Exception:
        pass
    try:
        cam.MV_CC_CloseDevice()
    except Exception:
        pass
    try:
        cam.MV_CC_DestroyHandle()
    except Exception:
        pass


def read_resulting_frame_rate(cam: Any, serial: str) -> float:
    """相机按当前 ROI/曝光/链路算出的帧率上限（ResultingFrameRate）。

    18F MV-CS032-60GC 实测 = min(117.5e6/(W·H), 1/(H·8.0µs+0.94ms))：全宽时卡在单台
    1GbE，裁窄后才轮到传感器行读出（2026-10-06 对 405 逐 ROI 读回）。
    """
    val = MVCC_FLOATVALUE()
    _check(cam.MV_CC_GetFloatValue("ResultingFrameRate", val), f"ResultingFrameRate({serial})")
    fps = float(val.fCurValue)
    if fps <= 0:
        raise RuntimeError(f"ResultingFrameRate({serial}) returned {fps}")
    return fps


def read_roi(cam: Any, serial: str) -> dict[str, int]:
    """读回相机实际生效的 ROI（OffsetX/OffsetY/Width/Height）。"""
    out: dict[str, int] = {}
    for node in ("OffsetX", "OffsetY", "Width", "Height"):
        val = MVCC_INTVALUE_EX()
        _check(cam.MV_CC_GetIntValueEx(node, val), f"{node}({serial})")
        out[node] = int(val.nCurValue)
    return out


def read_sensor_frame_rate(cam: Any, serial: str) -> float:
    """当前 ROI 高度下**传感器自己**的帧率上限：宽度临时压到最小、链路不再卡，读回
    ResultingFrameRate，再还原 OffsetX/Width。只能在停流时调（改 Width 要停流）。

    相机按它决定一次触发收不收：ResultingFrameRate 只是单台 1GbE 的**平均**上限，单次
    两帧挨得近一点，只要传感器读得出来相机就先缓存着慢慢发。2026-10-06 ASUS 真 tracker：
    405 按链路上限 130.07 fps 排触发，每格迟到 >0.55 ms 就跳，60 s 跳 593~1454 格；改按传感器
    156.8 fps 排，0 跳格、相机照单全收。
    """
    roi = read_roi(cam, serial)
    width = MVCC_INTVALUE_EX()
    _check(cam.MV_CC_GetIntValueEx("Width", width), f"Width({serial})")
    try:
        _check(cam.MV_CC_SetIntValue("OffsetX", 0), f"OffsetX=0({serial})")
        _check(cam.MV_CC_SetIntValue("Width", int(width.nMin)), f"Width={width.nMin}({serial})")
        fps = read_resulting_frame_rate(cam, serial)
    finally:
        cam.MV_CC_SetIntValue("Width", roi["Width"])
        cam.MV_CC_SetIntValue("OffsetX", roi["OffsetX"])
    if read_roi(cam, serial) != roi:
        raise RuntimeError(f"相机 {serial} 量完传感器帧率后 ROI 没还原：要 {roi}，实际 {read_roi(cam, serial)}")
    return fps


# ───────────────── Grabber ─────────────────


class ImageGrabber(threading.Thread):
    """
    单相机抓取器：后台取流入队（最大 10 张），提供 get_frame() 取图出队。

    自动校准 PC 与设备时钟偏移，每帧计算 exposure_start_pc（PC perf_counter 时间轴）。
    周期性校准在独立后台线程执行，不阻塞取帧热循环。
    """

    MAX_QUEUE = 10

    def __init__(
        self,
        cam: Any,
        serial: str = "",
        *,
        timeout_ms: int = 1000,
        recalib_every: int = 10,
        fps: float = 0.0,
        tick_frequency_hz: int = 0,
        # 给了就每帧直接交给它、不进队列（高速相机：一帧都不能被"只留最新"丢掉）
        frame_sink: Optional[Callable[[Frame], None]] = None,
    ):
        super().__init__(daemon=True, name=f"Grabber-{serial}")
        self._cam = cam
        self.serial = serial
        self._timeout_ms = int(timeout_ms)
        self._frame_sink = frame_sink
        self._queue: deque[Frame] = deque(maxlen=self.MAX_QUEUE)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recalib_every = recalib_every
        self._fps = float(fps)
        self._tick_frequency_hz = int(tick_frequency_hz) if int(tick_frequency_hz) > 0 else read_tick_frequency(cam, serial)
        self._time_offset: float = 0.0       # PC perf_counter - dev_time（秒）
        self._calib_thread: Optional[threading.Thread] = None
        # 丢帧归因计数：没有这几个数就分不清帧丢在网线上、SDK 里还是 Python 里。
        # frames_missing 用相机自己的 nFrameNum 缺口算（相机侧/传输侧丢），
        # frames_overwritten 是主机侧来不及消费被覆盖掉的（见 _drain_keep_latest）。
        self.frames_received = 0
        self.frames_missing = 0
        self.lost_packets = 0
        self.frames_overwritten = 0
        self.sink_errors = 0
        self._last_frame_num: Optional[int] = None

    def run(self) -> None:
        # 初始校准（取帧前同步执行）
        self._time_offset = calibrate_time_offset(self._cam, self._tick_frequency_hz)

        # 启动后台校准线程
        if self._recalib_every > 0:
            self._calib_thread = threading.Thread(
                target=self._recalib_loop, daemon=True,
                name=f"Calib-{self.serial}",
            )
            self._calib_thread.start()

        st_frame = MV_FRAME_OUT()
        while not self._stop_event.is_set():
            memset(byref(st_frame), 0, sizeof(st_frame))
            ret = self._cam.MV_CC_GetImageBuffer(st_frame, self._timeout_ms)
            if ret != 0:
                continue

            try:
                arrival_perf = time.perf_counter()
                info = st_frame.stFrameInfo
                w = int(info.nExtendWidth) if info.nExtendWidth else int(info.nWidth)
                h = int(info.nExtendHeight) if info.nExtendHeight else int(info.nHeight)
                flen = int(info.nFrameLenEx) if info.nFrameLenEx else int(info.nFrameLen)
                data = ctypes.string_at(st_frame.pBufAddr, flen) if flen > 0 else b""
                dev_ts = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)

                # 计算 PC 时间轴上的曝光开始时刻（读取 offset 是原子操作，无需加锁）
                exposure_start_pc = dev_ts / self._tick_frequency_hz + self._time_offset

                frame = Frame(
                    data=data, width=w, height=h,
                    frame_num=int(info.nFrameNum),
                    pixel_type=int(info.enPixelType),
                    dev_timestamp=dev_ts,
                    host_timestamp=int(info.nHostTimeStamp),
                    exposure_time=float(info.fExposureTime),
                    lost_packet=int(info.nLostPacket),
                    arrival_perf=arrival_perf,
                    exposure_start_pc=exposure_start_pc,
                )
                with self._lock:
                    self.frames_received += 1
                    self.lost_packets += frame.lost_packet
                    if self._last_frame_num is not None:
                        gap = frame.frame_num - self._last_frame_num - 1
                        if 0 < gap < 1000:      # 1000+ = 相机重启/计数回绕，不算丢
                            self.frames_missing += gap
                    self._last_frame_num = frame.frame_num
                    if self._frame_sink is None:
                        if len(self._queue) == self._queue.maxlen:
                            self.frames_overwritten += 1
                        self._queue.append(frame)
                if self._frame_sink is not None:
                    try:
                        self._frame_sink(frame)
                    except Exception as exc:  # 下游出错不能把取流线程带死
                        self.sink_errors += 1
                        if self.sink_errors == 1:
                            print(f"[Grabber {self.serial}] frame_sink 出错（只报第一次）: {exc!r}")
            finally:
                try:
                    self._cam.MV_CC_FreeImageBuffer(st_frame)
                except Exception:
                    pass

    def _recalib_loop(self) -> None:
        """后台线程：按帧间隔周期性重新校准 PC-设备时钟偏移。"""
        # 按真实帧率把“每多少帧重校准一次”换算成秒，避免魔法常数带来的抖动。
        if self._fps > 0:
            interval = max(self._recalib_every / self._fps, 0.5)
        else:
            interval = max(self._recalib_every * 0.043, 0.5)
        while not self._stop_event.is_set():
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break
            new_offset = calibrate_time_offset(self._cam, self._tick_frequency_hz)
            # 平滑更新：如果新旧 offset 差异过大（>5ms），可能是测量异常，跳过
            if abs(new_offset - self._time_offset) < 0.005:
                self._time_offset = new_offset

    def get_frame(self, timeout_s: float = 0.0) -> Optional[Frame]:
        """从队列取出一张图并出队。timeout_s=0 非阻塞。"""
        if timeout_s <= 0:
            with self._lock:
                return self._queue.popleft() if self._queue else None

        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            with self._lock:
                if self._queue:
                    return self._queue.popleft()
            time.sleep(0.005)
        return None

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stop(self) -> None:
        self._stop_event.set()

    def _drain_keep_latest(self) -> Optional[Frame]:
        """清空队列，仅返回最新帧（被丢掉的那些计入 frames_overwritten）。"""
        with self._lock:
            if not self._queue:
                return None
            self.frames_overwritten += len(self._queue) - 1
            latest = self._queue[-1]
            self._queue.clear()
            return latest

    def stats(self) -> dict:
        with self._lock:
            out = {
                "received": self.frames_received,
                "missing": self.frames_missing,
                "lost_packets": self.lost_packets,
                "overwritten": self.frames_overwritten,
            }
        if self._frame_sink is not None:
            out["sink_errors"] = self.sink_errors
        return out


# ───────────────── 同步采集 ─────────────────


#: Sleep stops this far short of the deadline and the rest is spun out. Windows
#: wakes a sleeper late by a millisecond or more even with the 1 ms timer, and a
#: late trigger costs a whole frame (see ActionTriggerLoop).
_TRIGGER_SPIN_GUARD_S = 0.0015


class _HighResolutionTimers:
    """Ask Windows for a 1 ms scheduler tick for as long as triggering runs."""

    def __init__(self) -> None:
        self._winmm = getattr(getattr(ctypes, "windll", None), "winmm", None)

    def __enter__(self) -> "_HighResolutionTimers":
        if self._winmm is not None:
            try:
                self._winmm.timeBeginPeriod(1)
            except Exception:
                self._winmm = None
        return self

    def __exit__(self, *exc) -> None:
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass


class ActionTriggerLoop(threading.Thread):
    """Broadcast GigE action commands on a fixed grid t0 + n*period.

    The cadence is a grid, never a running offset. The previous version advanced
    with ``next_t = now + period`` whenever a wake-up was late by more than
    ``period - minimum_period``, which permanently pushed the schedule out: at
    fps=40 / acquisition=48 that tolerance is only 3.9 ms, so ordinary Windows
    sleep jitter turned a 25.0 ms request into a 30.9 ms mean (0905 213942:
    inter-exposure gaps spread 21-31 ms with 2187 double gaps, 32.3 Hz delivered
    out of ~39.6 Hz issued). Fixing slots keeps every gap an exact multiple of
    the period, so no trigger can arrive faster than the sensor accepts and a
    late wake-up costs one slot instead of shifting all later ones.

    Multi-rate (one high-speed camera): with ``sync_every=N`` the grid runs at
    N x the base rate. Ordinary slots send ``fast_group_mask`` (only the
    high-speed camera listens to that bit); every N-th slot sends ``group_mask``
    and fires every camera at once. The slow cameras keep their own floor,
    ``sync_minimum_period_s``: if the previous all-camera fire was so late that
    this one would land inside their readout, the slot is downgraded to
    fast-only. A slow camera then loses one frame that we count, instead of
    silently swallowing a trigger that no counter sees.
    """

    def __init__(
        self,
        *,
        stop_event: threading.Event,
        fps: float,
        acquisition_frame_rate: float,
        device_key: int,
        group_key: int,
        group_mask: int,
        broadcast_address: str,
        timeout_ms: int,
        sync_every: int = 1,
        fast_group_mask: int = 0,
        sync_minimum_period_s: float = 0.0,
        minimum_period_margin_s: float = 0.00025,
    ) -> None:
        super().__init__(daemon=True, name="ActionTriggerLoop")
        self._stop_event = stop_event
        self._period = 1.0 / max(float(fps), 0.001)
        self._minimum_period = (
            1.0 / max(float(acquisition_frame_rate), 0.001)
            + float(minimum_period_margin_s)
        )
        if self._period < self._minimum_period:
            raise ValueError(
                f"trigger period {self._period * 1000:.2f} ms (fps={fps:g}) is shorter "
                f"than the camera's minimum {self._minimum_period * 1000:.2f} ms "
                f"(acquisition_frame_rate={acquisition_frame_rate:g}); the sensor would "
                "silently reject triggers. Raise acquisition_frame_rate to at least "
                f"{1.3 * float(fps):.0f} in the camera config."
            )
        self._sync_every = int(sync_every)
        if self._sync_every < 1:
            raise ValueError(f"sync_every must be >= 1, got {sync_every!r}")
        self._fast_group_mask = int(fast_group_mask)
        self._sync_minimum_period = float(sync_minimum_period_s)
        if self._sync_every > 1:
            if self._fast_group_mask == 0 or self._fast_group_mask & int(group_mask):
                raise ValueError(
                    "sync_every > 1 needs a fast_group_mask disjoint from group_mask "
                    f"(group_mask={group_mask:#x}, fast_group_mask={fast_group_mask:#x})"
                )
            if self._sync_every * self._period < self._sync_minimum_period:
                raise ValueError(
                    f"all-camera period {self._sync_every * self._period * 1000:.2f} ms is "
                    f"shorter than the slow cameras' minimum "
                    f"{self._sync_minimum_period * 1000:.2f} ms"
                )
        self._device_key = int(device_key)
        self._group_key = int(group_key)
        self._group_mask = int(group_mask)
        self._broadcast_address = str(broadcast_address)
        self._timeout_ms = int(timeout_ms)
        # Diagnostics: without these a lost frame is indistinguishable from a
        # trigger that was never issued.
        self.issued = 0
        self.skipped_slots = 0
        self.max_late_s = 0.0
        self.late_over_0_5ms = 0
        self.late_over_1ms = 0
        self.late_over_2ms = 0
        self.issued_sync = 0
        self.skipped_sync_slots = 0
        self.sync_downgraded = 0

    def _slot_kind(self, slot: int, fire_elapsed: float, last_sync_elapsed: float) -> str:
        """'sync' (every camera), 'fast' (high-speed camera only) or 'downgraded'."""
        if self._sync_every == 1:
            return "sync"
        if slot % self._sync_every:
            return "fast"
        if fire_elapsed - last_sync_elapsed < self._sync_minimum_period:
            return "downgraded"
        return "sync"

    def _next_slot(self, fired_slot: int, elapsed: float) -> int:
        """Grid slot for the next trigger, given how far we are past t0 now.

        Always the next slot, unless we are already so late that it would land
        closer than the sensor's minimum period — then skip whole slots.
        """
        slot = fired_slot + 1
        earliest = elapsed + self._minimum_period
        if slot * self._period < earliest:
            slot = int(math.ceil(earliest / self._period))
        return slot

    def _issue(self, group_mask: int) -> None:
        info = MV_ACTION_CMD_INFO()
        results = MV_ACTION_CMD_RESULT_LIST()
        info.nDeviceKey = self._device_key
        info.nGroupKey = self._group_key
        info.nGroupMask = int(group_mask)
        info.bActionTimeEnable = 0
        info.nActionTime = 0
        info.pBroadcastAddress = self._broadcast_address.encode("ascii")
        info.nTimeOut = self._timeout_ms
        info.bSpecialNetEnable = 0
        info.nSpecialNetIP = 0
        MvCamera.MV_GIGE_IssueActionCommand(info, results)

    def run(self) -> None:
        with _HighResolutionTimers():
            t0 = time.perf_counter()
            slot = 1
            last_sync_elapsed = -math.inf
            while not self._stop_event.is_set():
                deadline = t0 + slot * self._period
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    if self._stop_event.is_set():
                        return
                    if remaining > _TRIGGER_SPIN_GUARD_S:
                        time.sleep(min(remaining - _TRIGGER_SPIN_GUARD_S, 0.002))
                fire_elapsed = time.perf_counter() - t0
                kind = self._slot_kind(slot, fire_elapsed, last_sync_elapsed)
                self._issue(self._group_mask if kind == "sync" else self._fast_group_mask)
                self.issued += 1
                if kind == "sync":
                    self.issued_sync += 1
                    last_sync_elapsed = fire_elapsed
                elif kind == "downgraded":
                    self.sync_downgraded += 1
                elapsed = time.perf_counter() - t0
                late = elapsed - slot * self._period
                self.max_late_s = max(self.max_late_s, late)
                self.late_over_0_5ms += late > 0.0005
                self.late_over_1ms += late > 0.001
                self.late_over_2ms += late > 0.002
                next_slot = self._next_slot(slot, elapsed)
                self.skipped_slots += next_slot - slot - 1
                if self._sync_every > 1:
                    self.skipped_sync_slots += sum(
                        1 for s in range(slot + 1, next_slot) if s % self._sync_every == 0
                    )
                slot = next_slot

    def stats(self) -> dict:
        out = {
            "issued": self.issued,
            "skipped_slots": self.skipped_slots,
            "max_late_ms": round(self.max_late_s * 1000.0, 3),
            "late_over_0_5ms": self.late_over_0_5ms,
            "late_over_1ms": self.late_over_1ms,
            "late_over_2ms": self.late_over_2ms,
        }
        if self._sync_every > 1:
            out.update({
                "slot_fps": round(1.0 / self._period, 3),
                "sync_every": self._sync_every,
                "issued_sync": self.issued_sync,
                "skipped_sync_slots": self.skipped_sync_slots,
                "sync_downgraded": self.sync_downgraded,
            })
        return out


_REPO_ROOT = Path(__file__).resolve().parents[1]


class ActionTriggerProcess:
    """ActionTriggerLoop 搬到独立子进程（HIGH 优先级）里跑，接口同 ActionTriggerLoop（start/join/stats）。

    tracker 进程里十几条线程抢 GIL，触发线程醒来常要等别人让出：2026-10-06 ASUS 真 tracker
    （405 120Hz）同进程跑，每秒 ~9 格迟到超过 0.55 ms 被跳掉（8%），GIL 切换间隔压到 0.5 ms
    也只是从 1454 格降到 593 格。子进程里只有这一条线程要 GIL，主线程阻塞在 stdin 上。

    协议：stdin 第一行 = ActionTriggerLoop 参数（JSON）；子进程起好触发后回一行 ``ready``；
    之后父进程写任意一行或关管道 = 停；子进程把 stats（JSON 一行）写到 stdout 后退出。
    父进程死了管道就断，子进程读到 EOF 自己停，不会对着相机一直发。
    """

    def __init__(self, *, params: dict, dry_run: bool = False) -> None:
        # 参数在父进程先校验一遍：配错了在开场就抛，别等子进程起来才发现
        ActionTriggerLoop(stop_event=threading.Event(), **params)
        self._params = dict(params)
        self._dry_run = bool(dry_run)
        self._proc: Optional[Any] = None
        self._final: Optional[dict] = None

    def start(self, *, ready_timeout_s: float = 20.0) -> None:
        import subprocess

        code = (
            "import sys; sys.path.insert(0, {root!r}); "
            "from src.ball_grabber import _action_trigger_child_main; _action_trigger_child_main()"
        ).format(root=str(_REPO_ROOT))
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            # 不吃控制台 Ctrl+C：停只走 stdin 协议，保证父进程收尾时还能拿到统计
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps({**self._params, "dry_run": self._dry_run}) + "\n")
        self._proc.stdin.flush()
        ready: list[str] = []

        def _wait_ready() -> None:
            for line in self._proc.stdout:
                if line.strip() == "ready":
                    ready.append(line)
                    return
                print(f"[trigger-proc] {line.rstrip()}")

        waiter = threading.Thread(target=_wait_ready, daemon=True)
        waiter.start()
        waiter.join(timeout=ready_timeout_s)
        if not ready:
            self._proc.kill()
            raise RuntimeError(
                f"触发子进程 {ready_timeout_s:g}s 内没回 ready（exit={self._proc.poll()}），看上面 stderr"
            )

    def join(self, timeout: float = 5.0) -> None:
        if self._proc is None or self._final is not None:
            return
        import subprocess

        try:
            self._proc.stdin.write("stop\n")
            self._proc.stdin.flush()
        except OSError:
            pass
        try:
            out, _ = self._proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            out, _ = self._proc.communicate()
        stats_lines = [line for line in (out or "").splitlines() if line.startswith("{")]
        self._final = (
            json.loads(stats_lines[-1]) if stats_lines
            else {"error": f"trigger process exited {self._proc.returncode} without stats"}
        )
        self._final["process"] = "subprocess"

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stats(self) -> dict:
        return dict(self._final) if self._final is not None else {"running": self.is_alive()}


def _action_trigger_child_main() -> None:
    """ActionTriggerProcess 的子进程入口（见其文档里的协议）。"""
    params = json.loads(sys.stdin.readline())
    dry_run = bool(params.pop("dry_run", False))
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
        except Exception:
            pass
    if not dry_run:
        MvCamera.MV_CC_Initialize()
    stop = threading.Event()
    loop = ActionTriggerLoop(stop_event=stop, **params)
    if dry_run:
        loop._issue = lambda _mask: None  # 单测：不往网上发 action 包
    loop.start()
    print("ready", flush=True)
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    stop.set()
    loop.join(timeout=2.0)
    print(json.dumps(loop.stats()), flush=True)
    if not dry_run:
        MvCamera.MV_CC_Finalize()


# ───────────────── 高速相机 ─────────────────


#: 高速相机在 action 组掩码里多认的一位。普通相机只认 action_group_mask（默认 1），
#: 触发环平时只发这一位 → 只有高速相机曝光；每 N 格发全组 → 所有相机同拍。
HIGH_SPEED_GROUP_BIT = 0x2
#: 按实测 ResultingFrameRate 排触发时留的余量。action 包在交换机上的抖动是几十 µs 级，
#: 普通流程那 0.25 ms 在 120Hz（周期 8.33 ms、相机下限 7.69 ms）下会白白吃掉四成迟到容忍。
_HIGH_SPEED_PERIOD_MARGIN_S = 0.0001
#: 同步组配高速帧时最多等多久。高速帧小（0.9 MB vs 2.85 MB），通常比普通帧先到。
_HIGH_SPEED_SYNC_WAIT_S = 0.012
#: 高速模式下 SDK 取流缓存节点数（高速相机 / 其余相机），都约合 0.2~0.27 s 的帧。
#: 1006 真 tracker 冒烟：SDK 默认节点下 405 每分钟 ~23 帧、其余 ~12 帧帧号跳号——取流线程
#: 在 tracker 进程里等 GIL，SDK 缓存满了就丢。
_HIGH_SPEED_IMAGE_NODES = 32
_SLOW_IMAGE_NODES_WITH_HIGH_SPEED = 8


@dataclass(frozen=True)
class HighSpeedConfig:
    """相机配置里的 high_speed_camera 段：哪台、几倍频、ROI（全幅像素坐标）。"""
    serial: str
    rate_multiplier: int
    roi_offset_x: int
    roi_offset_y: int
    roi_width: int
    roi_height: int
    acquisition_frame_rate: float

    @classmethod
    def from_dict(cls, raw: dict) -> "HighSpeedConfig":
        keys = ("serial", "rate_multiplier", "roi_offset_x", "roi_offset_y",
                "roi_width", "roi_height", "acquisition_frame_rate")
        missing = [k for k in keys if k not in raw]
        unknown = sorted(set(raw) - set(keys))
        if missing or unknown:
            raise ValueError(
                f"high_speed_camera 配置不对：缺 {missing}，多 {unknown}（必填 {list(keys)}，没有默认值）"
            )
        cfg = cls(
            serial=str(raw["serial"]).strip(),
            rate_multiplier=int(raw["rate_multiplier"]),
            roi_offset_x=int(raw["roi_offset_x"]),
            roi_offset_y=int(raw["roi_offset_y"]),
            roi_width=int(raw["roi_width"]),
            roi_height=int(raw["roi_height"]),
            acquisition_frame_rate=float(raw["acquisition_frame_rate"]),
        )
        if cfg.rate_multiplier < 2:
            raise ValueError(f"high_speed_camera.rate_multiplier 至少 2，给的是 {cfg.rate_multiplier}")
        if cfg.roi_offset_x % 2 or cfg.roi_offset_y % 2:
            # 奇数偏移会把 Bayer 相位挪一格，贴回画布后整幅颜色错位
            raise ValueError("high_speed_camera 的 roi_offset_x/roi_offset_y 必须是偶数（保持 Bayer 相位）")
        if cfg.roi_width <= 0 or cfg.roi_height <= 0:
            raise ValueError("high_speed_camera 的 roi_width/roi_height 必须 > 0")
        return cfg


_PAD_PIXEL_TYPES = (
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerGB8,
)


def pad_roi_frame(
    frame: Frame,
    *,
    canvas_width: int,
    canvas_height: int,
    offset_x: int,
    offset_y: int,
) -> Frame:
    """把 ROI 帧贴回全幅画布（ROI 外全 0），像素坐标与全幅标定逐点一致。

    同步组里其余相机都是全宽、OffsetY=0 的画面；高速相机的小 ROI 贴回同样大小的画布后，
    下游（切片、视频网格、黑标投影）都不用知道它裁过——0905 黑标整场归零就是因为
    ROI 尺寸和标定 image_size 对不上又没人重映射（v04-grid-video-roi-height-vs-calib-0905）。
    """
    import numpy as np

    if frame.pixel_type not in _PAD_PIXEL_TYPES:
        raise ValueError(f"pad_roi_frame 只支持 8-bit 单通道（Bayer8/Mono8），给的是 0x{frame.pixel_type:08X}")
    if offset_x % 2 or offset_y % 2:
        raise ValueError("ROI 偏移必须是偶数（保持 Bayer 相位）")
    if offset_x < 0 or offset_y < 0 or offset_x + frame.width > canvas_width or offset_y + frame.height > canvas_height:
        raise ValueError(
            f"ROI {frame.width}x{frame.height}@({offset_x},{offset_y}) 超出画布 {canvas_width}x{canvas_height}"
        )
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    roi = np.frombuffer(frame.data, dtype=np.uint8, count=frame.width * frame.height)
    canvas[offset_y:offset_y + frame.height, offset_x:offset_x + frame.width] = roi.reshape(frame.height, frame.width)
    return replace(frame, data=canvas.tobytes(), width=canvas_width, height=canvas_height)


class HighSpeedStream:
    """高速相机的全部帧：短环形缓冲供同步帧配对，同时转给录像器。

    帧由 ImageGrabber 线程经 frame_sink 直接推进来，不走 get_frames 的"只留最新"排空——
    那条路会把 2/3 的高速帧当积压丢掉，还会把相邻一格（120Hz 下 8.3 ms）的非同步帧
    错配进同步组（get_frames 的曝光门是 10 ms）。
    """

    def __init__(self, serial: str, *, slot_period_s: float, ring_size: int = 16) -> None:
        self.serial = serial
        # 半格：相邻高速帧差一整格，配对唯一
        self._tolerance_s = float(slot_period_s) / 2.0
        self._ring: deque[Frame] = deque(maxlen=int(ring_size))
        self._cond = threading.Condition()
        self._recorder: Optional["HighSpeedRecorder"] = None
        self.received = 0
        self.sync_matched = 0
        self.sync_missed = 0
        self.sync_dt_abs_max_s = 0.0
        self._sync_dt_abs_sum_s = 0.0
        self.sync_frame_nums: list[int] = []

    def push(self, frame: Frame) -> None:
        with self._cond:
            self._ring.append(frame)
            self.received += 1
            recorder = self._recorder
            self._cond.notify_all()
        if recorder is not None:
            recorder.submit(frame)

    def attach_recorder(self, recorder: "HighSpeedRecorder") -> None:
        with self._cond:
            self._recorder = recorder

    def detach_recorder(self) -> Optional["HighSpeedRecorder"]:
        with self._cond:
            recorder, self._recorder = self._recorder, None
        return recorder

    def match(self, exposure_pc: float, *, wait_s: float) -> Optional[Frame]:
        """曝光时刻离 exposure_pc 最近（半格以内）的那一帧；没到就最多等 wait_s。"""
        deadline = time.perf_counter() + float(wait_s)
        with self._cond:
            while True:
                best: Optional[Frame] = None
                best_dt = math.inf
                newest = -math.inf
                for f in self._ring:
                    newest = max(newest, f.exposure_start_pc)
                    dt = abs(f.exposure_start_pc - exposure_pc)
                    if dt < best_dt:
                        best, best_dt = f, dt
                if best is not None and best_dt <= self._tolerance_s:
                    self.sync_matched += 1
                    self._sync_dt_abs_sum_s += best_dt
                    self.sync_dt_abs_max_s = max(self.sync_dt_abs_max_s, best_dt)
                    self.sync_frame_nums.append(best.frame_num)
                    return best
                remaining = deadline - time.perf_counter()
                # 帧按序到：已经有比它晚半格以上的帧，说明同拍那一帧丢了，不必再等
                if remaining <= 0 or newest > exposure_pc + self._tolerance_s:
                    self.sync_missed += 1
                    return None
                self._cond.wait(remaining)

    def stats(self) -> dict:
        with self._cond:
            n = self.sync_matched
            return {
                "received": self.received,
                "sync_matched": n,
                "sync_missed": self.sync_missed,
                "sync_dt_ms_mean": round(self._sync_dt_abs_sum_s / n * 1000.0, 3) if n else None,
                "sync_dt_ms_max": round(self.sync_dt_abs_max_s * 1000.0, 3) if n else None,
            }


#: 逐帧索引记录（<base>.idx）：numpy dtype
#: [("frame_num","<i8"),("dev_timestamp","<u8"),("exposure_start_pc","<f8"),("arrival_perf","<f8"),("lost_packet","<i8")]
HIGH_SPEED_INDEX_FORMAT = "<qQddq"


class HighSpeedRecorder(threading.Thread):
    """把高速相机每一帧原样（8-bit Bayer）顺序写盘。

    - ``<base>.bayer``：各帧原始字节首尾相接，每帧 width*height 字节，第 i 帧在 i*width*height；
    - ``<base>.idx``：每帧一条 40 字节定长记录（HIGH_SPEED_INDEX_FORMAT），随写随落盘，
      进程崩了也能按记录数恢复时间戳；
    - ``<base>.json``：ROI / 画布 / 帧率等元数据，开录写一次，收尾带统计再写一次。

    不在采集机上编码：原始 108 MB/s 顺序写对 NVMe 是零负担，而且黑标要的是无损像素。
    """

    def __init__(self, base_path: Path, *, meta: dict, queue_max: int = 600) -> None:
        super().__init__(daemon=True, name="HighSpeedRecorder")
        base_path = Path(base_path)
        self.raw_path = base_path.with_name(base_path.name + ".bayer")
        self.index_path = base_path.with_name(base_path.name + ".idx")
        self.json_path = base_path.with_name(base_path.name + ".json")
        self._meta = dict(meta)
        self._queue: queue.Queue[Optional[Frame]] = queue.Queue(maxsize=int(queue_max))
        self.written = 0
        self.dropped = 0
        self.size_mismatch = 0
        self.bytes_written = 0
        self.queue_max_size = 0
        self.frame_bytes: Optional[int] = None
        self.frame_shape: Optional[tuple[int, int]] = None
        self.pixel_type: Optional[int] = None
        self.error: Optional[str] = None
        self._write_json(status="recording")

    def _write_json(self, *, status: str, extra: Optional[dict] = None) -> None:
        doc = {
            **self._meta,
            "status": status,
            "files": {
                "raw": self.raw_path.name,
                "index": self.index_path.name,
                "index_format": HIGH_SPEED_INDEX_FORMAT,
                "index_fields": ["frame_num", "dev_timestamp", "exposure_start_pc",
                                 "arrival_perf", "lost_packet"],
            },
            **(extra or {}),
        }
        with open(self.json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    def submit(self, frame: Frame) -> None:
        """取流线程调用：非阻塞，满了就丢并计数（写盘跟不上不能拖住取流）。"""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1
            return
        size = self._queue.qsize()
        if size > self.queue_max_size:
            self.queue_max_size = size

    def run(self) -> None:
        import struct

        pack = struct.Struct(HIGH_SPEED_INDEX_FORMAT).pack
        try:
            with open(self.raw_path, "wb", buffering=8 << 20) as raw, \
                    open(self.index_path, "wb", buffering=1 << 16) as idx:
                while True:
                    frame = self._queue.get()
                    if frame is None:
                        break
                    if self.frame_bytes is None:
                        self.frame_bytes = len(frame.data)
                        self.frame_shape = (int(frame.height), int(frame.width))
                        self.pixel_type = int(frame.pixel_type)
                    elif len(frame.data) != self.frame_bytes:
                        self.size_mismatch += 1
                        continue
                    raw.write(frame.data)
                    idx.write(pack(
                        int(frame.frame_num),
                        int(frame.dev_timestamp),
                        float(frame.exposure_start_pc),
                        float(frame.arrival_perf),
                        int(frame.lost_packet),
                    ))
                    self.written += 1
                    self.bytes_written += len(frame.data)
        except Exception as exc:
            self.error = repr(exc)
            print(f"[HighSpeedRecorder] 写盘出错，后续帧全部丢弃: {self.error}")
            # 写线程挂了也要一直排空到哨兵，否则 close() 的哨兵塞不进满队列
            while self._queue.get() is not None:
                self.dropped += 1

    def stats(self) -> dict:
        return {
            "written": self.written,
            "dropped": self.dropped,
            "size_mismatch": self.size_mismatch,
            "bytes_written": self.bytes_written,
            "frame_bytes": self.frame_bytes,
            # (height, width) 与 GigE PFNC 像素格式码（18F 实测 0x01080008 = BayerGR8）
            "frame_shape": list(self.frame_shape) if self.frame_shape else None,
            "pixel_type": self.pixel_type,
            "queue_max_size": self.queue_max_size,
            "error": self.error,
        }

    def close(self, *, extra: Optional[dict] = None, timeout_s: float = 60.0) -> dict:
        try:
            self._queue.put(None, timeout=timeout_s)
        except queue.Full:
            self.error = self.error or "close: queue stuck full"
        self.join(timeout=timeout_s)
        stats = self.stats()
        self._write_json(
            status="complete" if not self.is_alive() and self.error is None else "incomplete",
            extra={**(extra or {}), "recording": stats},
        )
        return stats


# ───────────────── 启动自动增益 ─────────────────

# 目标：每台相机画面原始像素的灰度中位数（隔 4 行 4 列取样，BayerGR8 下取到的全是绿通道）。
# 定标（2026-10-10 上午自然光）：400μs + 模拟 12.78dB + DigitalShift 16.5dB 时四台中位数
# 67/50/58/58，那两场车定位 98~99% 帧、3D 观测 36~37% 帧；同光照 1500/2200μs 亮了 4~5 倍
# （饱和 20~40%）的两场车定位全 0、3D 观测 7~12% —— 过曝先杀 AprilTag。
AUTO_GAIN_TARGET_MEDIAN = 60.0
_AUTO_GAIN_TOLERANCE_DB = 1.0   # |20·log10(目标/实测)| 在这以内算到位
_AUTO_GAIN_MAX_ROUNDS = 8    # 起点整幅过曝时中位数卡在 255，每轮只能退 ~11dB


class SyncCapture:
    """
    多相机硬触发同步采集上下文管理器。

    支持 Line 硬触发和 GigE Action Command 广播触发，由相机配置中的
    trigger_mode 显式选择。

    推荐用法（从配置文件创建）::

        with SyncCapture.from_config() as cap:
            frames = cap.get_frames(timeout_s=1.0)  # {serial: Frame}
            for sn, f in frames.items():
                img = frame_to_numpy(f)  # numpy BGR
    """

    def __init__(
        self,
        master_serial: str,
        slave_serials: list[str],
        *,
        trigger_mode: str,
        fps: float = 30.0,
        sync_threshold_ms: float = 200.0,
        master_line_output: str = "Line1",
        master_line_source: str = "ExposureStartActive",
        slave_trigger_source: str = "Line0",
        slave_trigger_activation: str = "FallingEdge",
        master_min_bandwidth: bool = False,
        acquisition_frame_rate: float = 30.0,
        action_trigger_source: str = "Action1",
        action_device_key: int = 1,
        action_group_key: int = 1,
        action_group_mask: int = 1,
        action_broadcast_address: str = "255.255.255.255",
        action_timeout_ms: int = 0,
        gev_scpd: int = 0,
        recalib_every: int = 10,
        exposure_us: float = 0.0,
        gain_db: float = -1.0,
        digital_shift: Optional[float] = None,
        pixel_format: str = "",
        # 全相机统一底部裁剪（OffsetY 固定 0，只砍底部行）。因为偏移是 0，
        # 裁剪后的像素坐标与全幅逐点相同，标定内参和三角测量无需任何重映射。
        roi_height: int = 0,
        slave_params: Optional[dict[str, dict]] = None,
        # 可选：一台相机裁小 ROI 按 rate_multiplier 倍帧率跑（见模块文档「高速相机」）
        high_speed_camera: Optional[dict] = None,
    ):
        self._master = master_serial
        self._slaves = list(slave_serials)
        self._all_serials = [master_serial] + self._slaves
        self._trigger_mode = str(trigger_mode).strip().lower()
        if self._trigger_mode not in ("line", "action"):
            raise ValueError(f"unsupported trigger_mode: {trigger_mode!r}")
        if self._trigger_mode == "line" and master_min_bandwidth:
            self._sync_serials = list(self._slaves)
        else:
            self._sync_serials = list(self._all_serials)
        self._sync_set = set(self._sync_serials)
        self._fps = fps
        self._recalib_every = recalib_every
        self._sync_threshold_s = sync_threshold_ms / 1000.0

        self._high_speed = (
            HighSpeedConfig.from_dict(high_speed_camera) if high_speed_camera else None
        )
        hs = self._high_speed
        if hs is not None:
            if self._trigger_mode != "action":
                raise ValueError("high_speed_camera 只支持 trigger_mode=action（靠 action 组掩码分频）")
            if hs.serial not in self._sync_set:
                raise ValueError(f"high_speed_camera.serial={hs.serial} 不在同步相机 {self._sync_serials} 里")
            if int(action_group_mask) & HIGH_SPEED_GROUP_BIT:
                raise ValueError(
                    f"action_group_mask={action_group_mask:#x} 占了高速相机专用位 {HIGH_SPEED_GROUP_BIT:#x}"
                )
        # get_frames 从各 grabber 拼的相机；高速相机不在里面，拼好后按曝光时刻另配
        self._assembly_serials = [
            sn for sn in self._sync_serials if hs is None or sn != hs.serial
        ]
        self._hs_stream: Optional[HighSpeedStream] = None
        self._hs_recorder: Optional[HighSpeedRecorder] = None
        self._hs_info: Optional[dict] = None

        self._cameras: dict[str, Any] = {}
        self._grabbers: dict[str, ImageGrabber] = {}
        self._action_trigger: Optional[ActionTriggerLoop | ActionTriggerProcess] = None
        self._stop = threading.Event()
        self._exposure_spread_discards = 0
        self._arrival_spread_discards = 0
        self._assembly_timeouts = 0
        self._final_stats: Optional[dict] = None

        MvCamera.MV_CC_Initialize()

        st_dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, st_dev_list)
        _check(ret, "EnumDevices")

        try:
            if self._trigger_mode == "line":
                self._cameras[master_serial] = open_camera(
                    master_serial,
                    trigger_source=None,
                    frame_rate=fps,
                    full_frame=True,
                    line_output=master_line_output,
                    line_source=master_line_source,
                    exposure_us=exposure_us,
                    gain_db=gain_db,
                    digital_shift=digital_shift,
                    pixel_format=pixel_format,
                    roi_height=16 if master_min_bandwidth else 0,
                    _st_dev_list=st_dev_list,
                )
                for sn in self._slaves:
                    params = (slave_params or {}).get(sn, {})
                    self._cameras[sn] = open_camera(
                        sn,
                        trigger_source=slave_trigger_source,
                        trigger_activation=slave_trigger_activation,
                        full_frame=True,
                        exposure_us=exposure_us,
                        gain_db=gain_db,
                        digital_shift=digital_shift,
                        pixel_format=pixel_format,
                        roi_offset_y=params.get("roi_offset_y", 0),
                        roi_height=params.get("roi_height", roi_height),
                        roi_width=params.get("roi_width", 0),
                        binning=params.get("binning", 1),
                        _st_dev_list=st_dev_list,
                    )
            else:
                for sn in self._all_serials:
                    params = (slave_params or {}).get(sn, {})
                    if hs is not None and sn == hs.serial:
                        roi_kwargs = dict(
                            roi_offset_x=hs.roi_offset_x,
                            roi_offset_y=hs.roi_offset_y,
                            roi_width=hs.roi_width,
                            roi_height=hs.roi_height,
                        )
                        cam_group_mask = int(action_group_mask) | HIGH_SPEED_GROUP_BIT
                        cam_frame_rate = hs.acquisition_frame_rate
                        cam_trigger_fps = self._fps * hs.rate_multiplier
                        cam_image_nodes = _HIGH_SPEED_IMAGE_NODES
                    else:
                        roi_kwargs = dict(
                            roi_offset_y=params.get("roi_offset_y", 0),
                            roi_height=params.get("roi_height", roi_height),
                            roi_width=params.get("roi_width", 0),
                        )
                        cam_group_mask = action_group_mask
                        cam_frame_rate = acquisition_frame_rate
                        cam_trigger_fps = self._fps
                        cam_image_nodes = _SLOW_IMAGE_NODES_WITH_HIGH_SPEED if hs is not None else 0
                    self._cameras[sn] = open_camera(
                        sn,
                        trigger_source=action_trigger_source,
                        action_device_key=action_device_key,
                        action_group_key=action_group_key,
                        action_group_mask=cam_group_mask,
                        frame_rate=cam_frame_rate,
                        trigger_fps=cam_trigger_fps,
                        full_frame=True,
                        exposure_us=exposure_us,
                        gain_db=gain_db,
                        digital_shift=digital_shift,
                        pixel_format=pixel_format,
                        binning=params.get("binning", 1),
                        gev_scpd=gev_scpd,
                        image_node_num=cam_image_nodes,
                        _st_dev_list=st_dev_list,
                        **roi_kwargs,
                    )
                if hs is not None:
                    self._setup_high_speed(hs)

            # 启动 grabber
            for sn, cam in self._cameras.items():
                is_hs = hs is not None and sn == hs.serial
                g = ImageGrabber(cam, sn, timeout_ms=1000,
                                 recalib_every=self._recalib_every,
                                 fps=self._fps * hs.rate_multiplier if is_hs else self._fps,
                                 frame_sink=self._hs_stream.push if is_hs else None)
                self._grabbers[sn] = g
                g.start()

            if self._trigger_mode == "action":
                time.sleep(0.2)
                if hs is None:
                    self._action_trigger = ActionTriggerLoop(
                        stop_event=self._stop,
                        fps=self._fps,
                        acquisition_frame_rate=acquisition_frame_rate,
                        device_key=action_device_key,
                        group_key=action_group_key,
                        group_mask=action_group_mask,
                        broadcast_address=action_broadcast_address,
                        timeout_ms=action_timeout_ms,
                    )
                else:
                    # 高速格按 405 的传感器上限排（链路上限是平均值，相机会缓存，见
                    # read_sensor_frame_rate）；全组格仍按其余相机读回的 ResultingFrameRate 把关。
                    # 触发环放子进程，不跟本进程的线程抢 GIL（见 ActionTriggerProcess）。
                    self._action_trigger = ActionTriggerProcess(params=dict(
                        fps=self._fps * hs.rate_multiplier,
                        acquisition_frame_rate=self._hs_info["sensor_fps"][hs.serial],
                        device_key=action_device_key,
                        group_key=action_group_key,
                        group_mask=action_group_mask,
                        fast_group_mask=HIGH_SPEED_GROUP_BIT,
                        sync_every=hs.rate_multiplier,
                        sync_minimum_period_s=(
                            1.0 / min(self._hs_info["resulting_fps"][sn] for sn in self._assembly_serials)
                            + _HIGH_SPEED_PERIOD_MARGIN_S
                        ),
                        minimum_period_margin_s=_HIGH_SPEED_PERIOD_MARGIN_S,
                        broadcast_address=action_broadcast_address,
                        timeout_ms=action_timeout_ms,
                    ))
                self._action_trigger.start()
        except Exception:
            self.close()
            raise

    def _setup_high_speed(self, hs: HighSpeedConfig) -> None:
        """核对高速相机 ROI 真的按配置生效、画布尺寸，建 HighSpeedStream。"""
        rois = {sn: read_roi(cam, sn) for sn, cam in self._cameras.items()}
        want = {
            "OffsetX": hs.roi_offset_x,
            "OffsetY": hs.roi_offset_y,
            "Width": hs.roi_width,
            "Height": hs.roi_height,
        }
        if rois[hs.serial] != want:
            # open_camera 写 ROI 失败是静默的（步进不对会被相机拒），这里必须拦：
            # 偏移错了，贴回画布的位置就错了，黑标投影整体偏移而不报错
            raise RuntimeError(
                f"高速相机 {hs.serial} ROI 没按配置生效：要 {want}，相机实际 {rois[hs.serial]}"
                "（Width/OffsetX 步进 4、Height/OffsetY 步进 2，先在 MVS 里确认）"
            )
        canvases = {
            (rois[sn]["OffsetX"], rois[sn]["OffsetY"], rois[sn]["Width"], rois[sn]["Height"])
            for sn in self._assembly_serials
        }
        if len(canvases) != 1:
            raise RuntimeError(f"其余相机画幅不一致，没法定高速相机的画布：{rois}")
        off_x, off_y, canvas_w, canvas_h = canvases.pop()
        if off_x or off_y:
            raise RuntimeError(f"其余相机 ROI 偏移必须为 0（标定按全幅像素），实际 ({off_x},{off_y})")
        if hs.roi_offset_x + hs.roi_width > canvas_w or hs.roi_offset_y + hs.roi_height > canvas_h:
            raise RuntimeError(
                f"高速相机 ROI {hs.roi_width}x{hs.roi_height}@({hs.roi_offset_x},{hs.roi_offset_y}) "
                f"超出其余相机画幅 {canvas_w}x{canvas_h}"
            )
        resulting = {
            sn: round(read_resulting_frame_rate(cam, sn), 3)
            for sn, cam in self._cameras.items()
        }
        # 高速相机每格只有 1/120 s，触发下限得按传感器排（见 read_sensor_frame_rate）；
        # 此时 grabber 还没起，停一下流量完再开
        hs_cam = self._cameras[hs.serial]
        _check(hs_cam.MV_CC_StopGrabbing(), f"StopGrabbing({hs.serial})")
        try:
            sensor_fps = round(read_sensor_frame_rate(hs_cam, hs.serial), 3)
        finally:
            _check(hs_cam.MV_CC_StartGrabbing(), f"StartGrabbing({hs.serial})")
        slot_fps = self._fps * hs.rate_multiplier
        self._hs_info = {
            "serial": hs.serial,
            "rate_multiplier": hs.rate_multiplier,
            "base_fps": self._fps,
            "slot_fps": slot_fps,
            "roi": {
                "offset_x": hs.roi_offset_x,
                "offset_y": hs.roi_offset_y,
                "width": hs.roi_width,
                "height": hs.roi_height,
            },
            "canvas": {"width": canvas_w, "height": canvas_h},
            "acquisition_frame_rate": hs.acquisition_frame_rate,
            # 链路平均上限（ResultingFrameRate）与高速相机传感器上限（触发按它排）
            "resulting_fps": resulting,
            "sensor_fps": {hs.serial: sensor_fps},
            "action_group_bits": {
                "high_speed_only": HIGH_SPEED_GROUP_BIT,
            },
        }
        self._hs_stream = HighSpeedStream(hs.serial, slot_period_s=1.0 / slot_fps)
        print(
            f"[high-speed] {hs.serial} ROI {hs.roi_width}x{hs.roi_height}"
            f"@({hs.roi_offset_x},{hs.roi_offset_y}) → {slot_fps:g} fps"
            f"（链路上限 {resulting[hs.serial]:.2f}，传感器上限 {sensor_fps:.2f}），"
            f"每 {hs.rate_multiplier} 帧与其余相机同拍"
            f"（其余链路上限 {min(resulting[sn] for sn in self._assembly_serials):.2f}）"
        )

    def _high_speed_sync_frame(self, assembled: dict[str, Frame]) -> Frame:
        """给拼好的同步组配上高速相机同拍的那一帧，贴回全幅画布。"""
        assert self._hs_stream is not None and self._hs_info is not None
        exposure_pc = sum(f.exposure_start_pc for f in assembled.values()) / len(assembled)
        canvas_w = self._hs_info["canvas"]["width"]
        canvas_h = self._hs_info["canvas"]["height"]
        frame = self._hs_stream.match(exposure_pc, wait_s=_HIGH_SPEED_SYNC_WAIT_S)
        if frame is None:
            # 同拍那一帧丢了：给全黑画布占位（下游当这一拍没看到），绝不拿相邻一格的帧顶替
            ref = next(iter(assembled.values()))
            return Frame(
                data=bytes(canvas_w * canvas_h),
                width=canvas_w,
                height=canvas_h,
                frame_num=-1,
                pixel_type=ref.pixel_type,
                arrival_perf=time.perf_counter(),
                exposure_start_pc=exposure_pc,
            )
        roi = self._hs_info["roi"]
        return pad_roi_frame(
            frame,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            offset_x=roi["offset_x"],
            offset_y=roi["offset_y"],
        )

    @property
    def high_speed_serial(self) -> Optional[str]:
        """配了高速相机就是它的序列号，否则 None。"""
        return self._high_speed.serial if self._high_speed is not None else None

    def high_speed_info(self) -> Optional[dict]:
        """高速相机的静态配置 + 实测帧率上限 + 录像路径（写进 session json 的 config）。"""
        if self._hs_info is None:
            return None
        info = dict(self._hs_info)
        if self._hs_recorder is not None:
            info["recording"] = {
                "raw": str(self._hs_recorder.raw_path),
                "index": str(self._hs_recorder.index_path),
                "json": str(self._hs_recorder.json_path),
            }
        return info

    def start_high_speed_recording(self, base_path: Path) -> HighSpeedRecorder:
        """开始把高速相机每一帧写盘：<base>.bayer / .idx / .json。"""
        if self._hs_stream is None:
            raise RuntimeError("相机配置里没有 high_speed_camera")
        if self._hs_recorder is not None:
            raise RuntimeError("高速相机已经在录")
        recorder = HighSpeedRecorder(Path(base_path), meta=self.high_speed_info() or {})
        recorder.start()
        self._hs_recorder = recorder
        self._hs_stream.attach_recorder(recorder)
        return recorder

    def _finish_high_speed_recording(self) -> None:
        if self._hs_recorder is None or self._hs_stream is None:
            return
        if self._hs_stream.detach_recorder() is None:
            return  # 已经收过尾（close 被调了两次）
        trigger = self._action_trigger.stats() if self._action_trigger is not None else None
        self._hs_recorder.close(extra={
            "stream": self._hs_stream.stats(),
            "trigger": trigger,
            # 进了同步组的那些高速帧（与其余相机同拍），按 frame_num 记
            "sync_frame_nums": list(self._hs_stream.sync_frame_nums),
        })

    def get_frames(self, timeout_s: float = 1.0) -> Optional[dict[str, Frame]]:
        """
        获取最新一组同步帧。

        仅返回 sync_serials 中的相机帧（Line 模式可不含仅用作触发源的主相机）。
        用 exposure_start_pc 判断是否属于同一触发周期（阈值 10ms）。
        配了高速相机时，先拼其余相机，再按曝光时刻配上它同拍的那一帧（贴回全幅画布）。

        返回 {serial: Frame} 字典，或超时返回 None。
        """
        deadline = time.perf_counter() + timeout_s
        latest: dict[str, Frame] = {}

        while time.perf_counter() < deadline:
            for sn, g in self._grabbers.items():
                f = g._drain_keep_latest()
                # 仅收集 sync_serials 的帧；其他 grabber 照常排空
                # （高速相机的帧走 frame_sink，它的队列恒空）
                if f is not None and sn in self._sync_set:
                    latest[sn] = f

            if len(latest) == len(self._assembly_serials):
                # 粗同步检查：arrival_perf spread
                arrivals = [f.arrival_perf for f in latest.values()]
                spread = max(arrivals) - min(arrivals)
                if spread <= self._sync_threshold_s:
                    # 精同步检查：exposure_start_pc 是否属于同一触发周期
                    exp_starts = [f.exposure_start_pc for f in latest.values()]
                    exp_spread = max(exp_starts) - min(exp_starts)
                    if exp_spread > 0.010:  # >10ms 说明不是同一触发周期
                        self._exposure_spread_discards += 1
                        oldest_sn = min(latest, key=lambda sn: latest[sn].exposure_start_pc)
                        del latest[oldest_sn]
                        time.sleep(0.001)
                        continue
                    if self._hs_stream is not None:
                        latest[self._high_speed.serial] = self._high_speed_sync_frame(latest)
                    return latest
                # 帧不同步，丢弃最老的重新收集
                self._arrival_spread_discards += 1
                oldest_sn = min(latest, key=lambda sn: latest[sn].arrival_perf)
                del latest[oldest_sn]

            time.sleep(0.001)

        self._assembly_timeouts += 1
        return None

    def stats(self) -> dict:
        """整条取帧链的丢帧归因，供 run_tracker 写进 summary。

        `missing` 是相机自己的 nFrameNum 缺口（触发被拒 / 传输丢帧），
        `overwritten` 是主机来不及消费被覆盖的，两者相加才是「触发发了但没进主循环」
        的总数——0905 靠 exposure 间隔反推出的那 18% 以后可以直接读出来。
        """
        if self._final_stats is not None:
            return self._final_stats
        per_cam = {sn: g.stats() for sn, g in self._grabbers.items()}
        out = {
            "per_camera": per_cam,
            "received": sum(v["received"] for v in per_cam.values()),
            "missing": sum(v["missing"] for v in per_cam.values()),
            "lost_packets": sum(v["lost_packets"] for v in per_cam.values()),
            "overwritten": sum(v["overwritten"] for v in per_cam.values()),
            "exposure_spread_discards": self._exposure_spread_discards,
            "arrival_spread_discards": self._arrival_spread_discards,
            "assembly_timeouts": self._assembly_timeouts,
        }
        if self._action_trigger is not None:
            out["trigger"] = self._action_trigger.stats()
        if self._hs_stream is not None:
            # received 按 ImageGrabber 计（per_camera 里也有）；sync_* 是配进同步组的情况
            out["high_speed"] = {
                "serial": self._high_speed.serial,
                "stream": self._hs_stream.stats(),
                "recorder": self._hs_recorder.stats() if self._hs_recorder is not None else None,
            }
        return out

    @classmethod
    def from_config(cls, config_path: str = "", **overrides):
        """从 JSON 配置文件创建 SyncCapture。默认路径 config/camera_18.json。"""
        import json
        if not config_path:
            config_path = str(Path(__file__).resolve().parent / "config" / "camera_18.json")
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(overrides)
        master = cfg.pop("master_serial")
        slaves = cfg.pop("slave_serials")
        return cls(master, slaves, **cfg)

    @property
    def serials(self) -> list[str]:
        """所有打开的相机序列号列表（主机在前）。"""
        return list(self._all_serials)

    @property
    def sync_serials(self) -> list[str]:
        """参与同步输出的相机序列号列表（get_frames 返回的相机）。"""
        return list(self._sync_serials)

    def camera_settings(self) -> dict[str, dict[str, float | bool]]:
        """逐相机读回相机**实际生效**的曝光/模拟增益/数字增益。

        不能用帧头的 fExposureTime —— MV-CS050 这一档根本不填该字段（实测恒为 0）。
        写进 session json 是为了事后能直接判定"这场是不是曝光太长导致回球段拖影漏检"，
        而不是靠 git 翻配置反推（配置可能被 --exposure-us 覆盖，或越界写入被相机拒绝）。
        """
        out: dict[str, dict[str, float | bool]] = {}
        for sn, cam in self._cameras.items():
            entry: dict[str, float | bool] = {}
            try:
                enabled = ctypes.c_bool()
                if int(cam.MV_CC_GetBoolValue("DigitalShiftEnable", enabled)) == 0:
                    entry["digital_shift_enabled"] = bool(enabled.value)
            except Exception:
                pass
            for node, key in (
                ("ExposureTime", "exposure_us"),
                ("Gain", "gain_db"),
                ("DigitalShift", "digital_shift"),
            ):
                try:
                    val = MVCC_FLOATVALUE()
                    if int(cam.MV_CC_GetFloatValue(node, val)) == 0:
                        entry[key] = round(float(val.fCurValue), 2)
                except Exception:
                    pass
            if entry:
                out[sn] = entry
        return out

    def auto_gain(self) -> None:
        """启动时测一次光：每台同步相机把画面灰度中位数拉到 AUTO_GAIN_TARGET_MEDIAN，之后整场不动。

        曝光不动（它决定拖影），只动增益：模拟 Gain 先用满，不够再用 DigitalShift 补
        （CS032 上它就是 ±24dB 的数字增益）；画面偏亮时反过来先退 DigitalShift。
        相机自带 GainAuto 只调模拟增益（0~12.78dB），而模拟增益常年顶格，所以自己测。
        增益到了上下限还不到位只告警：那是该改曝光了。
        """
        import numpy as np

        period_s = 1.0 / self._fps
        hs_serial = self.high_speed_serial
        limits: dict[str, tuple[float, float, float]] = {}  # (模拟上限, 数字下限, 数字上限)
        total_db: dict[str, float] = {}
        for sn in self._sync_serials:
            cam = self._cameras[sn]
            _check(cam.MV_CC_SetEnumValueByString("GainAuto", "Off"), f"GainAuto=Off({sn})")
            _check(cam.MV_CC_SetBoolValue("DigitalShiftEnable", True), f"DigitalShiftEnable({sn})")
            gain, shift = MVCC_FLOATVALUE(), MVCC_FLOATVALUE()
            _check(cam.MV_CC_GetFloatValue("Gain", gain), f"Gain({sn})")
            _check(cam.MV_CC_GetFloatValue("DigitalShift", shift), f"DigitalShift({sn})")
            # 上下限往里收到 0.01dB：两者都是整数寄存器换算出的浮点，写正好等于 fMax 的值
            # 会换算进位越界被拒（实测 Gain=12.7804 → 0x80000102，12.78 照收）
            limits[sn] = (math.floor(gain.fMax * 100) / 100,
                          math.ceil(shift.fMin * 100) / 100,
                          math.floor(shift.fMax * 100) / 100)
            total_db[sn] = float(gain.fCurValue) + float(shift.fCurValue)

        first: dict[str, float] = {}
        for rounds in range(1, _AUTO_GAIN_MAX_ROUNDS + 1):
            # 只认改完增益之后才开始曝光的帧：新值从下一帧起生效，留两帧余量。
            # frame_num < 0 是高速相机同拍帧丢了的黑画布占位，不能拿来测光。
            settled_after = time.perf_counter() + 2.0 * period_s
            while True:
                frames = self.get_frames(timeout_s=1.0)
                if frames is None or time.perf_counter() > settled_after + 2.0:
                    raise RuntimeError("自动增益：取不到改增益之后的同步帧")
                if all(f.frame_num >= 0 and f.exposure_start_pc > settled_after for f in frames.values()):
                    break
            medians: dict[str, float] = {}
            for sn, f in frames.items():
                img = np.frombuffer(f.data, dtype=np.uint8, count=f.width * f.height).reshape(f.height, f.width)
                if sn == hs_serial:  # 贴回全幅画布的高速帧：只量 ROI
                    roi = self._hs_info["roi"]
                    img = img[roi["offset_y"]:roi["offset_y"] + roi["height"],
                              roi["offset_x"]:roi["offset_x"] + roi["width"]]
                medians[sn] = float(np.median(img[::4, ::4]))
            first = first or medians
            if rounds == _AUTO_GAIN_MAX_ROUNDS:
                break
            changed = False
            for sn, median in medians.items():
                err_db = 20.0 * math.log10(AUTO_GAIN_TARGET_MEDIAN / max(median, 1.0))
                if abs(err_db) <= _AUTO_GAIN_TOLERANCE_DB:
                    continue
                gain_max, shift_min, shift_max = limits[sn]
                new_total = min(max(total_db[sn] + err_db, shift_min), gain_max + shift_max)
                if abs(new_total - total_db[sn]) < 0.05:
                    continue  # 已经顶在增益上/下限
                analog = min(max(new_total, 0.0), gain_max)
                cam = self._cameras[sn]
                _check(cam.MV_CC_SetFloatValue("Gain", analog), f"Gain={analog:.2f}({sn})")
                _check(cam.MV_CC_SetFloatValue("DigitalShift", new_total - analog),
                       f"DigitalShift={new_total - analog:.2f}({sn})")
                total_db[sn] = new_total
                changed = True
            if not changed:
                break

        for sn, median in medians.items():
            err_db = 20.0 * math.log10(AUTO_GAIN_TARGET_MEDIAN / max(median, 1.0))
            warn = ""
            if abs(err_db) > _AUTO_GAIN_TOLERANCE_DB:
                gain_max, shift_min, shift_max = limits[sn]
                warn = (f"  ⚠ 还差 {err_db:+.1f}dB：总增益 {total_db[sn]:.1f}dB"
                        f"（可调 {shift_min:.1f}~{gain_max + shift_max:.1f}dB），顶到头了就改曝光")
            print(f"  自动增益 {sn}: 灰度中位数 {first[sn]:.0f} → {median:.0f}"
                  f"（目标 {AUTO_GAIN_TARGET_MEDIAN:.0f}）{warn}")

    @property
    def fps(self) -> float:
        return float(self._fps)

    def close(self) -> None:
        """停止所有 grabber 并关闭所有相机。"""
        self._stop.set()
        if self._action_trigger is not None:
            # 子进程版要走一趟 stdin 停止协议 + 回传统计，给足时间（线程版几 ms 就停）
            self._action_trigger.join(timeout=5.0)
        for g in self._grabbers.values():
            g.stop()
        for g in self._grabbers.values():
            g.join(timeout=2.0)
        # 高速录像等取流线程都停了再收尾，最后几帧才不会漏写
        self._finish_high_speed_recording()
        # 线程都停下之后把丢帧计数快照下来：run_tracker 的最终统计跑在 with 块之外，
        # 下面一清 _grabbers/_action_trigger，不快照就只剩 0。
        if self._final_stats is None:
            self._final_stats = self.stats()
        self._action_trigger = None
        for cam in self._cameras.values():
            close_camera(cam)
        self._cameras.clear()
        self._grabbers.clear()
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
