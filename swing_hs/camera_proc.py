# -*- coding: utf-8 -*-
"""swing_hs 相机进程：一台相机一个进程。

为什么拆进程：Python 有 GIL，4 × 100 fps 取帧 + 识别挤在一个进程里会互相饿死（1006 hs405_trial 实测
4 条忙线程把 40 fps 同步组拖到 5 fps；record_swing_hs.py 一台一个进程 100 fps 0 丢帧）。

流程：
  1. 开相机，全幅、软触发：自动增益（灰度中位数拉到 60，同 tracker 启动自动增益）→ 全幅样张，
     认 AprilTag，把角点发给主进程定车位；
  2. 等主进程发 ROI（按车位 + 黑标包络 + 两块 tag 算的）→ ROI + action 触发，读回帧率上限；
  3. 开录：取帧线程 + 识别线程。帧时间 = 曝光开始 + 读回曝光 / 2（曝光中点），取到帧时算这一次，下游
     （关节角插值、格号、发给主进程的时间、小块记录）一律直接用、不再补半曝光（同 tracker，用户 1010 定）。
     识别线程每帧：帧时间的关节角（ROS 桥 UDP 转来）+ 车位姿（主进程共享）→ FK 黑标锚 → 投到本相机 → 在原始 Bayer ROI 帧上截 ±(48+4) px 小块解码 → 找黑方块候选
     （与 tracker 在线黑标、收场工具同一个 racket_ht_black_marker._marker_candidates）；每 tag_every 帧在
     tag 窗里认一次 tag。结果是小消息，发主进程三角化；
  4. 可选：每帧小块原样存盘（patches_<serial>.bin，离线换算法重认）、整幅 ROI 原始帧存盘（hs_<serial>.*，
     与 HighSpeedRecorder / test_src/hs_recording.py 同格式）。
"""

from __future__ import annotations

import ctypes
import math
import queue
import signal
import struct
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from swing_hs import geometry
from swing_hs.joints import JointBuffer, start_joint_listener
from swing_hs.winutil import ABOVE_NORMAL_PRIORITY_CLASS, HighResolutionTimer, set_priority, write_json

#: 每帧索引（同 src/ball_grabber.HIGH_SPEED_INDEX_FORMAT）：frame_num, dev_timestamp, exposure_start_pc, arrival_perf, lost_packet
INDEX_FORMAT = "<qQddq"
#: 小块记录头：slot, frame_num, 帧时间（曝光中点，PC perf）, 全幅 x0, 全幅 y0, w, h（后接 w*h 字节原始 Bayer）
PATCH_HEADER = struct.Struct("<qqdiiii")
#: 海康 Bayer 命名与 OpenCV 相反（同 src/ball_grabber.frame_to_numpy）
BAYER_TO_BGR = {
    0x01080008: "COLOR_BayerGB2BGR",   # BayerGR8（18F MV-CS032-60GC 实测）
    0x01080009: "COLOR_BayerBG2BGR",   # BayerRG8
    0x0108000A: "COLOR_BayerGR2BGR",   # BayerGB8
    0x0108000B: "COLOR_BayerRG2BGR",   # BayerBG8
}
MONO8 = 0x01080001
#: 截块在候选半径外多留的解码边（去马赛克在块边上会失真）
PATCH_BORDER_PX = 4
#: 识别线程等关节角追上本帧曝光中点的最长时长（ROS 网络延时 + 200 Hz 采样间隔通常 < 20 ms）
JOINT_WAIT_S = 0.08
#: 待识别帧队列上限（帧）：识别跟不上时丢最老的，计数
PROCESS_QUEUE_FRAMES = 200
#: 原始帧写盘队列上限（帧）
RAW_QUEUE_FRAMES = 3000
#: 开录前设备时钟锁存次数（取往返最短的作初始偏移），之后每秒一次
CLOCK_LATCH_INITIAL = 7
CLOCK_LATCH_PERIOD_S = 1.0
#: 自动增益（同 src/ball_grabber.SyncCapture.auto_gain）
AUTO_GAIN_TOLERANCE_DB = 1.0
AUTO_GAIN_MAX_ROUNDS = 8


def bayer_code(pixel_type: int) -> int:
    import cv2

    if pixel_type == MONO8:
        return cv2.COLOR_GRAY2BGR
    return getattr(cv2, BAYER_TO_BGR[pixel_type])


def clock_offset_from_samples(samples: list[tuple[float, float, int]], tick_hz: int) -> Optional[float]:
    """设备钟→perf 的偏移（perf = dev/tick + offset），取往返最短的那一次锁存。"""
    best = None
    for t_before, t_after, dev in samples:
        if dev <= 0:
            continue
        rtt = t_after - t_before
        if best is None or rtt < best[0]:
            best = (rtt, 0.5 * (t_before + t_after) - dev / float(tick_hz))
    return None if best is None else best[1]


def slot_of(frame_t: float, t0: float, period: float) -> int:
    """触发栅格上的格号（帧时间 = 曝光中点，比触发发出晚 ~0.06 ms + 半个曝光，远小于半格）。"""
    return int(round((frame_t - t0) / period))


def detect_marker_in_roi(
    raw: np.ndarray,
    roi_origin: tuple[int, int],
    anchor_uv: np.ndarray,
    code: int,
    *,
    radius: int,
) -> tuple[list[tuple], Optional[tuple[int, int, np.ndarray]]]:
    """原始 Bayer ROI 帧上，锚点（全幅像素）附近截块找黑方块候选。

    块的左上角对齐到偶数（ROI 偏移本身是偶数），Bayer 相位与全幅一致；块比候选半径多留
    PATCH_BORDER_PX 的解码边。候选检测 = racket_ht_black_marker._marker_candidates（tracker 在线
    黑标同参数：local_radius=48、不做大范围搜索）。
    返回 (候选行 [(u, v, score, area, bx, by, bw, bh)，全幅像素], (块左上全幅 x, y, 原始块) 或 None)。
    """
    import cv2

    rbm = geometry.load_test_src("racket_ht_black_marker")
    ox, oy = roi_origin
    height, width = raw.shape[:2]
    au, av = float(anchor_uv[0]) - ox, float(anchor_uv[1]) - oy
    if not (math.isfinite(au) and math.isfinite(av)):
        return [], None
    if au < -radius or au >= width + radius or av < -radius or av >= height + radius:
        return [], None
    half = radius + PATCH_BORDER_PX
    x0 = max(0, (int(math.floor(au - half)) // 2) * 2)
    y0 = max(0, (int(math.floor(av - half)) // 2) * 2)
    x1 = min(width, x0 + 2 * half + 2)
    y1 = min(height, y0 + 2 * half + 2)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return [], None
    patch = np.ascontiguousarray(raw[y0:y1, x0:x1])
    bgr = cv2.cvtColor(patch, code)
    found = rbm._marker_candidates(bgr, (au - x0, av - y0), local_radius=radius, broad_radius=None)
    gx, gy = ox + x0, oy + y0
    rows = [
        (round(c.uv[0] + gx, 2), round(c.uv[1] + gy, 2), round(float(c.score), 1), int(c.area),
         int(c.bbox_xywh[0] + gx), int(c.bbox_xywh[1] + gy), int(c.bbox_xywh[2]), int(c.bbox_xywh[3]))
        for c in found
    ]
    return rows, (gx, gy, patch)


def tag_windows(
    camera,
    pose: tuple[float, float, float],
    tag_corners_car: dict[int, np.ndarray],
    roi: list[int],
    pad_px: float,
) -> list[tuple[int, int, int, int]]:
    """按车位姿预测各 tag 在本相机的位置 → ROI 内的检测窗 (x0, y0, x1, y1)（ROI 像素、偶数对齐），重叠的合并。"""
    ox, oy, w, h = roi
    boxes: list[list[int]] = []
    for corners in tag_corners_car.values():
        uv = geometry.project_m(camera, geometry.car_to_world(corners, pose))
        if not np.all(np.isfinite(uv)):
            continue
        x0 = max(0, (int(math.floor(uv[:, 0].min() - pad_px - ox)) // 2) * 2)
        y0 = max(0, (int(math.floor(uv[:, 1].min() - pad_px - oy)) // 2) * 2)
        x1 = min(w, int(math.ceil(uv[:, 0].max() + pad_px - ox)))
        y1 = min(h, int(math.ceil(uv[:, 1].max() + pad_px - oy)))
        x1 += x1 % 2 if x1 < w else 0
        y1 += y1 % 2 if y1 < h else 0
        if x1 - x0 >= 32 and y1 - y0 >= 32:
            boxes.append([x0, y0, x1, y1])
    merged: list[list[int]] = []
    for box in sorted(boxes):
        for m in merged:
            if box[0] < m[2] and box[2] > m[0] and box[1] < m[3] and box[3] > m[1]:
                m[0], m[1] = min(m[0], box[0]), min(m[1], box[1])
                m[2], m[3] = max(m[2], box[2]), max(m[3], box[3])
                break
        else:
            merged.append(list(box))
    return [tuple(m) for m in merged]


def detect_tags(carloc, image_bgr: np.ndarray, origin: tuple[int, int], tag_ids: set[int]) -> list[tuple[int, list]]:
    """全幅样张：CarLocalizer 的 ArUco 检测（同 tracker，含暗光救援）认 tag；同一 tag 出现两次作废。"""
    detections = carloc.detect(image_bgr)
    found = [d for d in detections if d.tag_id in tag_ids]
    ids = [d.tag_id for d in found]
    if len(ids) != len(set(ids)):
        return []
    gx, gy = origin
    return [(d.tag_id, [[float(u) + gx, float(v) + gy] for u, v in np.asarray(d.corners).reshape(4, 2)]) for d in found]


def make_window_tag_detector():
    """tag 窗用的 ArUco 检测器：参数同 src/car_localizer，只是不设 maxMarkerPerimeterRate=0.6——那是相对整幅
    2048 px 的比例（周长上限 ~1229 px），tag 窗只有 ~200 px，50 px 的 tag 周长率 ~1，会被当成太大扔掉（1010 假相机实测）。"""
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.perspectiveRemovePixelPerCell = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.3
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)


def detect_tags_in_window(detector, gray: np.ndarray, origin: tuple[int, int], tag_ids: set[int]) -> list[tuple[int, list]]:
    """tag 窗里认 tag：原图认不出配置里的 tag 就拉一次局部对比度再认（同 CarLocalizer 的暗光救援，同一个增强），
    角点换回全幅像素；同一 tag 出现两次整窗作废。"""
    from src.car_localizer import CarLocalizer

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or not any(int(i) in tag_ids for i in ids.ravel()):
        corners, ids, _ = detector.detectMarkers(CarLocalizer._enhance_for_low_light(gray))
    if ids is None:
        return []
    found = [(int(i), np.asarray(c, dtype=np.float64).reshape(4, 2)) for i, c in zip(ids.ravel(), corners) if int(i) in tag_ids]
    if len({tid for tid, _ in found}) != len(found):
        return []
    gx, gy = origin
    return [(tid, [[float(u) + gx, float(v) + gy] for u, v in c]) for tid, c in found]


class CameraWorker:
    """一台相机：开 → 自动增益 → 全幅样张 → ROI + action 触发 → 取帧 / 识别 / 写盘 → 收尾。"""

    def __init__(self, spec: dict, out_q, ctrl_q, shared_car, t0, stop_evt, start_evt) -> None:
        import src.mvs_env  # noqa: F401  （MvImport / DLL 路径，必须在 SDK 之前）
        import CameraParams_header as cph
        from MvCameraControl_class import MvCamera

        self.cph = cph
        self.MvCamera = MvCamera
        self.spec = spec
        self.serial = str(spec["serial"])
        self.out_q = out_q
        self.ctrl_q = ctrl_q
        self.shared_car = shared_car
        self.t0 = t0
        self.stop_evt = stop_evt
        self.start_evt = start_evt
        self.cam: Any = None
        self.out_dir = Path(spec["out_dir"])
        self.period = 1.0 / float(spec["fps"])
        self.half_exposure_s: Optional[float] = None   # 读回曝光 / 2，configure_streaming 里定
        self.tick_hz = 0
        self.sensor: dict = {}
        self.incs: dict = {}
        self.readback: dict = {}
        self.roi: list[int] = [0, 0, 0, 0]
        self.pixel_type: Optional[int] = None
        self.frame_shape: Optional[tuple[int, int]] = None
        self.clock_samples: list[tuple[float, float, int]] = []
        self._clock_lock = threading.Lock()
        self.offset_s = 0.0
        self.snapshot_bgr: Optional[np.ndarray] = None
        # 识别用
        rbm = geometry.load_test_src("racket_ht_black_marker")
        eab = geometry.load_test_src("extract_arm_bag")
        self.car = str(spec["car"])
        self.fk = eab.fk
        self.z_offset = geometry.marker_z_offset_m(self.car)
        self.camera = rbm._load_cameras(Path(spec["calib_path"]), list(spec["all_serials"]))[self.serial]
        vehicle = geometry.load_vehicle_reference(Path(spec["vehicle_path"]))
        self.tag_corners = geometry.tag_corners_car(vehicle, 1.0)
        self.tag_ids = set(self.tag_corners)
        from src.car_localizer import CarLocalizer

        self.carloc = CarLocalizer(spec["calib_path"], spec["vehicle_path"])
        self.tag_detector = make_window_tag_detector()
        self.joints = JointBuffer(eab.CAR_MODELS[self.car].slot_joint_names)
        self.radius = int(spec["crop_radius_px"])
        self.tag_every = max(1, int(spec["tag_every_frames"]))
        self.tag_pad = float(spec["tag_window_pad_px"])
        self.tags_in_roi = True
        # 取帧 → 识别
        self._frames: deque = deque()
        self._frames_cv = threading.Condition()
        self._grab_done = threading.Event()
        self._raw_q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=RAW_QUEUE_FRAMES)
        self._patch_q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=4000)
        self.stats: dict[str, Any] = {
            "received": 0, "missing": 0, "lost_packets": 0, "lost_packet_frames": 0, "size_mismatch": 0,
            "proc_dropped": 0, "processed": 0, "with_candidates": 0, "no_car_pose": 0, "no_joint_state": 0,
            "out_of_roi": 0, "tag_runs": 0, "tag_hits": 0, "raw_written": 0, "raw_dropped": 0,
            "patches_written": 0, "patches_dropped": 0, "proc_ms_sum": 0.0, "proc_ms_max": 0.0,
            "joint_wait_ms_sum": 0.0, "queue_max": 0, "before_trigger": 0,
        }
        self._last_frame_num: Optional[int] = None

    # ── SDK 小封装 ──
    def _check(self, ret: int, what: str) -> None:
        if int(ret) != 0:
            raise RuntimeError(f"[{self.serial}] {what} 失败 ret=0x{int(ret) & 0xFFFFFFFF:08X}")

    def _get_int(self, node: str) -> dict:
        v = self.cph.MVCC_INTVALUE_EX()
        self._check(self.cam.MV_CC_GetIntValueEx(node, v), f"Get {node}")
        return {"cur": int(v.nCurValue), "min": int(v.nMin), "max": int(v.nMax), "inc": int(v.nInc)}

    def _set_int(self, node: str, value: int) -> None:
        self._check(self.cam.MV_CC_SetIntValueEx(node, int(value)), f"{node}={value}")

    def _get_float(self, node: str) -> dict:
        v = self.cph.MVCC_FLOATVALUE()
        self._check(self.cam.MV_CC_GetFloatValue(node, v), f"Get {node}")
        return {"cur": float(v.fCurValue), "min": float(v.fMin), "max": float(v.fMax)}

    def _set_float(self, node: str, value: float) -> None:
        self._check(self.cam.MV_CC_SetFloatValue(node, float(value)), f"{node}={value}")

    def _set_enum(self, node: str, value: str) -> None:
        self._check(self.cam.MV_CC_SetEnumValueByString(node, value), f"{node}={value}")

    def _try(self, fn, *a) -> None:
        try:
            fn(*a)
        except Exception:
            pass

    def _send(self, kind: str, payload: Any) -> None:
        self.out_q.put((kind, self.serial, payload))

    # ── 打开 / 配置 ──
    def open(self) -> None:
        cph = self.cph
        self.MvCamera.MV_CC_Initialize()
        st_list = cph.MV_CC_DEVICE_INFO_LIST()
        self._check(self.MvCamera.MV_CC_EnumDevices(cph.MV_GIGE_DEVICE, st_list), "EnumDevices")
        for i in range(int(st_list.nDeviceNum)):
            info = ctypes.cast(st_list.pDeviceInfo[i], ctypes.POINTER(cph.MV_CC_DEVICE_INFO)).contents
            raw = bytes(info.SpecialInfo.stGigEInfo.chSerialNumber)
            if raw.split(b"\x00", 1)[0].decode("ascii", "ignore").strip() != self.serial:
                continue
            cam = self.MvCamera()
            self._check(cam.MV_CC_CreateHandle(info), "CreateHandle")
            ret = cam.MV_CC_OpenDevice(3, 0)  # MV_ACCESS_Control
            if int(ret) != 0:
                cam.MV_CC_DestroyHandle()
                raise RuntimeError(f"[{self.serial}] OpenDevice 失败 ret=0x{int(ret) & 0xFFFFFFFF:08X}"
                                   "（tracker / MVS 客户端 / 别的脚本是不是还开着这台相机？）")
            self.cam = cam
            break
        else:
            raise RuntimeError(f"未找到序列号为 {self.serial} 的相机")
        cam = self.cam
        spec = self.spec
        ps = int(cam.MV_CC_GetOptimalPacketSize())
        if ps > 0:
            self._try(self._set_int, "GevSCPSPacketSize", ps)
        self._try(cam.MV_GIGE_SetResend, True, 100, 50)
        self._set_int("GevSCPD", int(spec["gev_scpd"]))
        self._set_enum("ExposureAuto", "Off")
        self._set_float("ExposureTime", float(spec["exposure_us"]))
        self._set_enum("GainAuto", "Off")
        self._check(cam.MV_CC_SetBoolValue("DigitalShiftEnable", True), "DigitalShiftEnable")
        self._check(cam.MV_CC_SetBoolValue("ReverseX", False), "ReverseX")
        self._check(cam.MV_CC_SetBoolValue("ReverseY", False), "ReverseY")
        self._check(cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True), "AcquisitionFrameRateEnable")
        self._set_float("AcquisitionFrameRate", float(spec["acquisition_frame_rate"]))
        self._try(self._set_enum, "TriggerSelector", "FrameBurstStart")
        self._set_enum("TriggerMode", "On")
        self._try(self._set_float, "TriggerDelay", 0.0)
        self._try(cam.MV_CC_SetBoolValue, "TriggerCacheEnable", False)
        self._check(cam.MV_CC_SetImageNodeNum(int(spec["sdk_image_nodes"])), "SetImageNodeNum")
        self.tick_hz = self._get_int("GevTimestampTickFrequency")["cur"]
        self.sensor = {"width_max": self._get_int("WidthMax")["cur"], "height_max": self._get_int("HeightMax")["cur"]}
        self.incs = {node: self._get_int(node)["inc"] for node in ("OffsetX", "OffsetY", "Width", "Height")}
        for node, step in (("OffsetX", geometry.ROI_X_STEP), ("Width", geometry.ROI_X_STEP),
                           ("OffsetY", geometry.ROI_Y_STEP), ("Height", geometry.ROI_Y_STEP)):
            if step % max(1, self.incs[node]):
                raise RuntimeError(f"[{self.serial}] {node} 步进 {self.incs[node]} 不整除规划步进 {step}")

    def _set_roi(self, ox: int, oy: int, w: int, h: int) -> None:
        # 先归零偏移再改宽高，最后才挪偏移（否则新宽高 + 旧偏移可能越界被拒）
        self._set_int("OffsetX", 0)
        self._set_int("OffsetY", 0)
        self._set_int("Width", w)
        self._set_int("Height", h)
        self._set_int("OffsetX", ox)
        self._set_int("OffsetY", oy)
        got = [self._get_int(k)["cur"] for k in ("OffsetX", "OffsetY", "Width", "Height")]
        if got != [ox, oy, w, h]:
            raise RuntimeError(f"[{self.serial}] ROI 没按要求生效：要 {[ox, oy, w, h]}，相机实际 {got}")

    def _grab_soft(self, timeout_ms: int = 2000) -> tuple[np.ndarray, int]:
        """软触发一帧（全幅），返回 (uint8 数组, pixel_type)。"""
        st = self.cph.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(st), 0, ctypes.sizeof(st))
        self._check(self.cam.MV_CC_SetCommandValue("TriggerSoftware"), "TriggerSoftware")
        self._check(self.cam.MV_CC_GetImageBuffer(st, timeout_ms), "GetImageBuffer(软触发)")
        try:
            info = st.stFrameInfo
            w = int(info.nExtendWidth or info.nWidth)
            h = int(info.nExtendHeight or info.nHeight)
            n = int(info.nFrameLenEx or info.nFrameLen)
            data = ctypes.string_at(st.pBufAddr, n)
            return np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w).copy(), int(info.enPixelType)
        finally:
            self.cam.MV_CC_FreeImageBuffer(st)

    def auto_gain_and_snapshot(self) -> dict:
        """全幅软触发：增益拉到画面灰度中位数 = 目标（模拟 Gain 先用满、不够补 DigitalShift，同 tracker）→ 全幅样张认 tag。"""
        import cv2

        target = float(self.spec["auto_gain_target"])
        gain = self._get_float("Gain")
        shift = self._get_float("DigitalShift")
        # 上下限往里收 0.01 dB：写正好等于 fMax 的值会换算进位越界被拒（同 ball_grabber）
        gain_max = math.floor(gain["max"] * 100) / 100
        shift_min, shift_max = math.ceil(shift["min"] * 100) / 100, math.floor(shift["max"] * 100) / 100
        analog = min(gain_max, float(self.spec["gain_db"]))
        start_shift = min(max(float(self.spec.get("digital_shift") or 0.0), shift_min), shift_max)
        self._set_float("Gain", analog)
        self._set_float("DigitalShift", start_shift)
        total = analog + start_shift
        self._set_roi(0, 0, self.sensor["width_max"], self.sensor["height_max"])
        self._set_enum("TriggerSource", "Software")
        self._check(self.cam.MV_CC_StartGrabbing(), "StartGrabbing(全幅)")
        first = median = None
        try:
            for _ in range(AUTO_GAIN_MAX_ROUNDS):
                self._grab_soft()                    # 改完增益后的第一帧丢掉
                raw, pixel_type = self._grab_soft()
                median = float(np.median(raw[::4, ::4]))
                first = median if first is None else first
                err_db = 20.0 * math.log10(target / max(median, 1.0))
                if abs(err_db) <= AUTO_GAIN_TOLERANCE_DB:
                    break
                new_total = min(max(total + err_db, shift_min), gain_max + shift_max)
                if abs(new_total - total) < 0.05:
                    break                            # 顶在上 / 下限
                analog = min(max(new_total, 0.0), gain_max)
                self._set_float("Gain", analog)
                self._set_float("DigitalShift", new_total - analog)
                total = new_total
            self._grab_soft()
            raw, pixel_type = self._grab_soft()
        finally:
            self.cam.MV_CC_StopGrabbing()
        median = float(np.median(raw[::4, ::4]))
        err_db = 20.0 * math.log10(target / max(median, 1.0))
        cv2.imwrite(str(self.out_dir / f"full_{self.serial}.png"), raw)
        self.pixel_type = pixel_type
        bgr = cv2.cvtColor(raw, bayer_code(pixel_type))
        self.snapshot_bgr = bgr
        tags = detect_tags(self.carloc, bgr, (0, 0), self.tag_ids)
        return {
            "t": time.perf_counter(),
            "median_first": first, "median": median, "target": target,
            "gain_db": round(self._get_float("Gain")["cur"], 3),
            "digital_shift_db": round(self._get_float("DigitalShift")["cur"], 3),
            "gain_short_db": round(err_db, 2) if abs(err_db) > AUTO_GAIN_TOLERANCE_DB else 0.0,
            "gain_range_db": [shift_min, gain_max + shift_max],
            "pixel_type": pixel_type,
            "tags": tags,
            "incs": self.incs,
            "sensor": self.sensor,
        }

    def configure_streaming(self, roi: list[int]) -> dict:
        ox, oy, w, h = (int(v) for v in roi)
        self.roi = [ox, oy, w, h]
        self._set_roi(ox, oy, w, h)
        spec = self.spec
        self._set_enum("TriggerSource", str(spec["action_trigger_source"]))
        self._set_int("ActionDeviceKey", int(spec["action_device_key"]))
        self._set_int("ActionGroupKey", int(spec["action_group_key"]))
        self._set_int("ActionGroupMask", int(spec["action_group_mask"]))
        exposure_us = self._get_float("ExposureTime")["cur"]
        if not (math.isfinite(exposure_us) and exposure_us > 0.0):
            raise RuntimeError(f"[{self.serial}] 读不回曝光时间（{exposure_us}），帧时间无法定到曝光中点")
        self.half_exposure_s = 0.5e-6 * exposure_us
        self.readback = {
            "roi": {"offset_x": ox, "offset_y": oy, "width": w, "height": h},
            "resulting_fps": round(self._get_float("ResultingFrameRate")["cur"], 3),
            "sensor_fps_model": round(geometry.sensor_max_fps(h), 3),
            "exposure_us": round(exposure_us, 3),
            "gain_db": round(self._get_float("Gain")["cur"], 3),
            "digital_shift_db": round(self._get_float("DigitalShift")["cur"], 3),
            "acquisition_frame_rate": round(self._get_float("AcquisitionFrameRate")["cur"], 3),
            "packet_size": self._get_int("GevSCPSPacketSize")["cur"],
            "tick_hz": self.tick_hz,
            "sensor": self.sensor,
            "incs": self.incs,
        }
        for _ in range(CLOCK_LATCH_INITIAL):
            self.latch_clock()
        off = clock_offset_from_samples(self.clock_samples, self.tick_hz)
        self.offset_s = 0.0 if off is None else off
        return self.readback

    def write_overlay(self, overlay: dict) -> None:
        """全幅样张上画 ROI、黑标包络（名义车位）、tag 角点，半分辨率存 roi_<serial>.jpg 目检覆盖。"""
        import cv2

        if self.snapshot_bgr is None:
            return
        img = self.snapshot_bgr.copy()
        ox, oy, w, h = self.roi
        for u, v in overlay.get("marker_uv", []):
            if math.isfinite(u) and math.isfinite(v):
                cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), -1)
        for poly in overlay.get("tags_uv", []):
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (255, 0, 255), 3)
        cv2.rectangle(img, (ox, oy), (ox + w - 1, oy + h - 1), (0, 255, 255), 4)
        cv2.putText(img, f"{self.serial} ROI {w}x{h}@({ox},{oy})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 255, 255), 3)
        cv2.imwrite(str(self.out_dir / f"roi_{self.serial}.jpg"),
                    cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ── 时钟锁存 ──
    def latch_clock(self) -> None:
        t0 = time.perf_counter()
        ret = self.cam.MV_CC_SetCommandValue("GevTimestampControlLatch")
        t1 = time.perf_counter()
        if int(ret) != 0:
            return
        v = self.cph.MVCC_INTVALUE_EX()
        if int(self.cam.MV_CC_GetIntValueEx("GevTimestampValue", v)) != 0:
            return
        with self._clock_lock:
            self.clock_samples.append((t0, t1, int(v.nCurValue)))

    def _clock_loop(self, stop: threading.Event) -> None:
        while not stop.wait(CLOCK_LATCH_PERIOD_S):
            self.latch_clock()
            with self._clock_lock:
                recent = self.clock_samples[-10:]
            off = clock_offset_from_samples(recent, self.tick_hz)
            if off is not None:
                self.offset_s = off

    # ── 车位姿（主进程共享） ──
    def car_pose(self) -> Optional[tuple[float, float, float]]:
        with self.shared_car.get_lock():
            x, y, _z, yaw, _t, valid = self.shared_car[:6]
        return (x, y, yaw) if valid > 0.5 else None

    # ── 取帧 ──
    def _grab_loop(self) -> None:
        cph = self.cph
        cam = self.cam
        st = cph.MV_FRAME_OUT()
        frame_bytes = None
        record_raw = bool(self.spec.get("raw_dir"))
        stats = self.stats
        try:
            while True:
                ctypes.memset(ctypes.byref(st), 0, ctypes.sizeof(st))
                ret = cam.MV_CC_GetImageBuffer(st, 100)
                if int(ret) != 0:
                    if self.stop_evt.is_set():
                        break       # 触发已停、SDK 缓冲取空
                    continue
                try:
                    arrival = time.perf_counter()
                    info = st.stFrameInfo
                    n = int(info.nFrameLenEx or info.nFrameLen)
                    data = ctypes.string_at(st.pBufAddr, n)
                    fnum = int(info.nFrameNum)
                    dev = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
                    lost = int(info.nLostPacket)
                    if frame_bytes is None:
                        frame_bytes = n
                        self.pixel_type = int(info.enPixelType)
                        self.frame_shape = (int(info.nExtendHeight or info.nHeight), int(info.nExtendWidth or info.nWidth))
                finally:
                    cam.MV_CC_FreeImageBuffer(st)
                exposure_start_pc = dev / self.tick_hz + self.offset_s
                frame_t = exposure_start_pc + self.half_exposure_s    # 帧时间 = 曝光中点，只在这里算
                stats["received"] += 1
                if lost:
                    stats["lost_packets"] += lost
                    stats["lost_packet_frames"] += 1
                if self._last_frame_num is not None:
                    gap = fnum - self._last_frame_num - 1
                    if 0 < gap < 100000:
                        stats["missing"] += gap
                self._last_frame_num = fnum
                if n != frame_bytes:
                    stats["size_mismatch"] += 1
                    continue
                if record_raw:
                    try:
                        # 原始帧索引同 HighSpeedRecorder 格式（字段就叫 exposure_start_pc），只给 test_src/hs_recording.py 读
                        self._raw_q.put_nowait(((fnum, dev, exposure_start_pc, arrival, lost), data))
                    except queue.Full:
                        stats["raw_dropped"] += 1
                t0 = self.t0.value
                if t0 == 0.0:          # 触发进程还没起：不该有帧（触发模式），有也不知道是哪一格
                    stats["before_trigger"] += 1
                    continue
                h, w = self.frame_shape
                raw = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
                slot = slot_of(frame_t, t0, self.period)
                with self._frames_cv:
                    if len(self._frames) >= PROCESS_QUEUE_FRAMES:
                        self._frames.popleft()
                        stats["proc_dropped"] += 1
                    self._frames.append((slot, fnum, frame_t, raw))
                    stats["queue_max"] = max(stats["queue_max"], len(self._frames))
                    self._frames_cv.notify()
        finally:
            self._grab_done.set()
            with self._frames_cv:
                self._frames_cv.notify_all()

    # ── 识别 ──
    def _next_frame(self) -> Optional[tuple]:
        with self._frames_cv:
            while not self._frames:
                if self._grab_done.is_set():
                    return None
                self._frames_cv.wait(0.05)
            return self._frames.popleft()

    def _wait_joints(self, frame_t: float) -> float:
        """等关节角追上本帧帧时间（关节流断了就不等）；返回等了多久（s）。"""
        start = time.perf_counter()
        deadline = frame_t + JOINT_WAIT_S
        while True:
            latest = self.joints.latest_pc()
            now = time.perf_counter()
            if latest is not None and latest >= frame_t:
                break
            if now >= deadline or not self.joints.alive(now) or self._grab_done.is_set():
                break
            time.sleep(0.002)
        return time.perf_counter() - start

    def _process_loop(self) -> None:
        stats = self.stats
        code = bayer_code(self.pixel_type if self.pixel_type is not None else 0x01080008)
        code_for = self.pixel_type
        roi_origin = (self.roi[0], self.roi[1])
        record_patches = bool(self.spec.get("record_patches"))
        while True:
            item = self._next_frame()
            if item is None:
                break
            slot, fnum, frame_t, raw = item
            if self.pixel_type != code_for:
                code_for = self.pixel_type
                code = bayer_code(code_for)
            stats["joint_wait_ms_sum"] += 1e3 * self._wait_joints(frame_t)
            t_start = time.perf_counter()
            pose = self.car_pose()
            q = self.joints.at_pc(frame_t)
            anchor_mm = None
            rows: list[tuple] = []
            if pose is None:
                status = "no_car_pose"
                stats["no_car_pose"] += 1
            elif q is None:
                status = "no_joint_state"
                stats["no_joint_state"] += 1
            else:
                marker = self.fk(q, car=self.car)["marker"]
                anchor_m = geometry.car_to_world(marker, pose, self.z_offset)[0]
                anchor_mm = [round(float(v) * 1000.0, 1) for v in anchor_m]
                anchor_uv = geometry.project_m(self.camera, anchor_m)[0]
                rows, patch = detect_marker_in_roi(raw, roi_origin, anchor_uv, code, radius=self.radius)
                if patch is None:
                    status = "out_of_roi"
                    stats["out_of_roi"] += 1
                else:
                    status = "ok"
                    if rows:
                        stats["with_candidates"] += 1
                    if record_patches:
                        try:
                            self._patch_q.put_nowait((slot, fnum, frame_t, patch))
                        except queue.Full:
                            stats["patches_dropped"] += 1
            self.out_q.put(("f", self.serial, (
                slot, fnum, frame_t, status, anchor_mm,
                None if q is None else [round(float(v), 6) for v in q], rows,
            )))
            if pose is not None and self.tags_in_roi and slot % self.tag_every == 0:
                self._detect_tags(raw, pose, slot, frame_t, code)
            stats["processed"] += 1
            dt_ms = 1e3 * (time.perf_counter() - t_start)
            stats["proc_ms_sum"] += dt_ms
            stats["proc_ms_max"] = max(stats["proc_ms_max"], dt_ms)

    def _detect_tags(self, raw: np.ndarray, pose, slot: int, frame_t: float, code: int) -> None:
        import cv2

        self.stats["tag_runs"] += 1
        found: list[tuple[int, list]] = []
        ox, oy = self.roi[0], self.roi[1]
        for x0, y0, x1, y1 in tag_windows(self.camera, pose, self.tag_corners, self.roi, self.tag_pad):
            gray = cv2.cvtColor(cv2.cvtColor(np.ascontiguousarray(raw[y0:y1, x0:x1]), code), cv2.COLOR_BGR2GRAY)
            found.extend(detect_tags_in_window(self.tag_detector, gray, (ox + x0, oy + y0), self.tag_ids))
        ids = [tid for tid, _ in found]
        if len(ids) != len(set(ids)):
            found = []
        if found:
            self.stats["tag_hits"] += 1
        self.out_q.put(("t", self.serial, (slot, frame_t, found)))

    # ── 写盘 ──
    def _patch_writer(self) -> None:
        path = self.out_dir / f"patches_{self.serial}.bin"
        with open(path, "wb", buffering=1 << 20) as f:
            while True:
                item = self._patch_q.get()
                if item is None:
                    break
                slot, fnum, frame_t, (gx, gy, patch) = item
                h, w = patch.shape
                f.write(PATCH_HEADER.pack(slot, fnum, frame_t, gx, gy, w, h))
                f.write(patch.tobytes())
                self.stats["patches_written"] += 1

    def _raw_writer(self) -> None:
        base = Path(self.spec["raw_dir"]) / f"hs_{self.serial}"
        pack = struct.Struct(INDEX_FORMAT).pack
        with open(base.with_suffix(".bayer"), "wb", buffering=16 << 20) as fr, \
                open(base.with_suffix(".idx"), "wb", buffering=1 << 16) as fi:
            n = 0
            while True:
                item = self._raw_q.get()
                if item is None:
                    break
                rec, data = item
                fr.write(data)
                fi.write(pack(*rec))
                self.stats["raw_written"] += 1
                n += 1
                if n % 200 == 0:
                    fi.flush()

    def _write_raw_meta(self) -> None:
        if not self.spec.get("raw_dir"):
            return
        ox, oy, w, h = self.roi
        write_json(Path(self.spec["raw_dir"]) / f"hs_{self.serial}.json", {
            "serial": self.serial,
            "roi": {"offset_x": ox, "offset_y": oy, "width": w, "height": h},
            "slot_fps": float(self.spec["fps"]),
            "readback": self.readback,
            "files": {"index_format": INDEX_FORMAT,
                      "index_fields": ["frame_num", "dev_timestamp", "exposure_start_pc", "arrival_perf", "lost_packet"]},
            "recording": {"frame_shape": list(self.frame_shape) if self.frame_shape else None,
                          "pixel_type": self.pixel_type, "written": self.stats["raw_written"],
                          "dropped": self.stats["raw_dropped"]},
        })

    def _write_clock_csv(self) -> None:
        with self._clock_lock:
            rows = list(self.clock_samples)
        with open(self.out_dir / f"clock_{self.serial}.csv", "w", encoding="utf-8", newline="\n") as f:
            f.write("perf_before,perf_after,dev_ticks\n")
            for a, b, d in rows:
                f.write(f"{a:.9f},{b:.9f},{d}\n")

    def stats_snapshot(self) -> dict:
        s = dict(self.stats)
        s["queue"] = len(self._frames)
        s["joint_rx"] = self.joints.received
        s["joint_offset_s"] = self.joints.offset()
        if self.joints.name_mismatch is not None:
            s["joint_name_mismatch"] = self.joints.name_mismatch
        s["low_light_retries"] = int(getattr(self.carloc, "low_light_retries", 0))
        return s

    # ── 主流程 ──
    def _wait_ctrl(self, kind: str) -> Optional[Any]:
        while not self.stop_evt.is_set():
            try:
                msg_kind, payload = self.ctrl_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg_kind == "abort":
                return None
            if msg_kind == kind:
                return payload
        return None

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.spec.get("raw_dir"):
            Path(self.spec["raw_dir"]).mkdir(parents=True, exist_ok=True)
        self.open()
        try:
            self._send("snapshot", self.auto_gain_and_snapshot())
            plan = self._wait_ctrl("roi")
            if plan is None:
                return
            readback = self.configure_streaming(plan["roi"])
            self.tags_in_roi = bool(plan.get("tags_in_roi", True))
            self.write_overlay(plan.get("overlay") or {})
            self._send("ready", readback)
            while not self.start_evt.wait(0.1):
                if self.stop_evt.is_set():
                    return
            listener_stop = threading.Event()
            start_joint_listener(int(self.spec["joint_port"]), self.joints, listener_stop, name=f"joints-{self.serial}")
            self._check(self.cam.MV_CC_StartGrabbing(), "StartGrabbing")
            threads = [
                threading.Thread(target=self._grab_loop, name=f"grab-{self.serial}", daemon=True),
                threading.Thread(target=self._process_loop, name=f"proc-{self.serial}", daemon=True),
            ]
            if self.spec.get("record_patches"):
                threads.append(threading.Thread(target=self._patch_writer, name=f"patch-{self.serial}", daemon=True))
            if self.spec.get("raw_dir"):
                threads.append(threading.Thread(target=self._raw_writer, name=f"raw-{self.serial}", daemon=True))
            clock_stop = threading.Event()
            threads.append(threading.Thread(target=self._clock_loop, args=(clock_stop,), name=f"clock-{self.serial}",
                                            daemon=True))
            for th in threads:
                th.start()
            while not self.stop_evt.wait(1.0):
                self._send("stats", self.stats_snapshot())
            # 收尾：取帧线程把 SDK 缓冲取空后退出 → 识别线程处理完剩下的帧 → 写盘线程收尾
            threads[0].join(timeout=10.0)
            self.cam.MV_CC_StopGrabbing()
            threads[1].join(timeout=30.0)
            clock_stop.set()
            self._patch_q.put(None)
            self._raw_q.put(None)
            for th in threads[2:]:
                th.join(timeout=60.0)
            listener_stop.set()
            for _ in range(3):
                self.latch_clock()
            self._write_clock_csv()
            self._write_raw_meta()
            self._send("done", {**self.stats_snapshot(), "readback": self.readback})
        finally:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.cam.MV_CC_CloseDevice()
            finally:
                self.cam.MV_CC_DestroyHandle()
                self.MvCamera.MV_CC_Finalize()


def camera_main(spec: dict, out_q, ctrl_q, shared_car, t0, stop_evt, start_evt) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # 只认主进程的停止事件
    set_priority(ABOVE_NORMAL_PRIORITY_CLASS)
    try:
        with HighResolutionTimer():
            CameraWorker(spec, out_q, ctrl_q, shared_car, t0, stop_evt, start_evt).run()
    except BaseException as exc:  # noqa: BLE001 —— 任何异常都要报回主进程
        out_q.put(("error", str(spec.get("serial")), f"{exc!r}\n{traceback.format_exc()}"))
