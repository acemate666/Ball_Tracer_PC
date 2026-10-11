# -*- coding: utf-8 -*-
"""swing_hs 相机侧：每台相机一个取流进程（CameraWorker）+ DETECT_WORKERS 个黑标识别进程（detect_main）。

为什么拆进程：Python 有 GIL，4 × 100 fps 取帧 + 识别挤在一个进程里会互相饿死（1006 hs405_trial 实测
4 条忙线程把 40 fps 同步组拖到 5 fps；record_swing_hs.py 一台一个进程 100 fps 0 丢帧）。黑标候选检测夜里每块
~6.7 ms（增益顶满、噪点连通域多），再单独拆一个识别进程（用户 1011：「把黑标当成一个独立进程，不影响视频读流」），
取流进程只做轻活（每帧 < 0.5 ms）：识别跟不上时只在任务队列满了丢任务、计数，不拖取帧。

流程：
  1. 开相机，全幅、软触发：自动增益（灰度中位数拉到 60，同 tracker 启动自动增益）→ 全幅样张，
     认 AprilTag，把角点发给主进程定车位；
  2. 等主进程发 ROI（按车位 + 黑标包络 + 两块 tag 算的）→ ROI + action 触发，读回帧率上限；
  3. 开录：取帧线程 + 准备线程。帧时间 = 曝光开始 + 读回曝光 / 2（曝光中点），取到帧时算这一次，下游
     （关节角插值、格号、发给主进程的时间、小块记录）一律直接用、不再补半曝光（同 tracker，用户 1010 定）。
     准备线程每帧：帧时间的关节角（ROS 桥 UDP 转来）+ 车位姿（主进程共享）→ FK 黑标锚 → 投到本相机 → 在原始
     Bayer ROI 帧上截 ±(48+4) px 小块；tag 窗按触发格每 tag_every 帧一块累加平均（车静止，平均压夜里的噪声）。
     小块 / tag 平均图发给本相机的识别进程：解码 → 黑标 YOLO 候选（marker_yolo.py；1011 起替换阈值暗块候选器
     racket_ht_black_marker._marker_candidates，夜里那个满块噪点、在线解多数是错的）/ 认 tag，结果小消息发主进程三角化 / 解车位。
     一台相机配 DETECT_WORKERS 个识别进程，共用一个任务队列（阈值候选器挥拍时夜里每块 8–10 ms，一个进程贴着 100 fps 的 10 ms 预算；
     YOLO 每块 CPU 单线程几 ms）；
     任务带本相机的序号，结果到主进程的先后会乱几个，主进程按序号放行；
  4. 存盘：每帧小块原样存（patches_<serial>.bin，离线换算法重认，可关）；录像 = 整幅 ROI 原分辨率原始 Bayer，
     每 video_every 个触发格存一帧（默认 100/20 = 5 fps，用户 1011：「保存视频就按 5 FPS 保留原版的，不要降低分辨率」；
     按格号挑帧，4 台存的是同一批触发；hs_<serial>.*，与 HighSpeedRecorder / test_src/hs_recording.py 同格式）。
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
#: 待准备帧队列上限（帧）：准备线程跟不上时丢最老的，计数
PROCESS_QUEUE_FRAMES = 200
#: 识别任务队列上限（2 s；本相机几个识别进程共用）：满了就丢这一帧的任务、计数，不阻塞取流
DETECT_QUEUE_JOBS = 200
#: 每台相机几个识别进程（共用上面的任务队列）：1011 00:57 挥拍时同进程识别每帧平均 8.4–10 ms，一个进程贴着 100 fps 的
#: 10 ms 预算。写成常量不进配置：配置会被复制去改（挥拍专场的临时配置），新增键旧副本里没有
DETECT_WORKERS = 2
#: 录像写盘队列上限（帧；5 fps 时 60 s）
VIDEO_QUEUE_FRAMES = 300
#: 开录前设备时钟锁存次数（取往返最短的作初始偏移），之后每秒一次
CLOCK_LATCH_INITIAL = 7
CLOCK_LATCH_PERIOD_S = 1.0
#: 自动增益（同 src/ball_grabber.SyncCapture.auto_gain）
AUTO_GAIN_TOLERANCE_DB = 1.0
AUTO_GAIN_MAX_ROUNDS = 8
#: 开场全幅样张取几帧平均再认 tag。夜里 0.8 ms 曝光增益顶满（36.8 dB），单帧噪声大：1011 00:42 ASUS 车在
#: (0.52, 2.46)、tag 边长 ~45 px，单帧 tag1 四台都认不出（连四边形都提不出），中值 / 双边滤波后才解得出 1 号。
#: 车静止，平均 8 帧噪声降到 ~1/2.8
SNAPSHOT_AVERAGE_FRAMES = 8


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


def crop_marker_patch(
    raw: np.ndarray,
    roi_origin: tuple[int, int],
    anchor_uv: np.ndarray,
    *,
    radius: int,
) -> Optional[tuple[int, int, np.ndarray]]:
    """原始 Bayer ROI 帧上，锚点（全幅像素）附近截块：左上角对齐到偶数（ROI 偏移本身是偶数），Bayer 相位与
    全幅一致；块比候选半径多留 PATCH_BORDER_PX 的解码边。返回 (块左上全幅 x, y, 原始块)；锚点出 ROI 返回 None。"""
    ox, oy = roi_origin
    height, width = raw.shape[:2]
    au, av = float(anchor_uv[0]) - ox, float(anchor_uv[1]) - oy
    if not (math.isfinite(au) and math.isfinite(av)):
        return None
    if au < -radius or au >= width + radius or av < -radius or av >= height + radius:
        return None
    half = radius + PATCH_BORDER_PX
    x0 = max(0, (int(math.floor(au - half)) // 2) * 2)
    y0 = max(0, (int(math.floor(av - half)) // 2) * 2)
    x1 = min(width, x0 + 2 * half + 2)
    y1 = min(height, y0 + 2 * half + 2)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return ox + x0, oy + y0, np.ascontiguousarray(raw[y0:y1, x0:x1])


def marker_candidates_in_patch(
    detector,
    patch: np.ndarray,
    patch_origin: tuple[int, int],
    anchor_uv: tuple[float, float],
    code: int,
) -> list[tuple]:
    """小块解码 → 黑标 YOLO 候选（detector = marker_yolo.MarkerYolo）。返回候选行 [(u, v, score, area, bx, by, bw, bh)]，
    全幅像素，score = 置信。"""
    import cv2

    bgr = cv2.cvtColor(np.ascontiguousarray(patch), code)
    return detector.candidates(bgr, patch_origin, anchor_uv)


def detect_marker_in_roi(
    detector,
    raw: np.ndarray,
    roi_origin: tuple[int, int],
    anchor_uv: np.ndarray,
    code: int,
    *,
    radius: int,
) -> tuple[list[tuple], Optional[tuple[int, int, np.ndarray]]]:
    """截块 + 认候选一步做完（取流 / 识别两进程各做一半，这里给单测和离线用）。"""
    patch = crop_marker_patch(raw, roi_origin, anchor_uv, radius=radius)
    if patch is None:
        return [], None
    gx, gy, block = patch
    return marker_candidates_in_patch(detector, block, (gx, gy), (float(anchor_uv[0]), float(anchor_uv[1])), code), patch


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


class TagWindowAverager:
    """tag 窗按触发格分块累加：格号 // every 相同的帧算一块（各相机格号同源 ⇒ 块边界一致，主进程能按块凑齐），
    块内逐帧把 tag 窗的原始 Bayer 加起来，换块时出平均图再认 tag——车静止，平均 every 帧噪声降到 1/√every
    （夜里单帧认不出，见 SNAPSHOT_AVERAGE_FRAMES）。窗位置在每块第一帧按当时的车位姿定、块内不变。
    窗左上角是偶数（tag_windows 保证），平均图的 Bayer 相位不变。"""

    def __init__(self, every: int) -> None:
        self.every = max(1, int(every))
        self._block: Optional[int] = None
        self._windows: list[tuple[int, int, int, int]] = []
        self._sums: list[np.ndarray] = []
        self._times: list[float] = []

    def add(self, slot: int, frame_t: float, raw: np.ndarray, windows_fn) -> Optional[tuple]:
        """加一帧；跨进新块时返回上一块的 (块首格号, 平均帧时间, [(x0, y0, 平均原始图), …])，否则 None。
        windows_fn() 给当前车位姿下的 tag 窗（ROI 像素 (x0, y0, x1, y1)），只在每块第一帧调用。"""
        block = slot // self.every
        done = None
        if self._block is not None and block != self._block:
            done = self.flush()
        if self._block is None:
            self._block = block
            self._windows = list(windows_fn())
            self._sums = [np.zeros((y1 - y0, x1 - x0), np.float32) for x0, y0, x1, y1 in self._windows]
            self._times = []
        for acc, (x0, y0, x1, y1) in zip(self._sums, self._windows):
            acc += raw[y0:y1, x0:x1]
        self._times.append(frame_t)
        return done

    def flush(self) -> Optional[tuple]:
        if self._block is None or not self._times:
            self._block = None
            return None
        n = float(len(self._times))
        out = (
            self._block * self.every,
            float(np.mean(self._times)),
            [(x0, y0, np.clip(np.rint(acc / n), 0, 255).astype(np.uint8))
             for acc, (x0, y0, _, _) in zip(self._sums, self._windows)],
        )
        self._block = None
        return out


def make_tag_detector(*, full_frame: bool):
    """AprilTag 36h11 检测器，参数同 src/car_localizer。全幅（开场样张）连 maxMarkerPerimeterRate=0.6 也一样；tag 窗不设它：
    那是相对图像长边的比例，整幅 2048 px 时上限 ~1229 px，~200 px 的窗里 50 px 的 tag 周长率 ~1，会被当成太大扔掉（1010 假相机实测）。"""
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.perspectiveRemovePixelPerCell = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.3
    if full_frame:
        params.maxMarkerPerimeterRate = 0.6
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)


def detect_tags(detector, gray: np.ndarray, origin: tuple[int, int], tag_ids: set[int]) -> list[tuple[int, list]]:
    """认 tag：原图 → 3×3 中值去噪 → 拉局部对比度（CarLocalizer 的暗光增强，同一个函数）依次试，前面认到的不被后面覆盖，
    配置里的 tag 都齐了就停；同一张图里同一 id 出现两次的那个 id 作废。角点换回全幅像素（加 origin）。
    中值那一档是 1011 00:42 ASUS 夜场实测加的：增益顶满、单帧噪声大时 tag1 原图和增强图都认不出，中值 / 双边滤波后才解得出。"""
    import cv2
    from collections import Counter

    from src.car_localizer import CarLocalizer

    found: dict[int, np.ndarray] = {}
    for prepare in (lambda g: g, lambda g: cv2.medianBlur(g, 3), CarLocalizer._enhance_for_low_light):
        corners, ids, _ = detector.detectMarkers(prepare(gray))
        if ids is not None:
            got = [(int(i), np.asarray(c, dtype=np.float64).reshape(4, 2)) for i, c in zip(ids.ravel(), corners)
                   if int(i) in tag_ids]
            counts = Counter(tid for tid, _ in got)
            for tid, c in got:
                if counts[tid] == 1 and tid not in found:
                    found[tid] = c
        if set(found) >= tag_ids:
            break
    gx, gy = origin
    return [(tid, [[float(u) + gx, float(v) + gy] for u, v in c]) for tid, c in sorted(found.items())]


class CameraWorker:
    """一台相机：开 → 自动增益 → 全幅样张 → ROI + action 触发 → 取帧 / 识别 / 写盘 → 收尾。"""

    def __init__(self, spec: dict, out_q, ctrl_q, detect_q, shared_car, t0, stop_evt, start_evt) -> None:
        import src.mvs_env  # noqa: F401  （MvImport / DLL 路径，必须在 SDK 之前）
        import CameraParams_header as cph
        from MvCameraControl_class import MvCamera

        self.cph = cph
        self.MvCamera = MvCamera
        self.spec = spec
        self.serial = str(spec["serial"])
        self.out_q = out_q
        self.ctrl_q = ctrl_q
        self.detect_q = detect_q
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
        self.full_tag_detector = make_tag_detector(full_frame=True)
        self.joints = JointBuffer(eab.CAR_MODELS[self.car].slot_joint_names)
        self.radius = int(spec["crop_radius_px"])
        self.tag_avg = TagWindowAverager(int(spec["tag_every_frames"]))
        self.tag_pad = float(spec["tag_window_pad_px"])
        self.tags_in_roi = True
        # 取帧 → 识别
        self._frames: deque = deque()
        self._frames_cv = threading.Condition()
        self._grab_done = threading.Event()
        self.video_every = int(spec["video_every_slots"])      # 录像每几个触发格存一帧，0 = 不录
        self.detect_workers = int(spec["detect_workers"])
        self._seq = 0                                          # 交给识别进程的任务序号（主进程按它恢复本相机的先后）
        self._video_q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=VIDEO_QUEUE_FRAMES)
        self._patch_q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=4000)
        self.stats: dict[str, Any] = {
            "received": 0, "missing": 0, "lost_packets": 0, "lost_packet_frames": 0, "size_mismatch": 0,
            "proc_dropped": 0, "prepared": 0, "detect_dropped": 0, "no_car_pose": 0, "no_joint_state": 0,
            "out_of_roi": 0, "tag_blocks": 0, "video_frames": 0, "video_dropped": 0,
            "patches_written": 0, "patches_dropped": 0, "prep_ms_sum": 0.0, "prep_ms_max": 0.0,
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
            frames = []
            for _ in range(SNAPSHOT_AVERAGE_FRAMES):
                frame, pixel_type = self._grab_soft()
                frames.append(frame)
        finally:
            self.cam.MV_CC_StopGrabbing()
        median = float(np.median(frames[-1][::4, ::4]))
        err_db = 20.0 * math.log10(target / max(median, 1.0))
        # 平均图（原始 Bayer 逐像素平均，线性运算、相位不变）认 tag、存样张
        raw = np.clip(np.rint(np.mean(np.stack(frames).astype(np.float32), axis=0)), 0, 255).astype(np.uint8)
        cv2.imwrite(str(self.out_dir / f"full_{self.serial}.png"), raw)
        self.pixel_type = pixel_type
        bgr = cv2.cvtColor(raw, bayer_code(pixel_type))
        self.snapshot_bgr = bgr
        tags = detect_tags(self.full_tag_detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), self.tag_ids)
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
        video_every = self.video_every
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
                t0 = self.t0.value
                if t0 == 0.0:          # 触发进程还没起：不该有帧（触发模式），有也不知道是哪一格
                    stats["before_trigger"] += 1
                    continue
                slot = slot_of(frame_t, t0, self.period)
                if video_every and slot % video_every == 0:
                    try:
                        # 索引同 HighSpeedRecorder 格式（字段就叫 exposure_start_pc），test_src/hs_recording.py 直接读
                        self._video_q.put_nowait(((fnum, dev, exposure_start_pc, arrival, lost), data))
                    except queue.Full:
                        stats["video_dropped"] += 1
                h, w = self.frame_shape
                raw = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
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

    def _submit(self, kind: str, payload: tuple) -> bool:
        """交给识别进程，带本相机的任务序号（交出成功才占号，序号连续）。"""
        try:
            self.detect_q.put_nowait((kind, self._seq, payload))
        except queue.Full:
            self.stats["detect_dropped"] += 1
            return False
        self._seq += 1
        return True

    def _prepare_loop(self) -> None:
        """每帧：等关节角 → FK 黑标锚 → 截块，连同帧信息交给识别进程；tag 窗按块累加平均，满一块交给识别进程。"""
        stats = self.stats
        pixel_type = self.pixel_type if self.pixel_type is not None else 0x01080008
        roi_origin = (self.roi[0], self.roi[1])
        record_patches = bool(self.spec.get("record_patches"))
        while True:
            item = self._next_frame()
            if item is None:
                break
            slot, fnum, frame_t, raw = item
            if self.pixel_type is not None:
                pixel_type = self.pixel_type
            stats["joint_wait_ms_sum"] += 1e3 * self._wait_joints(frame_t)
            t_start = time.perf_counter()
            pose = self.car_pose()
            q = self.joints.at_pc(frame_t)
            anchor_mm = None
            patch_job = None
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
                patch = crop_marker_patch(raw, roi_origin, anchor_uv, radius=self.radius)
                if patch is None:
                    status = "out_of_roi"
                    stats["out_of_roi"] += 1
                else:
                    status = "ok"
                    gx, gy, block = patch
                    patch_job = (gx, gy, float(anchor_uv[0]), float(anchor_uv[1]), block, pixel_type)
                    if record_patches:
                        try:
                            self._patch_q.put_nowait((slot, fnum, frame_t, patch))
                        except queue.Full:
                            stats["patches_dropped"] += 1
            self._submit("m", (slot, fnum, frame_t, status, anchor_mm,
                               None if q is None else [round(float(v), 6) for v in q], patch_job))
            if pose is not None and self.tags_in_roi:
                block = self.tag_avg.add(
                    slot, frame_t, raw,
                    lambda: tag_windows(self.camera, pose, self.tag_corners, self.roi, self.tag_pad))
                if block is not None:
                    stats["tag_blocks"] += 1
                    self._submit("t", (*block, roi_origin, pixel_type))
            stats["prepared"] += 1
            dt_ms = 1e3 * (time.perf_counter() - t_start)
            stats["prep_ms_sum"] += dt_ms
            stats["prep_ms_max"] = max(stats["prep_ms_max"], dt_ms)
        block = self.tag_avg.flush()
        if block is not None:
            stats["tag_blocks"] += 1
            self._submit("t", (*block, roi_origin, pixel_type))

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

    def _video_writer(self) -> None:
        base = self.out_dir / f"hs_{self.serial}"
        pack = struct.Struct(INDEX_FORMAT).pack
        with open(base.with_suffix(".bayer"), "wb", buffering=4 << 20) as fr, \
                open(base.with_suffix(".idx"), "wb", buffering=1 << 16) as fi:
            while True:
                item = self._video_q.get()
                if item is None:
                    break
                rec, data = item
                fr.write(data)
                fi.write(pack(*rec))
                self.stats["video_frames"] += 1
                if self.stats["video_frames"] % 10 == 0:      # 5 fps 时每 2 s 落一次盘，进程崩了也只丢 2 s
                    fr.flush()
                    fi.flush()

    def _write_video_meta(self) -> None:
        if not self.video_every:
            return
        ox, oy, w, h = self.roi
        fps = float(self.spec["fps"])
        write_json(self.out_dir / f"hs_{self.serial}.json", {
            "serial": self.serial,
            "roi": {"offset_x": ox, "offset_y": oy, "width": w, "height": h},
            # slot_fps = 录像自己的帧栅格（hs_recording.py 的 summary 按它数跳格）：每 video_every_slots 个触发格一帧
            "slot_fps": fps / self.video_every,
            "trigger_fps": fps,
            "video_every_slots": self.video_every,
            "readback": self.readback,
            "files": {"index_format": INDEX_FORMAT,
                      "index_fields": ["frame_num", "dev_timestamp", "exposure_start_pc", "arrival_perf", "lost_packet"],
                      "frame_time": "曝光中点 = exposure_start_pc + readback.exposure_us / 2；同一帧在 marker.jsonl 里按 frame_num 找"},
            "recording": {"frame_shape": list(self.frame_shape) if self.frame_shape else None,
                          "pixel_type": self.pixel_type, "written": self.stats["video_frames"],
                          "dropped": self.stats["video_dropped"]},
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
                threading.Thread(target=self._prepare_loop, name=f"prep-{self.serial}", daemon=True),
            ]
            if self.spec.get("record_patches"):
                threads.append(threading.Thread(target=self._patch_writer, name=f"patch-{self.serial}", daemon=True))
            if self.video_every:
                threads.append(threading.Thread(target=self._video_writer, name=f"video-{self.serial}", daemon=True))
            clock_stop = threading.Event()
            threads.append(threading.Thread(target=self._clock_loop, args=(clock_stop,), name=f"clock-{self.serial}",
                                            daemon=True))
            for th in threads:
                th.start()
            while not self.stop_evt.wait(1.0):
                self._send("stats", self.stats_snapshot())
            # 收尾：取帧线程把 SDK 缓冲取空后退出 → 准备线程把剩下的帧交给识别进程 → 写盘线程收尾 → 通知识别进程
            threads[0].join(timeout=10.0)
            self.cam.MV_CC_StopGrabbing()
            threads[1].join(timeout=30.0)
            clock_stop.set()
            self._patch_q.put(None)
            self._video_q.put(None)
            for th in threads[2:]:
                th.join(timeout=60.0)
            listener_stop.set()
            for _ in range(self.detect_workers):
                try:
                    # 每个识别进程一个结束标记，排在所有任务后面：剩下的任务做完就退（主进程兜底也会放）
                    self.detect_q.put(None, timeout=30.0)
                except queue.Full:
                    break
            for _ in range(3):
                self.latch_clock()
            self._write_clock_csv()
            self._write_video_meta()
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


def camera_main(spec: dict, out_q, ctrl_q, detect_q, shared_car, t0, stop_evt, start_evt) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # 只认主进程的停止事件
    set_priority(ABOVE_NORMAL_PRIORITY_CLASS)
    try:
        with HighResolutionTimer():
            CameraWorker(spec, out_q, ctrl_q, detect_q, shared_car, t0, stop_evt, start_evt).run()
    except BaseException as exc:  # noqa: BLE001 —— 任何异常都要报回主进程
        out_q.put(("error", str(spec.get("serial")), f"{exc!r}\n{traceback.format_exc()}"))


def detect_main(spec: dict, worker: int, detect_q, out_q) -> None:
    """一台相机的一个黑标识别进程（本相机几个识别进程共用 detect_q）：取流进程交来的小块 → 黑标 YOLO 候选（"f" 消息）、
    tag 平均图 → 认 tag（"t" 消息），带任务序号发主进程。收到 None（取流进程收尾时每个识别进程放一个；主进程兜底也会放）就退。"""
    import cv2

    from swing_hs.marker_yolo import MarkerYolo

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    serial = str(spec["serial"])
    try:
        cv2.setNumThreads(1)       # 小块上 OpenCV 内部多线程只有调度开销，8 个识别进程还会互相抢核
        detector = MarkerYolo(threads=1)      # 同理 ONNX Runtime 也单线程；8 个进程各一个会话、不碰 GPU
        tag_detector = make_tag_detector(full_frame=False)
        tag_ids = set(geometry.tag_corners_car(geometry.load_vehicle_reference(Path(spec["vehicle_path"])), 1.0))
        stats: dict[str, Any] = {"worker": int(worker), "processed": 0, "with_candidates": 0, "proc_ms_sum": 0.0,
                                 "proc_ms_max": 0.0, "tag_runs": 0, "tag_hits": 0, "backlog": 0, "backlog_max": 0}
        last_report = time.perf_counter()

        def report(kind: str) -> None:
            try:
                stats["backlog"] = detect_q.qsize()
            except NotImplementedError:
                stats["backlog"] = -1
            stats["backlog_max"] = max(stats["backlog_max"], stats["backlog"])
            out_q.put((kind, serial, dict(stats)))

        while True:
            if time.perf_counter() - last_report >= 1.0:
                last_report = time.perf_counter()
                report("dstats")
            try:
                job = detect_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            t_start = time.perf_counter()
            kind, seq, body = job
            if kind == "m":
                slot, fnum, frame_t, status, anchor_mm, q, patch_job = body
                rows: list[tuple] = []
                if patch_job is not None:
                    gx, gy, au, av, block, pixel_type = patch_job
                    rows = marker_candidates_in_patch(detector, block, (gx, gy), (au, av), bayer_code(pixel_type))
                    if rows:
                        stats["with_candidates"] += 1
                out_q.put(("f", serial, (seq, (slot, fnum, frame_t, status, anchor_mm, q, rows))))
                stats["processed"] += 1
                dt_ms = 1e3 * (time.perf_counter() - t_start)
                stats["proc_ms_sum"] += dt_ms
                stats["proc_ms_max"] = max(stats["proc_ms_max"], dt_ms)
            else:
                slot0, t_mean, windows, (ox, oy), pixel_type = body
                code = bayer_code(pixel_type)
                found: list[tuple[int, list]] = []
                for x0, y0, avg in windows:
                    gray = cv2.cvtColor(cv2.cvtColor(avg, code), cv2.COLOR_BGR2GRAY)
                    found.extend(detect_tags(tag_detector, gray, (ox + x0, oy + y0), tag_ids))
                if len({tid for tid, _ in found}) != len(found):
                    found = []
                stats["tag_runs"] += 1
                stats["tag_hits"] += bool(found)
                out_q.put(("t", serial, (seq, (slot0, t_mean, found))))
        report("ddone")
    except BaseException as exc:  # noqa: BLE001
        out_q.put(("error", f"detect-{serial}-{worker}", f"{exc!r}\n{traceback.format_exc()}"))
