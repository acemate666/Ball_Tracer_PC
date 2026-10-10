# -*- coding: utf-8 -*-
"""swing_hs 触发进程：HIGH 优先级，按绝对栅格 t0 + n/fps 广播 GigE action command，每次记发出前后时刻。

同 ArmCalibration/record_swing_hs.py 的触发环（1006 ASUS 自检：100 fps 60 s 0 跳格、迟到 max 0.1 ms）：
  - 触发环不能和取帧 / 识别同进程（抢 GIL 就迟到跳格，tennis-man memory/pc-cam-120fps-marker-roi-1006.md）；
  - 迟到只丢这一格、不平移后面所有格（0906 的 32 fps 教训）；
  - 跳格按发出前时刻判（action 调用偶尔 0.7–1.2 ms 才返回，按返回时刻判会白丢一格）。
t0 写进共享值：相机进程按 round((曝光时刻 − t0) × fps) 给每帧编格号，主进程按格号把 4 台凑成一组。
"""

from __future__ import annotations

import math
import signal
import struct
import time
import traceback

from swing_hs.winutil import HIGH_PRIORITY_CLASS, HighResolutionTimer, set_priority

#: 每次触发：slot, 发出前 perf, 发出后 perf
TRIGGER_FORMAT = "<qdd"
#: 睡到截止前这么多再自旋（Windows 即使 1 ms 定时器也会晚醒 1 ms 以上）
SPIN_GUARD_S = 0.0015


def next_trigger_slot(fired_slot: int, fired_elapsed: float, *, period: float, minimum_period: float) -> int:
    """绝对栅格上的下一格：正常是 fired_slot+1；本次发晚到下一格会落进相机最小周期内时，整格跳过。"""
    slot = fired_slot + 1
    earliest = fired_elapsed + minimum_period
    if slot * period < earliest:
        slot = int(math.ceil(earliest / period))
    return slot


def trigger_main(spec: dict, out_path: str, status_q, stop_evt, t0_shared) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    set_priority(HIGH_PRIORITY_CLASS, time_critical_thread=True)
    try:
        import src.mvs_env  # noqa: F401
        import CameraParams_header as cph
        from MvCameraControl_class import MvCamera

        MvCamera.MV_CC_Initialize()
        period = 1.0 / float(spec["fps"])
        minimum_period = float(spec["minimum_period_s"])
        info = cph.MV_ACTION_CMD_INFO()
        results = cph.MV_ACTION_CMD_RESULT_LIST()
        info.nDeviceKey = int(spec["action_device_key"])
        info.nGroupKey = int(spec["action_group_key"])
        info.nGroupMask = int(spec["action_group_mask"])
        info.bActionTimeEnable = 0
        info.nActionTime = 0
        bcast = str(spec["action_broadcast_address"]).encode("ascii")   # 存住，别让 ctypes 指针悬空
        info.pBroadcastAddress = bcast
        info.nTimeOut = int(spec["action_timeout_ms"])
        info.bSpecialNetEnable = 0
        info.nSpecialNetIP = 0
        pack = struct.Struct(TRIGGER_FORMAT).pack
        issued = skipped = errors = 0
        late_over = {"0.2ms": 0, "0.5ms": 0, "1ms": 0, "2ms": 0}
        max_late = 0.0
        last_report = time.perf_counter()
        with open(out_path, "wb", buffering=1 << 16) as f, HighResolutionTimer():
            t0 = time.perf_counter() + 0.05
            t0_shared.value = t0
            slot = 0
            while not stop_evt.is_set():
                deadline = t0 + slot * period
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    if remaining > SPIN_GUARD_S:
                        time.sleep(min(remaining - SPIN_GUARD_S, 0.002))
                tb = time.perf_counter()
                ret = MvCamera.MV_GIGE_IssueActionCommand(info, results)
                ta = time.perf_counter()
                if int(ret) != 0:
                    errors += 1
                f.write(pack(slot, tb, ta))
                issued += 1
                late = tb - deadline
                max_late = max(max_late, late)
                for key, limit in (("0.2ms", 2e-4), ("0.5ms", 5e-4), ("1ms", 1e-3), ("2ms", 2e-3)):
                    if late > limit:
                        late_over[key] += 1
                nxt = next_trigger_slot(slot, tb - t0, period=period, minimum_period=minimum_period)
                skipped += nxt - slot - 1
                slot = nxt
                if ta - last_report >= 1.0:
                    last_report = ta
                    f.flush()
                    status_q.put(("trigger", "trigger", {
                        "issued": issued, "skipped_slots": skipped, "errors": errors,
                        "max_late_ms": round(max_late * 1e3, 3), "late_over": dict(late_over)}))
        MvCamera.MV_CC_Finalize()
        status_q.put(("done", "trigger", {
            "issued": issued, "skipped_slots": skipped, "errors": errors, "max_late_ms": round(max_late * 1e3, 3),
            "late_over": late_over, "period_s": period, "minimum_period_s": minimum_period, "t0": t0}))
    except BaseException as exc:  # noqa: BLE001
        status_q.put(("error", "trigger", f"{exc!r}\n{traceback.format_exc()}"))
