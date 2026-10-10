# -*- coding: utf-8 -*-
"""/joint_states 在 swing_hs 里的流转：ROS 桥（ros_bridge.py）收到一条就按 UDP 发给每个相机进程和主进程，
各自存一份 JointBuffer，按曝光时刻（PC 轴）插值取关节角。

RK→PC 换算与 tracker 在线黑标同一做法（src/racket_marker_online.JointStateBuffer）：这条流自己的
(PC 收到时刻 − header.stamp) 取 10% 分位 ≈ 两轴偏移 + 最小网络延时；收到时刻是桥进程里的
perf_counter（QueryPerformanceCounter 全机共用，各进程同一根 PC 轴）。
"""

from __future__ import annotations

import bisect
import socket
import threading
import time
from collections import deque

import numpy as np

from swing_hs.udp_msgs import decode_joint_state

# 与 racket_marker_online 同值
JOINT_BUFFER_S = 2.0
JOINT_MAX_GAP_S = 0.05
JOINT_MAX_HOLD_S = 0.01
JOINT_OFFSET_PERCENTILE = 10.0
#: 偏移估计每收这么多条重算一次
_OFFSET_REFRESH = 20


class JointBuffer:
    """关节角缓存（控制器 6 槽，五轴臂槽 4 恒 0）。一个线程写、一个线程读，内部加锁。"""

    def __init__(self, slot_names: tuple) -> None:
        self._slot_names = tuple(slot_names)
        self._expected = sorted(n for n in self._slot_names if n is not None)
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque()
        self._offsets: deque[float] = deque()
        self._positions: deque[np.ndarray] = deque()
        self._offset: float | None = None
        self._since_refresh = 0
        self._last_recv_pc: float | None = None
        self.received = 0
        self.name_mismatch: list[str] | None = None

    def add(self, stamp_rk: float, names: list[str], positions: list[float], recv_pc: float) -> None:
        if sorted(names) != self._expected:
            self.name_mismatch = list(names)
            return
        by_name = dict(zip(names, positions))
        q = np.asarray([0.0 if n is None else by_name[n] for n in self._slot_names], dtype=np.float64)
        with self._lock:
            self.received += 1
            self._last_recv_pc = recv_pc
            if self._stamps and stamp_rk <= self._stamps[-1]:
                return
            self._stamps.append(stamp_rk)
            self._offsets.append(recv_pc - stamp_rk)
            self._positions.append(q)
            while stamp_rk - self._stamps[0] > JOINT_BUFFER_S:
                self._stamps.popleft()
                self._offsets.popleft()
                self._positions.popleft()
            self._since_refresh += 1
            if self._offset is None or self._since_refresh >= _OFFSET_REFRESH:
                self._offset = float(np.percentile(self._offsets, JOINT_OFFSET_PERCENTILE))
                self._since_refresh = 0

    def offset(self) -> float | None:
        """PC − RK（s）。"""
        with self._lock:
            return self._offset

    def latest_pc(self) -> float | None:
        """最新一条关节角对应的 PC 时刻。"""
        with self._lock:
            if not self._stamps or self._offset is None:
                return None
            return self._stamps[-1] + self._offset

    def alive(self, now_pc: float, max_silence_s: float = 0.3) -> bool:
        with self._lock:
            return self._last_recv_pc is not None and now_pc - self._last_recv_pc <= max_silence_s

    def at_pc(self, pc_t: float) -> np.ndarray | None:
        """PC 轴 pc_t 时刻的关节角（前后两条线性插值）；缓存里没有这一刻返回 None。"""
        with self._lock:
            if len(self._stamps) < 20 or self._offset is None:
                return None
            rk_t = pc_t - self._offset
            stamps = self._stamps
            right = bisect.bisect_left(stamps, rk_t)
            if right == len(stamps):
                return self._positions[-1].copy() if rk_t - stamps[-1] <= JOINT_MAX_HOLD_S else None
            if right == 0 or stamps[right] - stamps[right - 1] > JOINT_MAX_GAP_S:
                return None
            fraction = (rk_t - stamps[right - 1]) / (stamps[right] - stamps[right - 1])
            a, b = self._positions[right - 1], self._positions[right]
            return a + fraction * (b - a)


def start_joint_listener(port: int, buffer: JointBuffer, stop_event, *, name: str = "joints") -> threading.Thread:
    """127.0.0.1:port 收 ROS 桥转发的 /joint_states。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind(("127.0.0.1", int(port)))
    sock.settimeout(0.2)

    def loop() -> None:
        try:
            while not stop_event.is_set():
                try:
                    data = sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    if stop_event.is_set():
                        break
                    time.sleep(0.01)
                    continue
                try:
                    stamp, recv_pc, names, positions = decode_joint_state(data)
                except (ValueError, KeyError, TypeError):
                    continue
                buffer.add(stamp, names, positions, recv_pc)
        finally:
            sock.close()

    thread = threading.Thread(target=loop, name=name, daemon=True)
    thread.start()
    return thread
