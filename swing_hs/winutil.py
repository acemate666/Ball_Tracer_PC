# -*- coding: utf-8 -*-
"""Windows 小工具：进程/线程优先级、1 ms 定时器精度、按 PID 判进程是否还活着、JSON 原子写。"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
HIGH_PRIORITY_CLASS = 0x80
THREAD_PRIORITY_TIME_CRITICAL = 15
_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x102


def set_priority(priority_class: int, *, time_critical_thread: bool = False) -> None:
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), priority_class)
        if time_critical_thread:
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            k32.SetThreadPriority(k32.GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)
    except Exception:
        pass


class HighResolutionTimer:
    """进程存活期间向 Windows 要 1 ms 调度精度（sleep 才睡得准）。"""

    def __enter__(self):
        self._winmm = getattr(getattr(ctypes, "windll", None), "winmm", None)
        if self._winmm is not None:
            try:
                self._winmm.timeBeginPeriod(1)
            except Exception:
                self._winmm = None
        return self

    def __exit__(self, *exc):
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass


def pid_alive(pid: int) -> bool:
    """⚠ Windows 上不能用 os.kill(pid, 0) 探活——它会直接 TerminateProcess。"""
    try:
        k32 = ctypes.windll.kernel32
    except AttributeError:
        return True
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(_SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    try:
        return k32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == _WAIT_TIMEOUT
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def write_json(path: Path, doc: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
