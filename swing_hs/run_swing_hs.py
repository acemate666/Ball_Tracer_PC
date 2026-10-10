#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""100 fps 静态挥拍识别（swing_hs）：车不动、臂挥拍，4 台 PC 相机只开 ROI、100 fps 同步触发，
在线认拍上黑标 + 车位，全程录 rosbag。

用户 2026-10-10：「做一个 100fps 的版本，只用来做车静态的挥拍识别，因此可以大幅裁减 roi，甚至正手和反手
挥拍的裁剪区域也可以不一样……识别内容只需要是球拍黑标位置和车的 car_loc，然后和 tracker 一样开个 ros bag
记录所有信息」。tracker（src/run_tracker.py）40 fps 全幅、看球；本工具不看球、不出报告。

一次运行：
  1. 4 台相机各起一个进程，全幅软触发：自动增益（灰度中位数 60，同 tracker）→ 全幅样张认 tag →
     主进程用双 tag 解车位（车 yaw 必须双 tag）；
  2. ROI 规划：--side 那一侧的黑标包络（swing_hs/envelope_<car>.json 里实打挥拍的关节采样，按 J1 范围筛、
     FK 出黑标）+ 两块 tag，按车位投到各相机，外扩、对齐；超 100 fps 预算、双 tag 不够 2 台、看得到黑标的
     不够 2 台就不开录。各相机目录里 roi_<serial>.jpg 画着 ROI、包络（绿）和 tag（紫）——--plan-only 到这里就停；
  3. ROS 桥（/joint_states 进、本工具话题出）+ rosbag（录全部话题，同 tracker）起来；
  4. 触发进程按 100 Hz 绝对栅格发 action command。每台相机每帧：帧时间（曝光中点）的关节角 + 车位姿 → FK 黑标锚 →
     全分辨率截块找黑方块候选（每台相机 1 个取流进程 + 2 个识别进程，识别不拖取流）；tag 窗每 10 帧一块平均后认 tag
     （开场全幅样张也是 8 帧平均：夜里单帧太噪）；整幅 ROI 原始帧每 20 格存一帧（5 fps 录像）；
  5. 主进程按触发格把 4 台凑齐：多视图解黑标（marker_solver）、双 tag 刚体拟合解车位（CarLocalizer），
     写 marker.jsonl / car.jsonl，发 /swing_hs/racket_marker、/pc_car_loc、/swing_hs/status（都进 bag）；
  6. Ctrl+C / 停止文件 / 到时长 → 停触发 → 各相机把剩下的帧认完 → 停 ROS → meta.json（含自检汇总）。

/pc_car_loc 照 tracker 主线契约发（载荷、BEST_EFFORT depth 1 都一样，不带 rk_timestamp）：挥拍专场车会被推着动，
车位要一直有；挥拍时不起 bot_center（用户 1010），不用管它收 /pc_car_loc 的对时门。rosbag 不连球拍 IMU（用户 1010：
现在不开，bag 不能因为它停）。时间全在 PC 轴（perf_counter）：所有 t 都是帧时间 = 曝光开始 + 读回曝光 / 2（曝光中点），
相机进程取到帧时算一次，下游不再补半曝光（同 tracker ee34909，用户 1010 定）；/joint_states 的 RK 时刻按这条流自己的
收发差换算（同 tracker 在线黑标），黑标消息里的 rk_t 只是这个估计（含最小网络延时），精确对时离线做。

产物 tracker_output/swing_hs_<YYYYmmdd_HHMMSS>/：
  meta.json          参数、车位、ROI 规划、各相机读回、统计、收尾汇总
  marker.jsonl       每个触发格一行：t、关节角、车位、FK 锚、各相机候选（全幅像素）、在线解
  car.jsonl          每次车位解一行（字段同 tracker car_locs）
  trigger.bin        每次触发 <qdd>：slot, 发出前 perf, 发出后 perf
  <session>_rosbag/  全部话题（含本工具发的三个）；ros_bridge.log；cyclonedds_swing_hs.xml（补了本机 peer 的 DDS 配置）
  cam_<serial>/      full_<serial>.png（全幅原始 Bayer 样张）、roi_<serial>.jpg、clock_<serial>.csv、
                     patches_<serial>.bin（每帧锚点小块原始 Bayer，--no-patches 关）、
                     hs_<serial>.{bayer,idx,json}：录像，整幅 ROI 原分辨率原始 Bayer，触发格号 % 20 == 0 的帧（5 fps，
                     4 台同一批触发），4 台共 ~11 MB/s ≈ 40 GB/h；test_src/hs_recording.py 读 / 导 mp4（--play-fps 5 是原速）。
                     用户 1011：「保存视频就按 5 FPS 保留原版的（不要降低分辨率）」——1011 01:01 那场 100 fps 全录，写盘跟不上丢了
                     2–12% 的帧。--video-fps 改帧率（要整除触发帧率），0 不录

用法（ASUS，仓库根，.venv_ros2 的 python；或 .\\run_swing_hs.ps1 -Car v05 -Side fore）：
  python swing_hs/run_swing_hs.py --car v05 --side fore
  python swing_hs/run_swing_hs.py --car v05 --side both --plan-only     # 只看 ROI 规划
  python swing_hs/run_swing_hs.py --car v05 --side fore --probe 20      # 相机 + 识别自检 20 s
停止：Ctrl+C，或建 tracker_output/.stop_swing_hs。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swing_hs import geometry, udp_msgs  # noqa: E402
from swing_hs.camera_proc import DETECT_QUEUE_JOBS, DETECT_WORKERS, camera_main, detect_main  # noqa: E402
from swing_hs.joints import JointBuffer  # noqa: E402
from swing_hs.marker_solver import MarkerFusion  # noqa: E402
from swing_hs.trigger_proc import trigger_main  # noqa: E402
from swing_hs.winutil import write_json  # noqa: E402

DEFAULT_CONFIG = Path(__file__).with_name("config_18.json")
STOP_FILE_NAME = ".stop_swing_hs"
#: 触发最小间隔在传感器最小周期之上留的余量（action 包在交换机上的抖动是几十 µs）
TRIGGER_MARGIN_S = 1.0e-4
#: 一个触发格最多等其它相机这么多格（3 s）就按不齐处理：各相机识别进程排队深浅不同（任务队列上限 2 s），
#: 结果到得有先有后
SLOT_TIMEOUT = 300
#: 车位相对开录时的漂移超过 ROI 余量就提醒（车没锁会被挥拍推）
DRIFT_WARN_PERIOD_S = 10.0
#: 录像帧率（整幅 ROI 原分辨率原始帧，按触发格号挑帧）：用户 1011「保存视频这一块就按 5FPS 保留原版的（不要降低分辨率）」。
#: --video-fps 覆盖。写成常量不进配置：配置会被复制去改（挥拍专场的临时配置），新增键旧副本里没有
VIDEO_FPS = 5.0


# ───────────────── 小件 ─────────────────


class SlotAssembler:
    """按触发格把各相机的消息凑齐：齐了、或比最新格落后 timeout 格，就交出去。"""

    def __init__(self, expected: set[str], timeout_slots: int) -> None:
        self.expected = set(expected)
        self.timeout = timeout_slots
        self.pending: dict[int, dict[str, Any]] = {}
        self.newest: Optional[int] = None
        self._closed: set[int] = set()      # 已交出的格（只留最近 2×timeout 格）
        self.late = 0                       # 格已交出后才到的消息

    def add(self, slot: int, serial: str, payload: Any) -> list[tuple[int, dict]]:
        if slot in self._closed:
            self.late += 1
            return []
        self.pending.setdefault(slot, {})[serial] = payload
        self.newest = slot if self.newest is None else max(self.newest, slot)
        done = []
        if self.expected <= set(self.pending[slot]):
            done.append((slot, self.pending.pop(slot)))
        done.extend(self.expire())
        self._closed.update(slot for slot, _ in done)
        if len(self._closed) > 4 * self.timeout:
            floor = self.newest - 2 * self.timeout
            self._closed = {slot for slot in self._closed if slot >= floor}
        return done

    def expire(self, force: bool = False) -> list[tuple[int, dict]]:
        if self.newest is None:
            return []
        out = []
        for slot in sorted(self.pending):
            if force or slot < self.newest - self.timeout:
                out.append((slot, self.pending.pop(slot)))
        self._closed.update(slot for slot, _ in out)
        return out


class SeqReorder:
    """一台相机的识别结果按交出顺序放行。一台相机几个识别进程共用一个任务队列，结果到主进程的先后会乱几个；按序号放行后
    每台相机的消息仍按格号先后到，凑格、黑标「连续」判断、marker.jsonl 和话题都保持时间顺序。
    序号只在任务交出成功后才占（camera_proc._submit），正常不缺号；真缺了攒到 max_hold 条就跳过去、计数。"""

    def __init__(self, max_hold: int) -> None:
        self.next = 0
        self.max_hold = max_hold
        self.held: dict[int, Any] = {}
        self.skipped = 0

    def push(self, seq: int, item: Any) -> list:
        if seq < self.next:          # 跳号之后才到的：直接放（凑格那边已交出的格会按迟到丢）
            return [item]
        self.held[seq] = item
        out = []
        while True:
            if self.next in self.held:
                out.append(self.held.pop(self.next))
                self.next += 1
            elif len(self.held) > self.max_hold:
                lowest = min(self.held)
                self.skipped += lowest - self.next
                self.next = lowest
            else:
                return out

    def flush(self) -> list:
        out = [self.held[seq] for seq in sorted(self.held)]
        self.held.clear()
        return out


def merge_detect_stats(per_worker: dict) -> dict:
    """一台相机几个识别进程的统计合成一份：计数相加，最大值取最大；积压是共用队列的长度，取最大。"""
    out: dict[str, Any] = {"workers": len(per_worker), "processed": 0, "with_candidates": 0, "proc_ms_sum": 0.0,
                           "proc_ms_max": 0.0, "tag_runs": 0, "tag_hits": 0, "backlog": 0, "backlog_max": 0}
    for s in per_worker.values():
        for key in ("processed", "with_candidates", "proc_ms_sum", "tag_runs", "tag_hits"):
            out[key] += s.get(key, 0)
        for key in ("proc_ms_max", "backlog", "backlog_max"):
            out[key] = max(out[key], s.get(key, 0))
    return out


def video_every_slots(fps: float, video_fps: float) -> int:
    """录像每几个触发格存一帧（0 = 不录）。必须整除：各相机按格号挑帧，4 台存的才是同一批触发，离线能多视图对上。"""
    if video_fps <= 0.0:
        return 0
    every = int(round(fps / video_fps))
    if every < 1 or abs(fps / every - video_fps) > 1e-6:
        raise ValueError(f"录像帧率 {video_fps:g} 要整除触发帧率 {fps:g}（如 5、10、20、25、50）")
    return every


class CarPose:
    """车位姿：最近 window 次双 tag 解的中位（车静止，滤掉单次噪声）；写进相机进程共享的数组。"""

    def __init__(self, shared, window: int) -> None:
        self.shared = shared
        self.history: deque[tuple[float, float, float]] = deque(maxlen=window)
        self.current: Optional[tuple[float, float, float]] = None
        self.plan: Optional[tuple[float, float, float]] = None
        self.accepted = 0
        self.missed = 0

    def set(self, pose: tuple[float, float, float], t: float) -> None:
        self.current = (float(pose[0]), float(pose[1]), float(pose[2]))
        with self.shared.get_lock():
            self.shared[:6] = [self.current[0], self.current[1], 0.0, self.current[2], float(t), 1.0]

    def accept(self, x: float, y: float, yaw: float, t: float) -> None:
        self.accepted += 1
        self.history.append((x, y, yaw))
        xs, ys, yaws = (np.asarray(v, dtype=np.float64) for v in zip(*self.history))
        self.set((float(np.median(xs)), float(np.median(ys)),
                  math.atan2(float(np.mean(np.sin(yaws))), float(np.mean(np.cos(yaws))))), t)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.f = open(path, "w", encoding="utf-8", newline="\n", buffering=1 << 16)
        self.n = 0
        self._last_flush = time.perf_counter()

    def write(self, doc: dict) -> None:
        self.f.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.n += 1
        now = time.perf_counter()
        if now - self._last_flush >= 1.0:
            self._last_flush = now
            self.f.flush()

    def close(self) -> None:
        self.f.close()


def pose_from_tag_lists(carloc, tag_lists: dict[str, list], t: float):
    """各相机认到的 tag 角点 → 双 tag 刚体拟合（CarLocalizer.estimate_car_pose，同 tracker）。只收双 tag、yaw 有效、
    重投影 ≤ tracker 的终门。"""
    from src import car_localizer as cl

    detections: dict[int, dict] = {}
    for serial, found in tag_lists.items():
        for tag_id, corners in found:
            c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            detections.setdefault(int(tag_id), {})[serial] = cl.CarDetection(
                tag_id=int(tag_id), cx=float(c[:, 0].mean()), cy=float(c[:, 1].mean()), corners=c)
    if len(detections) < 2 or not any(len(v) >= 2 for v in detections.values()):
        return None
    loc = carloc.estimate_car_pose(detections, t)
    if loc is None or len(loc.tag_ids) < 2 or not loc.yaw_valid or loc.yaw is None:
        return None
    if not all(math.isfinite(v) for v in (loc.x, loc.y, loc.yaw, loc.reprojection_error)):
        return None
    if not 0.0 <= loc.reprojection_error <= cl._MAX_REPROJECTION_ERROR_PX:
        return None
    return loc


def pc_car_loc_payload(loc) -> dict:
    """/pc_car_loc 载荷，与 tracker 主线（src/run_tracker.py 车定位线）逐字段相同。"""
    return {
        "topic": "car_loc", "x": round(loc.x, 4), "y": round(loc.y, 4), "z": round(loc.z, 4),
        "yaw": None if loc.yaw is None else round(loc.yaw, 4), "yaw_valid": loc.yaw_valid,
        "t": round(loc.t, 6), "tag_id": loc.tag_id, "tag_ids": loc.tag_ids,
    }


def car_row(loc, slot: Optional[int]) -> dict:
    return {
        "slot": slot, "t": round(loc.t, 6), "x": round(loc.x, 4), "y": round(loc.y, 4), "z": round(loc.z, 4),
        "yaw": round(loc.yaw, 5), "yaw_valid": loc.yaw_valid, "tag_id": loc.tag_id, "tag_ids": loc.tag_ids,
        "reference": "car_base", "cameras_used": loc.cameras_used,
        "reprojection_error": round(loc.reprojection_error, 2),
    }


def make_local_dds_config(base_xml: Path, out_path: Path) -> str:
    """在 tracker 的 CycloneDDS 配置上补本机 peer（= 配置里绑定的本机网卡地址），写到场次目录，返回 file:// URI。

    ros2/cyclonedds_18.xml 关了多播、Peers 只写了车，同一台 PC 上的两个参与者互相发现不了——tracker 的 bag
    里一直没有它自己发的 /pc_car_loc（tennis-man memory/rk-wifi-link-drop-0908.md）。本工具的话题由 ROS 桥发、
    rosbag 另一个进程录，必须能互相发现；tracker 那份配置不动。
    不用 127.0.0.1：Windows 上 Cyclone 的套接字绑在网卡地址上，往 127.0.0.1 发每包都报 retcode -3（1010 本机实测）。
    """
    text = Path(base_xml).read_text(encoding="utf-8")
    peers = re.findall(r'<NetworkInterface\s+address="([^"]+)"', text)
    if not peers:
        raise RuntimeError(f"{base_xml} 里没有 NetworkInterface address，不知道本机地址")
    missing = [a for a in dict.fromkeys(peers) if f'Peer Address="{a}"' not in text]
    if missing:
        insert = "".join(f'        <Peer Address="{a}" />  <!-- swing_hs：本机 ROS 桥 ↔ rosbag -->\n' for a in missing)
        text, n = re.subn(r"[ \t]*</Peers>", lambda m: insert + m.group(0), text, count=1)
        if n != 1:
            raise RuntimeError(f"{base_xml} 里没有 </Peers>，没法补本机 peer")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return "file://" + Path(out_path).resolve().as_posix()


class RosSide:
    """ROS 桥 + rosbag 两个子进程（都经 ros2/run_ros2.bat 起，ROS 自带的 pixi python），本机 UDP 通信。"""

    def __init__(self, session_dir: Path, session: str, ports: dict, camera_ports: list[int], topics: list[str],
                 best_effort_topics: list[str], joints: JointBuffer, dds_uri: str) -> None:
        self.session_dir = session_dir
        self.dds_uri = dds_uri
        self.best_effort_topics = best_effort_topics
        self.session = session
        self.ports = ports
        self.camera_ports = camera_ports
        self.topics = topics
        self.joints = joints
        self.bag_dir = session_dir / f"{session}_rosbag"
        self.bridge_stop = session_dir / "ros_bridge.stop"
        self.bag_stop = session_dir / f"{session}_rosbag.stop"
        self.bridge: Optional[subprocess.Popen] = None
        self.rosbag: Optional[subprocess.Popen] = None
        self.status: Optional[dict] = None
        self._send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._listener: Optional[threading.Thread] = None
        self._logs: list = []
        self.publish_errors = 0

    def _env(self) -> dict:
        # CYCLONEDDS_URI 必须由这里给：run_ros2.bat 在括号块里 set 再 %展开% 会拼出 "file://\=/"（1006 专场 rosbag 没起来）
        return {**os.environ, "CYCLONEDDS_URI": self.dds_uri}

    def _popen(self, args: list[str], log_name: Optional[str]) -> subprocess.Popen:
        """log_name=None：输出直接进本控制台（rosbag 同 tracker：订阅了哪些话题、球拍 IMU 连没连上都打在这里）。"""
        log = None
        if log_name is not None:
            log = open(self.session_dir / log_name, "w", encoding="utf-8", newline="\n")
            self._logs.append(log)
        return subprocess.Popen([str(REPO / "ros2" / "run_ros2.bat"), *args], env=self._env(), stdout=log,
                                stderr=subprocess.STDOUT if log is not None else None,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    def _listen(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(("127.0.0.1", int(self.ports["main"])))
        sock.settimeout(0.2)
        try:
            while not self._stop.is_set():
                try:
                    data = sock.recv(65536)
                except (socket.timeout, OSError):
                    continue
                try:
                    doc = udp_msgs.decode_any(data)
                except ValueError:
                    continue
                if doc.get("kind") == "bridge":
                    self.status = doc
                elif "s" in doc:
                    self.joints.add(float(doc["s"]), list(doc["n"]), [float(v) for v in doc["p"]], float(doc["r"]))
        finally:
            sock.close()

    def start(self) -> None:
        for path in (self.bridge_stop, self.bag_stop):
            path.unlink(missing_ok=True)
        self._listener = threading.Thread(target=self._listen, name="ros-listen", daemon=True)
        self._listener.start()
        self.bridge = self._popen([
            str(REPO / "swing_hs" / "ros_bridge.py"),
            "--listen-port", str(self.ports["bridge_listen"]), "--status-port", str(self.ports["main"]),
            "--joint-ports", *[str(p) for p in self.camera_ports], str(self.ports["main"]),
            "--topics", *self.topics, "--best-effort-topics", *self.best_effort_topics,
            "--stop-file", str(self.bridge_stop), "--parent-pid", str(os.getpid()),
        ], "ros_bridge.log")
        # 不连球拍 IMU（用户 1010：现在不开）：tracker.json 开着 racket_imu 时 IMU 找不到 / 2 s 没数据 rosbag 会自己退
        self.rosbag = self._popen([str(REPO / "src" / "rosbag_recorder.py"), "--output", str(self.bag_dir),
                                   "--stop-file", str(self.bag_stop), "--no-racket-imu"], None)

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.perf_counter() + timeout_s
        while self.status is None:
            if self.bridge is not None and self.bridge.poll() is not None:
                raise RuntimeError(f"ROS 桥起不来（退出码 {self.bridge.returncode}），看 {self.session_dir / 'ros_bridge.log'}")
            if time.perf_counter() > deadline:
                raise RuntimeError(f"ROS 桥 {timeout_s:g} s 内没回状态，看 {self.session_dir / 'ros_bridge.log'}")
            time.sleep(0.1)

    def publish(self, topic: str, payload: dict) -> None:
        try:
            self._send.sendto(udp_msgs.encode_publish(topic, payload), ("127.0.0.1", int(self.ports["bridge_listen"])))
        except OSError:
            self.publish_errors += 1

    def alive(self) -> dict:
        return {"bridge": self.bridge is not None and self.bridge.poll() is None,
                "rosbag": self.rosbag is not None and self.rosbag.poll() is None}

    def stop(self) -> dict:
        for path in (self.bridge_stop, self.bag_stop):
            try:
                path.write_text("stop", encoding="utf-8")
            except OSError:
                pass
        for name, proc in (("bridge", self.bridge), ("rosbag", self.rosbag)):
            if proc is None:
                continue
            try:
                proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                print(f"  {name} 20 s 没退，强杀", file=sys.stderr)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        self._stop.set()
        if self._listener is not None:
            self._listener.join(timeout=2.0)
        for log in self._logs:
            log.close()
        for path in (self.bridge_stop, self.bag_stop):
            path.unlink(missing_ok=True)
        self._send.close()
        return {"bag_dir": str(self.bag_dir), "bag_metadata": (self.bag_dir / "metadata.yaml").exists(),
                "bridge_status": self.status, "publish_errors": self.publish_errors}


# ───────────────── 主流程 ─────────────────


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_plan(plans: dict[str, geometry.CameraPlan]) -> list[str]:
    lines = ["  相机  ROI                    MP     上限fps  黑标在画内/在ROI   tag0             tag1"]
    for serial, p in plans.items():
        tags = []
        for tid in sorted(p.tags_in_roi):
            mark = "✓" if (p.tags_in_roi[tid] and not p.tags_dropped) else "✗"
            tags.append(f"{mark} {p.tag_side_px[tid]:3.0f}px {p.tag_view_deg[tid]:4.1f}°")
        roi = f"{p.roi[2]}x{p.roi[3]}@({p.roi[0]},{p.roi[1]})"
        lines.append(f"  {serial[-3:]}   {roi:<22s} {p.megapixels:5.2f}  {p.max_fps:6.1f}   "
                     f"{100 * p.marker_in_sensor:5.1f}% / {100 * p.marker_in_roi:5.1f}%   " + "   ".join(tags))
        lines.extend(f"        ⚠ {note}" for note in p.notes)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 车型、侧别都不给默认值：车型选错 tag 布局 / FK 全错且不报错；侧别选错 ROI 罩不住拍子
    ap.add_argument("--car", required=True, choices=["v04", "v05"])
    ap.add_argument("--side", required=True, choices=["fore", "back", "both"], help="正手 / 反手 / 两侧（ROI 取并集）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--duration", type=float, default=1800.0, help="最长录多久（s），到点自动收尾")
    ap.add_argument("--probe", type=float, default=None, help="自检：只录这么多秒，产物放 swing_hs_probe_*")
    ap.add_argument("--plan-only", action="store_true", help="开相机、定车位、出 ROI 规划和 roi_*.jpg 就停")
    ap.add_argument("--exposure-us", type=float, default=None, help="覆盖配置的曝光（默认 800 µs：1010 用户定 tracker 与本工具都用 0.8 ms + 自动增益）")
    ap.add_argument("--fps", type=float, default=None, help="覆盖配置的帧率（18F 灯 100 Hz，非 100/n 有拍频）")
    ap.add_argument("--j1-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="单侧时覆盖黑标包络的 J1 范围（度），默认正手 −45..45、反手 −225..−135")
    ap.add_argument("--car-pose", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW_DEG"),
                    help="开场认不到双 tag 时手给车位（世界 m / 度）定 ROI；录像中照样按 tag 更新")
    ap.add_argument("--no-ros", action="store_true", help="不起 ROS 桥和 rosbag（没有关节角就没有黑标，只用于相机自检）")
    ap.add_argument("--no-patches", action="store_true", help="不存每帧锚点小块")
    ap.add_argument("--video-fps", type=float, default=None,
                    help=f"录像帧率（整幅 ROI 原分辨率原始帧，默认 {VIDEO_FPS:g}；要整除触发帧率；0 = 不录）")
    args = ap.parse_args()

    cfg = _load_json(Path(args.config))
    cam_cfg = _load_json(REPO / cfg["camera_config"])
    calib_path = REPO / cfg["calibration"]
    vehicle_path = REPO / "src" / "config" / f"vehicle_{args.car}.json"
    envelope_path = Path(__file__).with_name(f"envelope_{args.car}.json")
    if not envelope_path.exists():
        print(f"没有 {envelope_path}：先跑 swing_hs/build_envelope.py --car {args.car} <实打场次…>", file=sys.stderr)
        return 2
    serials = list(cfg["serials"])
    fps = float(args.fps if args.fps is not None else cfg["fps"])
    period = 1.0 / fps
    exposure_us = float(args.exposure_us if args.exposure_us is not None else cfg["exposure_us"])
    duration = float(args.probe if args.probe is not None else args.duration)
    roi_cfg = cfg["roi"]
    margins = geometry.RoiMargins(marker_m=roi_cfg["marker_margin_m"], car_push_m=roi_cfg["car_push_m"],
                                  car_push_deg=roi_cfg["car_push_deg"], pixel=roi_cfg["pixel_margin"],
                                  tag_scale=roi_cfg["tag_scale"])
    headroom = float(roi_cfg["fps_headroom"])
    sides = ["fore", "back"] if args.side == "both" else [args.side]
    if args.j1_range is not None and len(sides) != 1:
        print("--j1-range 只能配单侧 --side", file=sys.stderr)
        return 2
    try:
        video_every = video_every_slots(fps, float(args.video_fps if args.video_fps is not None else VIDEO_FPS))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    video_fps = fps / video_every if video_every else 0.0
    detect_workers = DETECT_WORKERS

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # 只用来起目录名
    session = f"swing_hs_{'plan_' if args.plan_only else 'probe_' if args.probe is not None else ''}{stamp}"
    out_root = REPO / "tracker_output"
    session_dir = out_root / session
    session_dir.mkdir(parents=True, exist_ok=False)
    stop_file = out_root / STOP_FILE_NAME
    stop_file.unlink(missing_ok=True)

    # 黑标包络（车体系）
    j1_ranges = {s: (tuple(args.j1_range) if args.j1_range is not None else tuple(cfg["sides"][s]["j1_range_deg"]))
                 for s in sides}
    marker_car = np.vstack([
        geometry.marker_points_car(geometry.load_envelope(envelope_path, s, j1_ranges[s]), args.car) for s in sides
    ])
    z_offset = geometry.marker_z_offset_m(args.car)
    vehicle = geometry.load_vehicle_reference(vehicle_path)
    rbm = geometry.load_test_src("racket_ht_black_marker")
    cameras = rbm._load_cameras(calib_path, serials)
    from src.car_localizer import CarLocalizer

    carloc = CarLocalizer(str(calib_path), str(vehicle_path))
    print(f"[swing_hs] {session}  车 {args.car}  侧 {args.side}  {fps:g} fps  曝光 {exposure_us:g} µs  "
          f"黑标包络 {len(marker_car)} 点（J1 {j1_ranges}）  录像 "
          + (f"{video_fps:g} fps（每 {video_every} 格一帧，原分辨率）" if video_every else "不录")
          + f"  识别 {detect_workers} 进程/台")

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    ctrl_qs = {sn: ctx.Queue() for sn in serials}
    detect_qs = {sn: ctx.Queue(maxsize=DETECT_QUEUE_JOBS) for sn in serials}
    shared_car = ctx.Array("d", 6)
    t0_shared = ctx.Value("d", 0.0)
    cam_stop, cam_start, trig_stop = ctx.Event(), ctx.Event(), ctx.Event()
    ports = cfg["ports"]
    camera_ports = {sn: int(ports["camera_base"]) + i for i, sn in enumerate(serials)}
    procs: dict[str, Any] = {}
    meta: dict[str, Any] = {
        "session": session, "args": vars(args), "config": cfg, "camera_config": cam_cfg, "fps": fps,
        "exposure_us": exposure_us, "car": args.car, "side": args.side, "j1_ranges_deg": j1_ranges,
        "calibration": str(calib_path), "vehicle_config": str(vehicle_path), "envelope": str(envelope_path),
        "marker_z_offset_m": z_offset, "frame_time_reference": "exposure_center", "video_fps": video_fps,
        "video_every_slots": video_every, "detect_workers": detect_workers, "events": [],
    }
    errors: list[str] = []
    snapshots: dict[str, dict] = {}
    ready: dict[str, dict] = {}
    done: dict[str, dict] = {}
    latest: dict[str, dict] = {}
    done_det: dict[str, dict] = {}
    latest_det: dict[str, dict] = {}

    def handle_control(kind: str, who: str, payload: Any) -> bool:
        """非数据消息；返回 True 表示已处理。"""
        if kind == "snapshot":
            snapshots[who] = payload
        elif kind == "ready":
            ready[who] = payload
        elif kind == "done":
            done[who] = payload
        elif kind in ("stats", "trigger"):
            latest[who] = payload
        elif kind == "dstats":
            latest_det.setdefault(who, {})[payload["worker"]] = payload
        elif kind == "ddone":
            done_det.setdefault(who, {})[payload["worker"]] = payload
        elif kind == "warn":
            print(f"  [警告 {who}] {payload}")
            meta["events"].append({"warn": who, "msg": payload})
        elif kind == "error":
            print(f"  [出错 {who}] {payload}", file=sys.stderr)
            errors.append(f"{who}: {str(payload).splitlines()[0]}")
            meta["events"].append({"error": who, "msg": payload})
        else:
            return False
        return True

    def wait_for(store: dict, names: list[str], timeout_s: float, what: str) -> None:
        deadline = time.perf_counter() + timeout_s
        while not all(n in store for n in names) and not errors:
            if time.perf_counter() > deadline:
                errors.append(f"{what}：{timeout_s:g} s 内没等到 {[n for n in names if n not in store]}")
                return
            try:
                kind, who, payload = out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            handle_control(kind, who, payload)

    ros: Optional[RosSide] = None
    main_joints = JointBuffer(geometry.load_test_src("extract_arm_bag").CAR_MODELS[args.car].slot_joint_names)
    marker_writer = car_writer = None
    t_record: Optional[float] = None
    interrupted = False
    counters = {"slots": 0, "slots_incomplete": 0, "with_anchor": 0, "solved": 0, "by_cams": {"3": 0, "4": 0},
                "smooth": 0, "tag_slots": 0}
    car = CarPose(shared_car, int(cfg["car_loc"]["pose_median_window"]))
    fusion = MarkerFusion(cameras, serials)
    topics = cfg["topics"]
    try:
        # ── 1. 相机进程：自动增益 + 全幅样张 ──
        for sn in serials:
            spec = {
                "serial": sn, "all_serials": serials, "out_dir": str(session_dir / f"cam_{sn}"),
                "fps": fps, "acquisition_frame_rate": float(cfg["acquisition_frame_rate"]), "exposure_us": exposure_us,
                "gain_db": float(cam_cfg["gain_db"]), "digital_shift": cam_cfg.get("digital_shift"),
                "gev_scpd": int(cam_cfg.get("gev_scpd", 0)), "sdk_image_nodes": int(cfg["sdk_image_nodes"]),
                "action_trigger_source": cam_cfg["action_trigger_source"],
                "action_device_key": int(cam_cfg["action_device_key"]),
                "action_group_key": int(cam_cfg["action_group_key"]),
                "action_group_mask": int(cam_cfg["action_group_mask"]),
                "auto_gain_target": float(cfg["auto_gain_target"]), "car": args.car,
                "calib_path": str(calib_path), "vehicle_path": str(vehicle_path),
                "crop_radius_px": int(cfg["marker"]["crop_radius_px"]),
                "tag_every_frames": int(cfg["car_loc"]["tag_every_frames"]),
                "tag_window_pad_px": float(cfg["car_loc"]["tag_window_pad_px"]),
                "joint_port": camera_ports[sn], "record_patches": not args.no_patches,
                "video_every_slots": video_every, "detect_workers": detect_workers,
            }
            p = ctx.Process(target=camera_main,
                            args=(spec, out_q, ctrl_qs[sn], detect_qs[sn], shared_car, t0_shared, cam_stop, cam_start),
                            name=f"cam-{sn}")
            p.start()
            procs[sn] = p
            for k in range(detect_workers):
                p = ctx.Process(target=detect_main, args=(spec, k, detect_qs[sn], out_q), name=f"det-{sn}-{k}")
                p.start()
                procs[f"det-{sn}-{k}"] = p
        wait_for(snapshots, serials, 120.0, "相机开场（自动增益 + 全幅样张）")
        if errors:
            raise RuntimeError("; ".join(errors))
        for sn in serials:
            s = snapshots[sn]
            short = f"  ⚠ 还差 {s['gain_short_db']:+.1f} dB（增益顶到头，改曝光）" if s["gain_short_db"] else ""
            print(f"  {sn} 灰度中位数 {s['median_first']:.0f} → {s['median']:.0f}  增益 {s['gain_db']:.2f}"
                  f"+{s['digital_shift_db']:.2f} dB  认到 tag {sorted(t for t, _ in s['tags'])}{short}")
        meta["snapshots"] = {sn: {k: v for k, v in s.items() if k != "tags"} | {"tags": [t for t, _ in s["tags"]]}
                             for sn, s in snapshots.items()}

        # ── 2. 车位 + ROI 规划 ──
        t_snap = float(np.median([s["t"] for s in snapshots.values()]))    # 开场样张：只用来定 ROI，不进时间轴
        loc = pose_from_tag_lists(carloc, {sn: s["tags"] for sn, s in snapshots.items()}, t_snap)
        if loc is not None:
            pose = (loc.x, loc.y, loc.yaw)
            meta["car_pose_start"] = car_row(loc, None)
            print(f"  车位（双 tag，{len(loc.cameras_used)} 台）({loc.x:+.4f}, {loc.y:.4f}) yaw {math.degrees(loc.yaw):+.2f}°  "
                  f"重投影 {loc.reprojection_error:.2f} px")
        elif args.car_pose is not None:
            pose = (args.car_pose[0], args.car_pose[1], math.radians(args.car_pose[2]))
            meta["car_pose_start"] = {"manual": list(args.car_pose)}
            print(f"  ⚠ 开场没认到双 tag，用 --car-pose ({pose[0]:+.3f}, {pose[1]:.3f}) yaw {args.car_pose[2]:+.2f}°")
        else:
            raise RuntimeError("开场全幅样张没认到双 tag（车 yaw 必须双 tag）：看 cam_*/full_*.png，或用 --car-pose 手给车位")
        car.plan = pose
        car.set(pose, t_snap)
        plans = geometry.plan_rois(cameras, pose, marker_car, z_offset, vehicle, margins, fps=fps, fps_headroom=headroom)
        meta["roi_plan"] = {sn: p.to_json() for sn, p in plans.items()}
        meta["roi_margins"] = vars(margins)
        print("  ROI 规划（黑标包络 ∪ 两块 tag；车位扰动 "
              f"±{margins.car_push_m * 100:.0f} cm / ±{margins.car_push_deg:g}°，黑标外扩 {margins.marker_m * 100:.0f} cm）：")
        for line in _fmt_plan(plans):
            print(line)
        problems = geometry.plan_problems(plans, fps, headroom)
        if problems:
            meta["plan_problems"] = problems
            for msg in problems:
                print(f"  ✗ {msg}", file=sys.stderr)
            raise RuntimeError("ROI 规划不能开录（挪车 / 缩 J1 范围 / 单侧录）")
        tag_cameras = {sn for sn, p in plans.items() if not p.tags_dropped and any(p.tags_in_roi.values())}
        for sn, p in plans.items():
            nominal = geometry.project_m(cameras[sn], geometry.car_to_world(marker_car, pose, z_offset))
            step = max(1, len(nominal) // 600)
            tags_uv = [geometry.project_m(cameras[sn], geometry.car_to_world(c, pose)).tolist()
                       for c in geometry.tag_corners_car(vehicle, 1.0).values()]
            ctrl_qs[sn].put(("roi", {"roi": p.roi, "tags_in_roi": sn in tag_cameras,
                                     "overlay": {"marker_uv": nominal[::step].tolist(), "tags_uv": tags_uv}}))
        wait_for(ready, serials, 60.0, "相机切 ROI")
        if errors:
            raise RuntimeError("; ".join(errors))
        meta["readback"] = ready
        resulting = {sn: float(ready[sn]["resulting_fps"]) for sn in serials}
        sensor = {sn: float(ready[sn]["sensor_fps_model"]) for sn in serials}
        for sn in serials:
            print(f"  {sn} 就绪：ResultingFrameRate {resulting[sn]:.2f}（传感器 {sensor[sn]:.1f}）  包 {ready[sn]['packet_size']}")
            if resulting[sn] < fps + 0.5:
                raise RuntimeError(f"{sn} 相机读回上限 {resulting[sn]:.2f} fps < {fps:g}")
        minimum_period = 1.0 / min(sensor.values()) + TRIGGER_MARGIN_S
        if minimum_period > period:
            raise RuntimeError(f"触发周期 {1e3 * period:.2f} ms 短于相机最小 {1e3 * minimum_period:.2f} ms")
        print(f"  各相机 ROI 叠加图：{session_dir}\\cam_*\\roi_*.jpg")
        if video_every:
            rate = sum(int(p.roi[2]) * int(p.roi[3]) for p in plans.values()) * video_fps     # B/s（8-bit Bayer）
            free = shutil.disk_usage(session_dir).free
            meta["video_bytes_per_s"] = rate
            print(f"  录像：整幅 ROI 原始帧 {video_fps:g} fps，{len(plans)} 台共 {rate / 1e6:.1f} MB/s"
                  f"（{rate * 3600 / 1e9:.0f} GB/h），{session_dir.anchor} 剩 {free / 1e9:.0f} GB")
            if not args.plan_only and rate * duration > 0.9 * free:
                raise RuntimeError(f"录满 {duration:g} s 要 {rate * duration / 1e9:.0f} GB，盘上只剩 {free / 1e9:.0f} GB："
                                   "清盘、缩 --duration，或 --video-fps 0")
        if args.plan_only:
            print("[swing_hs] --plan-only：到此为止")
            return 0

        # ── 3. ROS ──
        if not args.no_ros:
            dds_uri = make_local_dds_config(REPO / cfg["cyclonedds_xml"], session_dir / "cyclonedds_swing_hs.xml")
            ros = RosSide(session_dir, session, ports, [camera_ports[sn] for sn in serials],
                          [topics["marker"], topics["status"]], [topics["car_loc"]], main_joints, dds_uri)
            ros.start()
            ros.wait_ready(40.0)
            print(f"  ROS 桥就绪；rosbag → {ros.bag_dir}")
        marker_writer = JsonlWriter(session_dir / "marker.jsonl")
        car_writer = JsonlWriter(session_dir / "car.jsonl")

        marker_asm = SlotAssembler(set(serials), SLOT_TIMEOUT)
        tag_asm = SlotAssembler(tag_cameras, SLOT_TIMEOUT)
        reorder = {sn: SeqReorder(DETECT_QUEUE_JOBS) for sn in serials}
        prev_rx: dict[str, int] = {}
        window = {"solved": 0, "slots": 0}

        def process_marker_slot(slot: int, msgs: dict) -> None:
            counters["slots"] += 1
            window["slots"] += 1
            if set(msgs) != set(serials):
                counters["slots_incomplete"] += 1
            t = float(np.median([m[2] for m in msgs.values()]))     # 帧时间（曝光中点）
            anchors = [m[4] for m in msgs.values() if m[4] is not None]
            anchor = np.median(np.asarray(anchors, dtype=np.float64), axis=0) if anchors else None
            q = next((m[5] for m in msgs.values() if m[5] is not None), None)
            cands = {s: m[6] for s, m in msgs.items() if m[6]}
            pose_now = car.current
            solution = None
            if anchor is not None and pose_now is not None:
                counters["with_anchor"] += 1
                solution, _ = fusion.solve_slot(anchor, pose_now[2], cands, t)
            if solution is not None:
                counters["solved"] += 1
                window["solved"] += 1
                key = str(min(4, len(solution.serials)))
                counters["by_cams"][key] = counters["by_cams"].get(key, 0) + 1
                counters["smooth"] += solution.smooth is True
            offset = fusion.offset.value
            marker_writer.write({
                "slot": slot, "t": round(t, 6), "q": q,
                "car": None if pose_now is None else [round(pose_now[0], 4), round(pose_now[1], 4), round(pose_now[2], 5)],
                "anchor_mm": None if anchor is None else [round(float(v), 1) for v in anchor],
                "status": {s: m[3] for s, m in msgs.items()},
                "frame_num": {s: m[1] for s, m in msgs.items()},
                "exposure_pc": {s: round(m[2], 6) for s, m in msgs.items()},
                "cands": {s: [list(r) for r in rows] for s, rows in cands.items()},
                "fit": None if solution is None else solution.to_json(),
                "offset_car_mm": None if offset is None else [round(float(v), 1) for v in offset],
            })
            if ros is not None:
                pc_minus_rk = main_joints.offset()
                xyz = None if solution is None else solution.xyz_mm / 1000.0
                ros.publish(topics["marker"], {
                    "t": round(t, 6), "slot": slot,
                    "x": None if xyz is None else round(float(xyz[0]), 4),
                    "y": None if xyz is None else round(float(xyz[1]), 4),
                    "z": None if xyz is None else round(float(xyz[2]), 4),
                    "n_cams": 0 if solution is None else len(solution.serials),
                    "smooth": None if solution is None else solution.smooth,
                    "rms_px": None if solution is None else round(solution.rms_px, 2),
                    "anchor": None if anchor is None else [round(float(v) / 1000.0, 4) for v in anchor],
                    "rk_t": None if pc_minus_rk is None else round(t - pc_minus_rk, 6),
                })

        def process_tag_slot(slot: int, msgs: dict) -> None:
            counters["tag_slots"] += 1
            t = float(np.median([m[1] for m in msgs.values()]))     # 帧时间（曝光中点），同 tracker 车位的 t
            loc = pose_from_tag_lists(carloc, {s: m[2] for s, m in msgs.items()}, t)
            if loc is None:
                car.missed += 1
                return
            car.accept(loc.x, loc.y, loc.yaw, t)
            car_writer.write(car_row(loc, slot))
            if ros is not None:
                ros.publish(topics["car_loc"], pc_car_loc_payload(loc))

        def dispatch(kind: str, who: str, body: tuple) -> None:
            if kind == "f":
                for slot, msgs in marker_asm.add(body[0], who, body):
                    process_marker_slot(slot, msgs)
            elif who in tag_asm.expected:
                for slot, msgs in tag_asm.add(body[0], who, body):
                    process_tag_slot(slot, msgs)

        def handle(kind: str, who: str, payload: Any) -> None:
            if kind in ("f", "t"):
                seq, body = payload
                for k, b in reorder[who].push(seq, (kind, body)):
                    dispatch(k, who, b)
            else:
                handle_control(kind, who, payload)

        # ── 4. 开录 ──
        cam_start.set()
        time.sleep(0.5)
        trig_spec = {
            "fps": fps, "minimum_period_s": minimum_period,
            "action_device_key": int(cam_cfg["action_device_key"]), "action_group_key": int(cam_cfg["action_group_key"]),
            "action_group_mask": int(cam_cfg["action_group_mask"]),
            "action_broadcast_address": cam_cfg["action_broadcast_address"],
            "action_timeout_ms": int(cam_cfg["action_timeout_ms"]),
        }
        p = ctx.Process(target=trigger_main, args=(trig_spec, str(session_dir / "trigger.bin"), out_q, trig_stop, t0_shared),
                        name="trigger")
        p.start()
        procs["trigger"] = p
        t_record = time.perf_counter()
        meta["record_start_perf"] = t_record
        print(f"[swing_hs] 开录（{duration:g} s 上限）。停止：Ctrl+C 或创建 {stop_file}")

        last_print = last_status = last_drift_warn = t_record
        while True:
            try:
                batch = [out_q.get(timeout=0.05)]
            except queue.Empty:
                batch = []
            while len(batch) < 4000:
                try:
                    batch.append(out_q.get_nowait())
                except queue.Empty:
                    break
            for kind, who, payload in batch:
                handle(kind, who, payload)
            now = time.perf_counter()
            if errors:
                print("[swing_hs] 有进程出错，收尾", file=sys.stderr)
                break
            dead = [k for k, pr in procs.items() if not pr.is_alive()]
            if dead:
                errors.append(f"进程意外退出: {dead}")
                print(f"[swing_hs] 进程意外退出 {dead}，收尾", file=sys.stderr)
                break
            if ros is not None and not all(ros.alive().values()):
                # 同 tracker：录制进程一退就停采、保存已有结果（没有 bag 的场次不完整；桥没了就没有关节角、认不了黑标）
                errors.append(f"ROS 子进程退出：{ros.alive()}")
                print(f"[swing_hs] ROS 子进程退出 {ros.alive()}，收尾。桥看 ros_bridge.log，rosbag 看上面它自己的输出",
                      file=sys.stderr)
                break
            if stop_file.exists():
                print("[swing_hs] 看到停止文件，收尾")
                break
            if now - t_record >= duration:
                print("[swing_hs] 到时长，收尾")
                break
            if car.current is not None and car.plan is not None and now - last_drift_warn >= DRIFT_WARN_PERIOD_S:
                dx = math.hypot(car.current[0] - car.plan[0], car.current[1] - car.plan[1])
                dyaw = abs(math.degrees(math.atan2(math.sin(car.current[2] - car.plan[2]), math.cos(car.current[2] - car.plan[2]))))
                if dx > margins.car_push_m or dyaw > margins.car_push_deg:
                    last_drift_warn = now
                    print(f"  ⚠ 车被推走 {100 * dx:.1f} cm / {dyaw:.1f}°（超出 ROI 余量 ±{100 * margins.car_push_m:.0f} cm / "
                          f"±{margins.car_push_deg:g}°）：黑标可能出 ROI，建议停下重开（ROI 按新车位重算）")
            if ros is not None and now - last_status >= 1.0:
                last_status = now
                ros.publish(topics["status"], {
                    "t": round(now, 3), "elapsed_s": round(now - t_record, 1), "cameras": latest,
                    "counters": counters, "car": car.current, "joint_rx": main_joints.received,
                })
            if now - last_print >= 2.0:
                dt = now - last_print
                last_print = now
                parts = []
                for sn in serials:
                    s = latest.get(sn)
                    if not s:
                        parts.append(f"{sn[-3:]} --")
                        continue
                    rate = (s["received"] - prev_rx.get(sn, 0)) / dt
                    prev_rx[sn] = s["received"]
                    d = merge_detect_stats(latest_det.get(sn) or {})
                    proc = max(1, d["processed"])
                    parts.append(f"{sn[-3:]} {rate:5.1f}fps 丢{s['proc_dropped'] + s['missing'] + s['detect_dropped']} "
                                 f"积压{d['backlog']:<3d} 候选{100 * d['with_candidates'] / proc:3.0f}% "
                                 f"{d['proc_ms_sum'] / proc:4.1f}ms")
                solved = 100.0 * window["solved"] / max(1, window["slots"])
                window["solved"] = window["slots"] = 0
                tr = latest.get("trigger") or {}
                pose_txt = "--" if car.current is None else \
                    f"({car.current[0]:+.3f},{car.current[1]:.3f}) {math.degrees(car.current[2]):+.1f}°"
                jt = "--" if not main_joints.received else f"{main_joints.received}"
                print(f"[{now - t_record:6.1f}s] " + " | ".join(parts)
                      + f" | 黑标解出 {solved:3.0f}% | 车 {pose_txt} n{car.accepted} | 关节 {jt}"
                      + f" | 触发 {tr.get('issued', 0)} 跳{tr.get('skipped_slots', 0)} 迟>1ms {tr.get('late_over', {}).get('1ms', 0)}",
                      flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[swing_hs] Ctrl+C，收尾")
    except Exception as exc:
        errors.append(repr(exc))
        print(f"[swing_hs] {exc}", file=sys.stderr)
    finally:
        # 停触发 → 等在途帧落地 → 停相机（各自把剩下的帧认完、收尾）→ 凑剩下的格 → 停 ROS（rosbag 写 metadata）
        if t_record is not None:
            meta["record_stop_perf"] = time.perf_counter()
        trig_stop.set()
        if "trigger" in procs:
            procs["trigger"].join(timeout=5.0)
        time.sleep(0.3)
        for sn in serials:
            ctrl_qs[sn].put(("abort", None))
        cam_stop.set()
        deadline = time.perf_counter() + 120.0
        streaming = t_record is not None
        sentinel_sent: set[str] = set()

        def detect_alive() -> bool:
            for sn in serials:
                if sn in procs and not procs[sn].is_alive() and sn not in sentinel_sent:
                    sentinel_sent.add(sn)
                    for _ in range(detect_workers):
                        try:
                            detect_qs[sn].put(None, timeout=1.0)
                        except queue.Full:
                            break
            return any(procs[k].is_alive() for k in procs if k.startswith("det-"))

        while time.perf_counter() < deadline and (
                any(procs[sn].is_alive() for sn in serials if sn in procs) or detect_alive()):
            try:
                kind, who, payload = out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            try:
                if streaming and marker_writer is not None:
                    handle(kind, who, payload)
                else:
                    handle_control(kind, who, payload)
            except Exception as exc:  # 收尾阶段只记不停
                meta["events"].append({"error": "drain", "msg": repr(exc)})
        try:
            while True:
                kind, who, payload = out_q.get_nowait()
                if streaming and marker_writer is not None:
                    handle(kind, who, payload)
                else:
                    handle_control(kind, who, payload)
        except (queue.Empty, EOFError, OSError):
            pass
        if streaming and marker_writer is not None:
            for sn in serials:
                for k, b in reorder[sn].flush():
                    dispatch(k, sn, b)
            for slot, msgs in marker_asm.expire(force=True):
                process_marker_slot(slot, msgs)
            for slot, msgs in tag_asm.expire(force=True):
                process_tag_slot(slot, msgs)
        if ros is not None:
            meta["ros"] = ros.stop()
        for writer in (marker_writer, car_writer):
            if writer is not None:
                writer.close()
        for k, pr in procs.items():
            pr.join(timeout=3.0)
            if pr.is_alive():
                print(f"  进程 {k} 没退出，强杀", file=sys.stderr)
                pr.kill()
        meta["interrupted"] = interrupted
        meta["errors"] = errors
        meta["done"] = done
        meta["done_detect"] = done_det
        meta["late_messages"] = {"marker": marker_asm.late if streaming else 0, "tag": tag_asm.late if streaming else 0}
        meta["reorder_skipped"] = {sn: reorder[sn].skipped for sn in serials} if streaming else {}
        if streaming:
            meta["summary"] = summarize(meta, counters, car, fusion, done, latest, done_det, latest_det, main_joints)
            print(json.dumps(meta["summary"], ensure_ascii=False, indent=2))
        write_json(session_dir / "meta.json", meta)
        print(f"[swing_hs] meta → {session_dir / 'meta.json'}")
        stop_file.unlink(missing_ok=True)
    return 1 if errors else 0


def summarize(meta: dict, counters: dict, car: CarPose, fusion: MarkerFusion, done: dict, latest: dict,
              done_det: dict, latest_det: dict, joints: JointBuffer) -> dict:
    duration = (meta.get("record_stop_perf") or time.perf_counter()) - meta["record_start_perf"]
    out: dict[str, Any] = {"duration_s": round(duration, 2)}
    trig = done.get("trigger") or latest.get("trigger") or {}
    out["trigger"] = trig
    issued = int(trig.get("issued") or 0)
    cams = {}
    for sn, s in done.items():
        if sn == "trigger":
            continue
        d = merge_detect_stats({**(latest_det.get(sn) or {}), **(done_det.get(sn) or {})})
        prep = max(1, int(s.get("prepared", 0)))
        proc = max(1, int(d["processed"]))
        cams[sn] = {
            "received": s.get("received"), "vs_issued": round(s.get("received", 0) / issued, 4) if issued else None,
            "missing": s.get("missing"), "lost_packet_frames": s.get("lost_packet_frames"),
            "proc_dropped": s.get("proc_dropped"), "detect_dropped": s.get("detect_dropped"),
            "prepared": s.get("prepared"), "processed": d["processed"], "detect_workers": d["workers"],
            "with_candidates": round(d["with_candidates"] / proc, 4),
            "out_of_roi": round(s.get("out_of_roi", 0) / prep, 4),
            "no_joint_state": round(s.get("no_joint_state", 0) / prep, 4),
            "no_car_pose": round(s.get("no_car_pose", 0) / prep, 4),
            "prep_ms_mean": round(s.get("prep_ms_sum", 0.0) / prep, 2), "prep_ms_max": round(s.get("prep_ms_max", 0.0), 1),
            "detect_ms_mean": round(d["proc_ms_sum"] / proc, 2),
            "detect_ms_max": round(d["proc_ms_max"], 1), "detect_backlog_max": d["backlog_max"],
            "joint_wait_ms_mean": round(s.get("joint_wait_ms_sum", 0.0) / prep, 2),
            "tag_blocks": s.get("tag_blocks"), "tag_hits": d["tag_hits"], "tag_runs": d["tag_runs"],
            "patches_written": s.get("patches_written"), "video_frames": s.get("video_frames"),
            "video_dropped": s.get("video_dropped"),
            "joint_name_mismatch": s.get("joint_name_mismatch"),
        }
    out["cameras"] = cams
    slots = max(1, counters["slots"])
    out["marker"] = {
        **counters, "solved_rate": round(counters["solved"] / slots, 4),
        "solved_rate_with_anchor": round(counters["solved"] / max(1, counters["with_anchor"]), 4),
        "offset_car_mm": None if fusion.offset.value is None else [round(float(v), 1) for v in fusion.offset.value],
        "offset_inliers": fusion.offset.inliers,
    }
    out["car"] = {
        "accepted": car.accepted, "missed": car.missed, "median_pose": car.current, "plan_pose": car.plan,
        "drift_m": None if (car.current is None or car.plan is None) else
        round(math.hypot(car.current[0] - car.plan[0], car.current[1] - car.plan[1]), 4),
    }
    out["joints"] = {"received": joints.received, "pc_minus_rk_s": joints.offset(), "name_mismatch": joints.name_mismatch}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
