#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swing_hs 的 ROS 桥：run_swing_hs.py 经 ros2/run_ros2.bat 起（ROS 自带的 pixi python，rclpy 在这边最稳；
1006 专场在相机进程那套 venv 的 spawn 子进程里 import rclpy 卡死过，LoadLibrary 126）。

  - 订阅 /joint_states → 每条按 UDP 发给各相机进程 + 主进程（收到时刻 = 本进程 perf_counter，全机同一根 PC 轴）；
  - 收主进程 UDP 发来的发布请求 → 发 std_msgs/String（RELIABLE，rosbag 录得全）；
  - 每秒给主进程回一条状态；停止文件出现 / 主进程没了就退出。
只用标准库 + rclpy：同目录的 udp_msgs / winutil 当顶层模块导入，不碰仓库其它代码（pixi python 里没有 cv2 / 相机 SDK）。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import udp_msgs  # noqa: E402
from winutil import pid_alive  # noqa: E402

#: 停止文件 / 主进程存活多久查一次
_CHECK_PERIOD_S = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-port", type=int, required=True, help="收主进程发布请求的本机 UDP 端口")
    ap.add_argument("--status-port", type=int, required=True, help="状态回给主进程的端口")
    ap.add_argument("--joint-ports", type=int, nargs="+", required=True, help="/joint_states 转发到这些本机端口")
    ap.add_argument("--topics", nargs="*", default=[], help="开场就建好的发布话题（std_msgs/String，RELIABLE depth 200）")
    ap.add_argument("--best-effort-topics", nargs="*", default=[],
                    help="按 tracker 契约用 BEST_EFFORT depth 1 发的话题（/pc_car_loc）")
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--parent-pid", type=int, default=0)
    args = ap.parse_args()

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("swing_hs_bridge")
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    joint_dests = [("127.0.0.1", int(p)) for p in args.joint_ports]
    counters = {"joint_rx": 0, "joint_send_errors": 0, "published": 0, "publish_errors": 0}
    joint_names: list[str] = []

    def on_joint(msg) -> None:
        recv_pc = time.perf_counter()
        stamp = msg.header.stamp
        data = udp_msgs.encode_joint_state(stamp.sec + 1e-9 * stamp.nanosec, recv_pc, list(msg.name), list(msg.position))
        counters["joint_rx"] += 1
        if not joint_names:
            joint_names.extend(msg.name)
        for dest in joint_dests:
            try:
                send_sock.sendto(data, dest)
            except OSError:
                counters["joint_send_errors"] += 1

    best_effort = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=50,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
    reliable = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=200,
                          reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
    # 同 src/ros2_support.make_best_effort_qos()（tracker 发 /pc_car_loc 用的）
    latest_only = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
    node.create_subscription(JointState, "/joint_states", on_joint, best_effort)
    publishers = {topic: node.create_publisher(String, topic, reliable) for topic in args.topics}
    publishers.update({topic: node.create_publisher(String, topic, latest_only) for topic in args.best_effort_topics})
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, name="spin", daemon=True)
    spinner.start()

    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    listen.bind(("127.0.0.1", int(args.listen_port)))
    listen.settimeout(0.1)
    status_dest = ("127.0.0.1", int(args.status_port))
    stop_file = Path(args.stop_file)
    last_check = last_status = 0.0
    try:
        while True:
            now = time.perf_counter()
            if now - last_check >= _CHECK_PERIOD_S:
                last_check = now
                if stop_file.exists() or (args.parent_pid and not pid_alive(args.parent_pid)):
                    break
            if now - last_status >= 1.0:
                last_status = now
                doc = {**counters, "joint_names": joint_names, "t": now,
                       "subscribers": {t: p.get_subscription_count() for t, p in publishers.items()}}
                try:
                    send_sock.sendto(udp_msgs.encode_status(doc), status_dest)
                except OSError:
                    pass
            try:
                data = listen.recv(65536)
            except (socket.timeout, OSError):
                continue
            try:
                topic, payload = udp_msgs.decode_publish(data)
                publisher = publishers.get(topic)
                if publisher is None:
                    publisher = publishers[topic] = node.create_publisher(String, topic, reliable)
                publisher.publish(String(data=payload))
                counters["published"] += 1
            except Exception:
                counters["publish_errors"] += 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        listen.close()
        send_sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
