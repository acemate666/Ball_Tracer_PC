# -*- coding: utf-8 -*-
"""从 tracker rosbag 提取机械臂数据为 {run_id}_arm.json。

必须在 ROS2 环境中运行（经 ros2/run_ros2.bat 启动），依赖 rosbag2_py 与
tennis-man arm_controller 的 compact_arm_kinematics.fk（TCP 正解，直接
sys.path 引用，不在本项目重复实现）。

输出供 test_src/generate_curve3_html.py 的 Arm tab 使用：
  states   — /joint_states 实际关节位置/速度/力矩 + FK TCP
  commands — /tennis/motor_command 目标（首个轨迹点）+ FK TCP
  events   — status / arm_command / hit_pos / predict_hit_pos 文本事件

时间轴 t 为相对 bag 第一条消息（接收时刻）的秒数。时间来源：
  states/commands — header.stamp（发送端时钟，消息自带；相对时序无接收抖动），
    用常数 median(stamp − recv) 校正"发送端时钟差 + 中位传输延迟"后落到接收轴；
  events — bag 接收时刻。std_msgs/String 没有 header，消息本身不带时间戳；
    /predict_hit_pos 的 payload 带 ct/ht（RK steady 时钟），由 HTML 端做桥。
    要根治需发布端把发布时刻写进 payload（或换带 header 的消息类型）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_ARM_SRC = Path(r"D:\tennis-man\arm_controller\src")

EVENT_TOPICS = (
    "/tennis/status",
    "/tennis/arm_command",
    "/tennis/hit_pos",
    "/predict_hit_pos",
    "/arm_controller/status",
)


def _ordered(values: list[float], names: list[str], joint_names: tuple[str, ...]) -> list[float | None]:
    """按 joint_names 顺序重排（与 session_viewer._ordered 一致）。"""
    by_name = {name: idx for idx, name in enumerate(names)}
    ordered: list[float | None] = []
    for idx, name in enumerate(joint_names):
        src = by_name.get(name, idx if not names else None)
        ordered.append(float(values[src]) if src is not None and src < len(values) else None)
    return ordered


def _round_list(values: list[float | None], digits: int) -> list[float | None]:
    return [None if v is None else round(v, digits) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag 目录（含 metadata.yaml）")
    parser.add_argument("--output", type=Path, required=True, help="输出 arm JSON 路径")
    parser.add_argument(
        "--arm-src", type=Path, default=DEFAULT_ARM_SRC,
        help="tennis-man arm_controller 的 src 目录（提供 FK）",
    )
    args = parser.parse_args()

    if not (args.arm_src / "arm_controller").is_dir():
        raise FileNotFoundError(f"arm_controller package not found under: {args.arm_src}")
    sys.path.insert(0, str(args.arm_src))

    from arm_controller.compact_arm_kinematics import SHORT_JOINT_NAMES, fk  # noqa: E402

    import rosbag2_py  # noqa: E402
    from rclpy.serialization import deserialize_message  # noqa: E402
    from rosidl_runtime_py.utilities import get_message  # noqa: E402

    joint_names = tuple(SHORT_JOINT_NAMES)

    def tcp_of(positions: list[float | None]) -> list[float] | None:
        if any(v is None for v in positions):
            return None
        try:
            return [round(float(v), 4) for v in fk(positions)["tcp"]]
        except Exception:
            return None

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    type_by_topic = {item.name: item.type for item in reader.get_all_topics_and_types()}
    msg_types = {}
    for topic, type_name in type_by_topic.items():
        try:
            msg_types[topic] = get_message(type_name)
        except Exception:
            pass  # 无法解析的类型只计数，不采样

    states: list[dict] = []
    commands: list[dict] = []
    events: list[dict] = []
    state_stamps: list[float] = []   # (stamp − recv) 样本，秒
    command_stamps: list[float] = []
    counts: dict[str, int] = {}
    seen_state_names: list[str] = []
    seen_command_names: list[str] = []
    start_ns: int | None = None
    end_ns: int | None = None

    def _header_stamp_sec(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        if start_ns is None:
            start_ns = timestamp
        end_ns = timestamp
        msg_type = msg_types.get(topic)
        if msg_type is None:
            continue
        t = round((timestamp - start_ns) / 1e9, 4)

        if topic == "/joint_states":
            msg = deserialize_message(data, msg_type)
            names = list(msg.name)
            if not seen_state_names and names:
                seen_state_names = names
            positions = _ordered(list(msg.position), names, joint_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 1e6:
                state_stamps.append(stamp - timestamp / 1e9)
            states.append(
                {
                    "t": t,
                    "stamp": stamp,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(msg.velocity), names, joint_names), 5),
                    "effort": _round_list(_ordered(list(msg.effort), names, joint_names), 5),
                    "tcp": tcp_of(positions),
                }
            )
        elif topic == "/tennis/motor_command":
            msg = deserialize_message(data, msg_type)
            if not msg.points:
                continue
            point = msg.points[0]
            names = list(msg.joint_names)
            if not seen_command_names and names:
                seen_command_names = names
            positions = _ordered(list(point.positions), names, joint_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 1e6:
                command_stamps.append(stamp - timestamp / 1e9)
            commands.append(
                {
                    "t": t,
                    "stamp": stamp,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(point.velocities), names, joint_names), 5),
                    "effort": _round_list(_ordered(list(point.effort), names, joint_names), 5),
                    "tcp": tcp_of(positions),
                }
            )
        elif topic in EVENT_TOPICS:
            msg = deserialize_message(data, msg_type)
            if hasattr(msg, "data"):
                raw = msg.data
                text = (
                    " ".join(f"{float(v):.4g}" for v in raw)
                    if isinstance(raw, (list, tuple)) or type(raw).__name__ == "array"
                    else str(raw)
                )
            else:
                text = str(msg)
            events.append({"t": t, "topic": topic, "text": text[:500]})

    if start_ns is None:
        raise RuntimeError(f"bag has no messages: {args.bag}")

    # 时间源改为消息自带的 header.stamp：相对时序取发送端时钟（无接收抖动），
    # 用 median(stamp − recv) 一个常数把发送端时钟差+中位传输延迟校正回接收轴，
    # 使 t 与 events（接收时刻，String 无 header）可比。stamp 缺失(=0)保留接收 t。
    def _rebase(rows: list[dict], stamp_diffs: list[float]) -> float | None:
        med = statistics.median(stamp_diffs) if stamp_diffs else None
        t0 = start_ns / 1e9
        for row in rows:
            stamp = row.pop("stamp")
            if med is not None and stamp > 1e6:
                row["t"] = round(stamp - med - t0, 4)
        return None if med is None else round(med, 4)

    state_stamp_offset = _rebase(states, state_stamps)
    command_stamp_offset = _rebase(commands, command_stamps)

    result = {
        "schema": "tracker_arm_bag_v2",
        "time_sources": {
            "states": "header.stamp − median(stamp−recv)（缺 stamp 退回接收时刻）",
            "commands": "header.stamp − median(stamp−recv)（缺 stamp 退回接收时刻）",
            "events": "bag 接收时刻（std_msgs/String 无 header；predict payload 带 ct/ht）",
        },
        "joint_states_stamp_minus_recv_sec": state_stamp_offset,
        "motor_command_stamp_minus_recv_sec": command_stamp_offset,
        "bag_dir": str(args.bag.resolve()),
        "fk_source": "arm_controller.compact_arm_kinematics.fk",
        "start_ns": start_ns,
        "duration_sec": round((end_ns - start_ns) / 1e9, 4),
        "joint_names": list(joint_names),
        "state_joint_names_raw": seen_state_names,
        "command_joint_names_raw": seen_command_names,
        "topics": [
            {"name": name, "type": type_by_topic.get(name, ""), "count": counts[name]}
            for name in sorted(counts)
        ],
        "states": states,
        "commands": commands,
        "events": events,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        "arm json saved: %s (states=%d commands=%d events=%d duration=%.1fs)"
        % (args.output, len(states), len(commands), len(events), result["duration_sec"])
    )
    print("state joint names: %s" % (seen_state_names,))
    print("command joint names: %s" % (seen_command_names,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
