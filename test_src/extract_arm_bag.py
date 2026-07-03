# -*- coding: utf-8 -*-
"""从 tracker rosbag 提取机械臂数据为 {run_id}_arm.json。

必须在 ROS2 环境中运行（经 ros2/run_ros2.bat 启动），依赖 rosbag2_py 与
tennis-man arm_controller 的 compact_arm_kinematics.fk（TCP 正解，直接
sys.path 引用，不在本项目重复实现）。

输出供 test_src/generate_curve3_html.py 的 Arm tab 使用：
  states   — /joint_states 实际关节位置/速度/力矩 + FK TCP
  commands — /tennis/motor_command 目标（首个轨迹点）+ FK TCP
  events   — status / arm_command / hit_pos / predict_hit_pos 文本事件
时间轴 t 为相对 bag 第一条消息的秒数。
"""

from __future__ import annotations

import argparse
import json
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
    counts: dict[str, int] = {}
    seen_state_names: list[str] = []
    seen_command_names: list[str] = []
    start_ns: int | None = None
    end_ns: int | None = None

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
            states.append(
                {
                    "t": t,
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
            commands.append(
                {
                    "t": t,
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

    result = {
        "schema": "tracker_arm_bag_v1",
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
