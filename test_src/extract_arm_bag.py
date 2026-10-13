# -*- coding: utf-8 -*-
"""从 tracker rosbag 提取机械臂原始数据为 {run_id}_arm.json（只抽原始量，不算任何臂派生量）。

必须在 ROS2 环境中运行（经 ros2/run_ros2.bat 启动），依赖 rosbag2_py。
黑标 / 甜点 / 拍面 / 拍速、方案收不收、零位平移这些都**不在这里算**：报告生成时（generate_curve3_html → test_src/arm_production.py）
用当时的臂生产代码（tennis-man/arm_controller 的端点模型 / node::receive_hit_pos / 零位帧登记表）现算——臂代码一变，报告重出就跟着变
（2026-10-13 用户「report 也只用一份代码」）。ROS2 解释器的 PATH 里也没有 g++，臂的 C 接口库在这里本来就调不了。

⚠ 车型（--car v04|v05）决定 /joint_states 关节名进哪个控制器槽，**没有默认值**。不给 --car 时从同目录的 tracker JSON
（config.car_config_path，run_tracker 按启动的 --car 写入）推断，推不出来直接失败。选中的车型写进输出的 "car"/"car_source"，
报告端按它取生产车型、并用生产代码（ArmEndpoint.slot_names）复核槽位，不自己猜。

关节一律按控制器的 6 个槽输出（position/velocity/effort 都是 6 列，joint_names 是各槽的名字）：槽名 = 臂车型 yaml
motors[].joint_name 按控制器槽位规则进槽（6 个电机 = 槽 0..5；5 个 = 五轴臂没有腕转，占槽 0,1,2,3,5，槽 4 空、恒 0）。
包里的关节名对不上所选车型直接失败——那是拿错车的另一种征兆。

输出供 test_src/generate_curve3_html.py 的 Arm tab 与北极星表使用：
  states   — /joint_states 实际关节位置/速度/力矩
  commands — /tennis/motor_command 目标（首个轨迹点）
  link_commands — /tennis/link_command（链侧补偿开时才有：补偿前的链侧命令；commands 此时是电机参考 = 链侧 + δ）
  events   — status / arm_command / hit_pos / predict_hit_pos 文本事件
  run_configs — 臂节点起栈的 /rosout run_config 行（只有臂起栈前就开录的包才有；其中 zero_frame = 录包关节角的零位帧）

时间轴（全项目只有两个时间轴）：
  所有 t 一律为 RK perf_counter 单调钟绝对秒 —— 与 /predict_hit_pos
  的 generated_at/source_ct/contact_ht（历史 ct/ht）、rk_tracking payload t 同钟。报告端固定用 scale=1 的 RK→PC
  常数偏移映射，臂数据没有任何独立的桥。

  新固件（damiao 驱动 + arm_controller 单调钟版）：
    /joint_states、/tennis/motor_command 的 header.stamp 就是 RK 单调钟，
    /tennis/status 文本尾缀 " t=<单调秒>"，全部原生直读、零换算。
  旧 bag 兼容（stamp 为 epoch 系统钟、status 无 t= 的场次）：
    t = stamp − median(stamp − recv) + median(bot_state.t − recv)，
    两个中位数里的传输延迟互相抵消，残差 ~ms 级；status 等无时间事件
    退回 recv + median(bot_state.t − recv)。clock_sync 里明示所用模式。

  clock_sync 自检随文件输出：RK 单调钟 vs PC 收钟漂移率（ms/min）、
  joint_states stamp 时钟域、ht 锚点残差（新调度触球≡ht 时 done−ht 中位，
  验证时间链）。漂移率若超 ~2ms/min，单 offset 假设需复查。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path

# 臂仓库（车型 yaml 在 cpp/arm_controller_cpp/config/cars/<car>.yaml）；报告端调生产代码也用同一个环境变量（arm_production.py）
ARM_CONTROLLER_ROOT = Path(os.environ.get("TENNIS_MAN_ARM_CONTROLLER", "D:/tennis-man/arm_controller"))
CAR_YAML_DIR = ARM_CONTROLLER_ROOT / "cpp" / "arm_controller_cpp" / "config" / "cars"
# 有臂车型 yaml 的现役车型（v0.3 已随车从臂删除：它的场次抽不了槽位，报告端标旧版本）
CARS = ("v04", "v05")
# 电机 → 控制器槽，臂 C++ car_config（kSixMotorSlots / kFiveMotorSlots）的规则：6 个电机 = 槽 0..5；5 个 = 槽 0,1,2,3,5。
# 抽取在 ROS2 解释器里跑、调不了臂的 C 接口库，只好按这条规则排槽；报告生成时用生产 ArmEndpoint.slot_names 复核，对不上不算端点。
_MOTOR_SLOTS = {6: (0, 1, 2, 3, 4, 5), 5: (0, 1, 2, 3, 5)}
# 输出 joint_names 里空槽的名字（报告 Arm tab 拿它当图例；该槽数值恒 0）
EMPTY_SLOT_LABEL = "腕转(空槽)"
SHORT_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def slot_joint_names(car: str) -> tuple:
    """各槽的 ROS 关节名。真值 = 臂车型 yaml 的 motors[].joint_name（控制器发 /joint_states 用的就是它们）。"""
    import yaml

    path = CAR_YAML_DIR / f"{car}.yaml"
    if not path.is_file():
        raise SystemExit(f"车型 {car} 没有臂车型 yaml：{path}（TENNIS_MAN_ARM_CONTROLLER 指向臂 checkout）")
    names = [str(m["joint_name"]) for m in yaml.safe_load(path.read_text(encoding="utf-8"))["motors"]]
    slots = _MOTOR_SLOTS.get(len(names))
    if slots is None:
        raise SystemExit(f"{path}：{len(names)} 个电机，控制器只支持 5 或 6 个")
    out: list = [None] * 6
    for slot, name in zip(slots, names):
        out[slot] = name
    return tuple(out)


# tracker JSON 的 config.car_config_path 文件名 → 车型（src/run_tracker.py CAR_LAYOUT_CONFIGS 的逆）。
CAR_BY_LAYOUT_CONFIG = {
    "arm_poe_racket_center.json": "v03",
    "vehicle_v04.json": "v04",
    "vehicle_v05.json": "v05",
}
KNOWN_CARS = tuple(sorted(set(CAR_BY_LAYOUT_CONFIG.values())))


def car_for_tracker_json(tracker_json: Path | str) -> tuple[str, str]:
    """从一场 tracker JSON 推车型，返回 (car, 依据)。

    判据 = config.car_config_path 的文件名（run_tracker 启动时按 --car 写进去的）。
    2026-08-15 之前的场次没有这个字段——那时只有 v0.3 一台车，按 v03 处理并在依据里写明。
    """
    path = Path(tracker_json)
    try:
        with path.open(encoding="utf-8") as fh:
            config = json.load(fh).get("config") or {}
    except Exception as exc:
        raise RuntimeError(f"读不了 {path} 的 config，无法推车型（{exc!r}）") from exc
    raw = config.get("car_config_path")
    if not raw:
        return "v03", f"{path.name} 无 config.car_config_path（0815 之前只有 v0.3 一台车）"
    name = Path(str(raw)).name
    car = CAR_BY_LAYOUT_CONFIG.get(name)
    if car is None:
        raise RuntimeError(
            f"{path.name} 的 car_config_path={name} 认不出车型；"
            f"已知：{CAR_BY_LAYOUT_CONFIG}。新车型要同时在这里和 run_tracker.CAR_LAYOUT_CONFIGS 登记。"
        )
    return car, f"{path.name} config.car_config_path={name}"


def car_for_session(arm: dict | None = None, tracker_json: Path | str | None = None,
                    explicit: str | None = None) -> tuple[str, str]:
    """一场数据用哪台车：显式 > arm JSON 自述 > tracker JSON 推断。三条都没有就抛。

    报告/离线分析都走这一份，免得各家自己猜出不同的车。
    """
    if explicit:
        if explicit not in KNOWN_CARS:
            raise ValueError(f"未知车型 {explicit!r}")
        return explicit, "显式指定"
    said = (arm or {}).get("car") if isinstance(arm, dict) else None
    if said:
        if said not in KNOWN_CARS:
            raise RuntimeError(f"arm JSON 自述车型 {said!r} 不认识")
        return said, f"arm JSON 自述（{(arm or {}).get('car_source', '来源未记')}）"
    if tracker_json:
        return car_for_tracker_json(tracker_json)
    raise RuntimeError("推不出车型：arm JSON 没有 car 字段，也没给 tracker JSON")


# 臂 C++ 节点名（/rosout 的 name 字段尾）：run_config 行只认它发的
ARM_NODE_NAME = "arm_controller_cpp"

EVENT_TOPICS = (
    "/tennis/status",
    "/tennis/arm_command",
    "/tennis/hit_pos",
    "/predict_hit_pos",
    "/arm_controller/status",
)

# RK 单调钟参考话题（按优先级）：payload 带发布时刻的单调钟 t，发布延迟接近 0。
# 仅旧 bag（epoch stamp / 无 t= 的 status）需要；新固件全部原生直读。
# 新合同 /predict_hit_pos 的 generated_at 接近发布时刻；历史 ct 是球观测时刻
# （比发布早一个 RK 管线时延，0716 实测 ~70ms）。后者只配当保底，用到时
# 在 clock_sync 里明示 biased。
MONO_REF_TOPICS = ("/bot_state", "/chassis_can/imu")

# 单调钟量级上限：RK 开机秒（连续运行数月也 <1e8）；epoch 秒 ~1.7e9。
MONO_MAX_SEC = 1e8

# status 文本尾缀发布时刻（arm_controller 单调钟版追加）："... t=9203.123456"
STATUS_T_RE = re.compile(r"\s+t=([0-9]+\.[0-9]+)$")

def _event_payload_time(topic: str, text: str) -> float | None:
    """Return the RK perf-counter timestamp carried by an event payload.

    ``final_hit_plan/v1`` and ``preaim_hint/v1`` use ``generated_at``: that is
    when the immutable payload exists and is published.  ``source_ct`` only
    timestamps the ball state consumed by the solve.  Legacy payloads use
    ``ct``.  A message that declares a
    ``kind`` never falls back to the legacy key: that would silently turn a
    malformed new contract into an apparently valid historical message.
    """
    if topic == "/predict_hit_pos":
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        key = "generated_at" if "kind" in payload else "ct"
        value = payload.get(key)
        return (
            float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            else None
        )
    return None


def _event_text(raw, msg) -> str:
    """Serialize a String/array event without imposing a report-side size cap."""
    if raw is None:
        return str(msg)
    if isinstance(raw, (list, tuple)) or type(raw).__name__ == "array":
        return " ".join(f"{float(v):.4g}" for v in raw)
    return str(raw)


def _ordered(values: list[float], names: list[str], slot_names: tuple) -> list[float | None]:
    """按控制器 6 个槽重排：slot_names[i] 是槽 i 的 ROS 关节名，None = 空槽（恒 0）。

    消息带名字就按名字取；不带名字按电机顺序（空槽不占位）。消息里这个字段整个没有（如 effort 为空）→ 全 None。
    """
    if not values:
        return [None] * len(slot_names)
    by_name = {name: idx for idx, name in enumerate(names)}
    ordered: list[float | None] = []
    motor = 0
    for name in slot_names:
        if name is None:
            ordered.append(0.0)
            continue
        src = by_name.get(name) if names else motor
        motor += 1
        ordered.append(float(values[src]) if src is not None and src < len(values) else None)
    return ordered


def check_joint_names(topic: str, seen: list[str], slot_names: tuple, car: str) -> None:
    """包里的关节名必须正好是本车型那一组。v0.4 的包按 v0.5 读，腕转 joint5 会被当成滚转，
    算出一条看似正常的错 TCP；反过来 v0.5 的包按 v0.4 读则整列缺 joint6。"""
    expected = sorted(name for name in slot_names if name is not None)
    if sorted(seen) != expected:
        raise SystemExit(f"{topic} 的关节名 {seen} 不是车型 {car} 的 {expected}——车型选错了？")


def _round_list(values: list[float | None], digits: int) -> list[float | None]:
    return [None if v is None else round(v, digits) for v in values]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _thirds_drift_ms_per_min(samples: list[tuple[float, float]]) -> dict | None:
    """(recv_s, diff_s) 样本按首/末三分之一中位差估漂移率。中位数抗离群，O(n)。"""
    if len(samples) < 30:
        return None
    samples = sorted(samples)
    span = samples[-1][0] - samples[0][0]
    if span < 10.0:
        return None
    third = span / 3.0
    lo = [d for t, d in samples if t <= samples[0][0] + third]
    hi = [d for t, d in samples if t >= samples[-1][0] - third]
    t_lo = statistics.median([t for t, _ in samples if t <= samples[0][0] + third])
    t_hi = statistics.median([t for t, _ in samples if t >= samples[-1][0] - third])
    if t_hi - t_lo < 1.0:
        return None
    rate = (statistics.median(hi) - statistics.median(lo)) / (t_hi - t_lo)
    return {
        "ms_per_min": round(rate * 60_000, 3),
        "span_s": round(span, 1),
        "n": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag 目录（含 metadata.yaml）")
    parser.add_argument("--output", type=Path, required=True, help="输出 arm JSON 路径")
    parser.add_argument(
        "--car",
        choices=CARS,
        help="车型（决定关节名进哪个控制器槽）。不给就从同名 tracker JSON 的 config.car_config_path 推",
    )
    args = parser.parse_args()

    # 车型：显式 > tracker JSON 推断 > 失败。默认成另一台车比报错危险得多（静默偏几厘米）。
    if args.car:
        car, car_source = args.car, "--car 显式指定"
    else:
        tracker_json = args.bag.with_name(args.bag.name[:-len("_rosbag")] + ".json") \
            if args.bag.name.endswith("_rosbag") else args.output.with_name(
                args.output.name.replace("_arm.json", ".json"))
        if not tracker_json.exists():
            parser.error(
                f"没给 --car，也找不到同场 tracker JSON（{tracker_json}）来推车型。"
                f"请补 --car {'/'.join(CARS)}。"
            )
        car, car_source = car_for_tracker_json(tracker_json)
    if car not in CARS:
        raise SystemExit(f"车型 {car}（{car_source}）已不在臂里（只有 {'/'.join(CARS)} 有车型 yaml），抽不了关节槽位")
    slot_names = slot_joint_names(car)
    print(f"[extract_arm_bag] 车型 {car}（依据：{car_source}）；槽位 {slot_names}")

    import rosbag2_py  # noqa: E402
    from rclpy.serialization import deserialize_message  # noqa: E402
    from rosidl_runtime_py.utilities import get_message  # noqa: E402

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
    link_commands: list[dict] = []
    events: list[dict] = []
    run_configs: list[dict] = []
    state_diffs: list[tuple[float, float]] = []    # (recv_s, stamp − recv) 全部有效 stamp
    command_diffs: list[tuple[float, float]] = []
    mono_diffs: dict[str, list[tuple[float, float]]] = {t: [] for t in MONO_REF_TOPICS}
    predict_event_diffs: list[tuple[float, float]] = []
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
        recv = timestamp / 1e9

        if topic == "/joint_states":
            msg = deserialize_message(data, msg_type)
            names = list(msg.name)
            if not seen_state_names and names:
                check_joint_names(topic, names, slot_names, car)
                seen_state_names = names
            positions = _ordered(list(msg.position), names, slot_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 0.0:
                state_diffs.append((recv, stamp - recv))
            states.append(
                {
                    "stamp": stamp,
                    "recv": recv,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(msg.velocity), names, slot_names), 5),
                    "effort": _round_list(_ordered(list(msg.effort), names, slot_names), 5),
                }
            )
        elif topic in ("/tennis/motor_command", "/tennis/link_command"):
            msg = deserialize_message(data, msg_type)
            if not msg.points:
                continue
            point = msg.points[0]
            names = list(msg.joint_names)
            if not seen_command_names and names:
                check_joint_names(topic, names, slot_names, car)
                seen_command_names = names
            positions = _ordered(list(point.positions), names, slot_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 0.0 and topic == "/tennis/motor_command":
                command_diffs.append((recv, stamp - recv))
            (commands if topic == "/tennis/motor_command" else link_commands).append(
                {
                    "stamp": stamp,
                    "recv": recv,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(point.velocities), names, slot_names), 5),
                    "effort": _round_list(_ordered(list(point.effort), names, slot_names), 5),
                }
            )
        elif topic == "/rosout":
            # 臂节点起栈 run_config 行（node/run_config.hpp：行尾 zero_frame= 等身份段）：只有臂起栈前就开录的包有
            msg = deserialize_message(data, msg_type)
            if str(msg.name).endswith(ARM_NODE_NAME) and str(msg.msg).startswith("run_config:"):
                run_configs.append({"recv": recv, "text": str(msg.msg)})
        elif topic in EVENT_TOPICS or topic in MONO_REF_TOPICS:
            msg = deserialize_message(data, msg_type)
            raw = msg.data if hasattr(msg, "data") else None
            if topic in MONO_REF_TOPICS:
                try:
                    payload = json.loads(raw)
                    t_mono = payload.get("t")
                    if isinstance(t_mono, (int, float)):
                        mono_diffs[topic].append((recv, float(t_mono) - recv))
                except Exception:
                    pass
                continue
            text = _event_text(raw, msg)
            event = {"recv": recv, "topic": topic, "text": text, "t_payload": None}
            if topic == "/predict_hit_pos":
                # 新合同 generated_at 是计划/提示的生成时刻；历史合同 ct 是观测时刻。
                payload_t = _event_payload_time(topic, text)
                if payload_t is not None:
                    predict_event_diffs.append((recv, payload_t - recv))
                    event["t_payload"] = payload_t
            else:
                # 新固件 status 尾缀 " t=<单调秒>" = 发布时刻，解析后从文本剥离
                m = STATUS_T_RE.search(event["text"])
                if m:
                    event["t_payload"] = float(m.group(1))
                    event["text"] = event["text"][: m.start()]
            events.append(event)

    if start_ns is None:
        raise RuntimeError(f"bag has no messages: {args.bag}")

    # ---- stamp 时钟域判定（按话题多数）----
    def _stamp_domain(rows: list[dict]) -> str | None:
        vals = [row["stamp"] for row in rows if row["stamp"] > 0.0]
        if not vals:
            return None
        mono_n = sum(1 for v in vals if v < MONO_MAX_SEC)
        return "rk_mono_native" if mono_n * 2 >= len(vals) else "rk_sys_converted_legacy"

    state_domain = _stamp_domain(states)
    command_domain = _stamp_domain(commands)
    # legacy = 旧固件的 epoch stamp；recv 映射 = 无自带时间的事件（PC 发的
    # hit_pos/arm_command、旧固件 status）。后者在新 bag 里也存在，不算 legacy。
    legacy_stamps = "rk_sys_converted_legacy" in (state_domain, command_domain)
    needs_c_mono = legacy_stamps or any(e["t_payload"] is None for e in events)

    # ---- RK 单调钟参考 C_mono = median(payload t − recv) ----
    mono_ref_topic = None
    mono_ref_biased = False
    c_mono = None
    if needs_c_mono:
        for topic in MONO_REF_TOPICS:
            if len(mono_diffs[topic]) >= 30:
                mono_ref_topic = topic
                c_mono = _median([d for _, d in mono_diffs[topic]])
                break
        if c_mono is None and len(predict_event_diffs) >= 10:
            # 新合同 generated_at 近似发布时刻；历史 ct 比发布早一段管线时延。
            mono_ref_topic = "/predict_hit_pos(generated_at/legacy ct)"
            mono_ref_biased = any(
                '"kind"' not in e["text"] for e in events if e["topic"] == "/predict_hit_pos"
            )
            c_mono = _median([d for _, d in predict_event_diffs])
        status_diffs = [e["t_payload"] - e["recv"] for e in events
                        if e["topic"] == "/tennis/status" and e["t_payload"] is not None]
        if c_mono is None and len(status_diffs) >= 10:
            # 专场空挥只发面板目标（无 generated_at）、bot 不跑：退回 status 尾缀 t=（RK 发布时刻，与 /bot_state 同口径）
            mono_ref_topic = "/tennis/status(t=)"
            c_mono = _median(status_diffs)
        if c_mono is None:
            raise RuntimeError(
                "no RK mono reference in bag (/bot_state, /chassis_can/imu or "
                "/predict_hit_pos payload) — cannot map stamp-less/legacy data onto the RK axis"
            )

    c_sys_js = _median([d for _, d in state_diffs])
    c_sys_mc = _median([d for _, d in command_diffs])

    # ---- 统一落到 RK 单调钟绝对秒 ----
    def _finish(rows: list[dict], c_sys: float | None) -> None:
        for row in rows:
            stamp = row.pop("stamp")
            recv = row.pop("recv")
            if 0.0 < stamp < MONO_MAX_SEC:
                row["t"] = round(stamp, 5)                    # 新固件：原生单调钟直读
            elif stamp >= MONO_MAX_SEC and c_sys is not None and c_mono is not None:
                row["t"] = round(stamp - c_sys + c_mono, 5)   # 旧 bag：epoch → 单调钟
            elif c_mono is not None:
                row["t"] = round(recv + c_mono, 5)            # 缺 stamp：接收时刻映射
            else:
                row["t"] = round(recv, 5)                     # 不可达（needs_legacy 已保证 c_mono）

    _finish(states, c_sys_js)
    _finish(commands, c_sys_mc)
    _finish(link_commands, c_sys_mc)
    status_payload_n = 0
    for e in events:
        if e["t_payload"] is not None:
            e["t"] = round(e["t_payload"], 5)
            status_payload_n += 1
        elif c_mono is not None:
            e["t"] = round(e["recv"] + c_mono, 5)
        else:
            e["t"] = round(e["recv"], 5)
    # dict 键序整理：t 放行首，便于肉眼查文件
    states = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort")}} for r in states]
    commands = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort")}} for r in commands]
    link_commands = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort")}} for r in link_commands]
    events = [{"t": e["t"], "topic": e["topic"], "text": e["text"]} for e in events]

    # ---- ht 锚点残差：新调度触球≡ht，done(=accepted.t+duration) − ht 应 ≈0 ----
    # 全链自检：stamp 域 + status 发布时刻 + predict ct 任一有偏都会体现在这里。
    predicts: list[dict] = []
    for e in events:
        if e["topic"] != "/predict_hit_pos":
            continue
        try:
            p = json.loads(e["text"])
            predicts.append({"t": e["t"], "duration": p.get("duration"),
                             "rel_x": p.get("rel_x"), "ht": p.get("ht")})
        except Exception:
            pass
    ht_residuals: list[float] = []
    acc_re = re.compile(r"^accepted hit x=([\-0-9.]+) z=[\-0-9.]+ duration=([0-9.]+)")
    for e in events:
        if e["topic"] != "/tennis/status":
            continue
        m = acc_re.match(e["text"])
        if not m:
            continue
        x, dur = float(m.group(1)), float(m.group(2))
        for p in reversed(predicts):
            if p["t"] > e["t"]:
                continue
            if e["t"] - p["t"] > 0.35:
                break
            d_ok = isinstance(p["duration"], (int, float)) and abs(p["duration"] - dur) < 2e-3
            x_ok = isinstance(p["rel_x"], (int, float)) and abs(p["rel_x"] - x) < 5e-4
            if (d_ok or x_ok) and isinstance(p["ht"], (int, float)):
                ht_residuals.append((e["t"] + dur) - p["ht"])
                break

    mono_drift_samples = (
        mono_diffs[mono_ref_topic] if mono_ref_topic in mono_diffs
        else (state_diffs if state_domain == "rk_mono_native" else predict_event_diffs)
    )
    clock_sync = {
        "joint_states_stamp_domain": state_domain,
        "motor_command_stamp_domain": command_domain,
        "events_with_payload_t": status_payload_n,
        "events_total": len(events),
        "legacy_stamps": legacy_stamps,
        "mono_ref_topic": mono_ref_topic,
        "mono_ref_biased": mono_ref_biased,
        "mono_minus_recv_median_s": None if c_mono is None else round(c_mono, 6),
        "joint_states_stamp_minus_recv_median_s": None if c_sys_js is None else round(c_sys_js, 6),
        "motor_command_stamp_minus_recv_median_s": None if c_sys_mc is None else round(c_sys_mc, 6),
        # 漂移自检：该钟 − PC 收钟的速率（原生模式下 joint_states stamp 本身就是单调钟）
        "mono_vs_pc_drift": _thirds_drift_ms_per_min(mono_drift_samples),
        "joint_states_stamp_vs_pc_drift": _thirds_drift_ms_per_min(state_diffs),
        "ht_anchor_residual_ms": (
            None if not ht_residuals
            else round(statistics.median(ht_residuals) * 1000, 1)
        ),
        "ht_anchor_n": len(ht_residuals),
    }

    result = {
        "schema": "tracker_arm_bag_v3",
        "time_axis": "rk_mono_abs_s",
        "time_sources": {
            "states": (
                "header.stamp（RK 单调钟原生）" if state_domain == "rk_mono_native"
                else "header.stamp(RK系统钟) − median(stamp−recv) + median(bot_state.t−recv)（旧 bag 兼容）"
            ),
            "commands": (
                "header.stamp（RK 单调钟原生）" if command_domain == "rk_mono_native"
                else "同 states 换算（用 motor_command 自己的 stamp−recv 中位）"
            ),
            "events": "status 尾缀 t= / 新计划 generated_at / 历史 predict ct（原生）；无时间事件 = recv + median(bot_state.t−recv)",
            "note": "与 /predict_hit_pos source_ct/contact_ht（历史 ct/ht）、rk_tracking payload t 同钟；报告端固定 scale=1，只用 RK→PC 常数偏移",
        },
        "clock_sync": clock_sync,
        "bag_dir": str(args.bag.resolve()),
        # 车型：报告端按它取生产车型（端点模型 / 方案判决 / 槽位复核），不猜；car_source 留着事后查是谁定的。
        "car": car,
        "car_source": car_source,
        "start_ns": start_ns,
        "duration_sec": round((end_ns - start_ns) / 1e9, 4),
        "joint_names": [EMPTY_SLOT_LABEL if name is None else name for name in slot_names],
        "state_joint_names_raw": seen_state_names,
        "command_joint_names_raw": seen_command_names,
        "topics": [
            {"name": name, "type": type_by_topic.get(name, ""), "count": counts[name]}
            for name in sorted(counts)
        ],
        "states": states,
        "commands": commands,
        "link_commands": link_commands,
        "events": events,
        "run_configs": run_configs,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        "arm json saved: %s (states=%d commands=%d link_commands=%d events=%d duration=%.1fs)"
        % (args.output, len(states), len(commands), len(link_commands), len(events), result["duration_sec"])
    )
    print("state joint names: %s" % (seen_state_names,))
    print("command joint names: %s" % (seen_command_names,))
    print("clock_sync: %s" % json.dumps(clock_sync, ensure_ascii=False))
    if legacy_stamps:
        print("NOTE: legacy epoch stamps converted (old firmware bag) — upgrade RK side for native mono stamps")
    d_mono = clock_sync["mono_vs_pc_drift"]
    if d_mono and abs(d_mono["ms_per_min"]) > 2.0:
        print("WARNING: RK mono vs PC drift %.2f ms/min > 2 — single-offset assumption needs review"
              % d_mono["ms_per_min"])
    if clock_sync["ht_anchor_residual_ms"] is not None and abs(clock_sync["ht_anchor_residual_ms"]) > 20.0:
        print("WARNING: |median(done - ht)| = %.1f ms — check clock chain or scheduler execution"
              % clock_sync["ht_anchor_residual_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
