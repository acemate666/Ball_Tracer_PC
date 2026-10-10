# -*- coding: utf-8 -*-
"""swing_hs 进程间的本机 UDP 消息（只用标准库：ROS 桥跑在 ROS 自带的 pixi python 里，也直接 import 本文件）。

  ROS 桥 → 相机进程 / 主进程：/joint_states 一条一包 {"s": header.stamp(RK 秒), "r": 桥收到时刻(PC perf), "n": 关节名, "p": 角度}
  主进程 → ROS 桥：发布请求 {"topic": 话题, "data": String 内容}
  ROS 桥 → 主进程：状态 {"kind": "bridge", ...}
"""

from __future__ import annotations

import json


def encode_joint_state(stamp_rk: float, recv_pc: float, names, positions) -> bytes:
    return json.dumps({"s": float(stamp_rk), "r": float(recv_pc), "n": list(names), "p": [float(v) for v in positions]},
                      separators=(",", ":")).encode("utf-8")


def decode_joint_state(data: bytes) -> tuple[float, float, list, list]:
    doc = json.loads(data.decode("utf-8"))
    return float(doc["s"]), float(doc["r"]), list(doc["n"]), [float(v) for v in doc["p"]]


def encode_publish(topic: str, payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps({"topic": topic, "data": data}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_publish(data: bytes) -> tuple[str, str]:
    doc = json.loads(data.decode("utf-8"))
    return str(doc["topic"]), str(doc["data"])


def encode_status(doc: dict) -> bytes:
    return json.dumps({"kind": "bridge", **doc}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_any(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))
