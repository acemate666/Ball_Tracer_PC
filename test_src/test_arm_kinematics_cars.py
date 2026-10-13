# -*- coding: utf-8 -*-
"""车型合同 + 报告调臂生产代码的胶水（test_src/arm_production.py）。

2026-10-13 起报告不再自带臂 FK（用户「report 也只用一份代码，做 fk 或者其他相关的时候，都去引用现在的生产代码」）：
黑标 / 甜点 / 拍面 / 拍速 = 臂端点模型（arm_controller.arm_endpoint → C++ control/arm_endpoint.hpp），方案收不收 = 臂
node::receive_hit_pos（arm_controller.hit_pos）。臂自己的黄金向量 / 对拍在臂仓库里测，这里不再复刻。这里守：
  1. 车型推断不猜——0816 事故：报告端 FK 写死 v0.3 链，v0.4 场次整场 TCP 偏 (−5.4, −8.6) cm、页面上没有任何征兆；
  2. 抽取脚本（ROS2 解释器里调不了臂 C 接口库）按臂槽位规则排的 /joint_states 槽 = 臂生产代码 ArmEndpoint.slot_names；
  3. 胶水：录包零位帧（run_config 行 > 登记表）、判不了时抛 Unavailable 带原因、端点量逐值 = 生产 eval_series / pose；
  4. 报告链源码里没有臂公式的第二份实现。
要臂 checkout（TENNIS_MAN_ARM_CONTROLLER）与 g++ 的用例在缺时 skip。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import arm_production as ap  # noqa: E402
import extract_arm_bag as eab  # noqa: E402

V05_SLOTS = ("joint1", "joint2", "joint3", "joint4", None, "joint5")
SIX_SLOTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _production():
    try:
        ap.endpoint("v05")
    except ap.Unavailable as exc:
        pytest.skip(f"臂生产代码不可用：{exc}")
    return ap.arm()


def test_car_for_tracker_json_reads_layout_config(tmp_path):
    """车型来源 = tracker JSON 里 run_tracker 按 --car 落下的 car_config_path。
    0815 之前没这个字段（当时只有 v0.3 一台车）→ v03；认不出的布局文件必须抛，不能猜。"""
    def write(name, config):
        p = tmp_path / name
        p.write_text(json.dumps({"config": config}), encoding="utf-8")
        return p

    car, why = eab.car_for_tracker_json(
        write("v04.json", {"car_config_path": r"D:\Ball_Tracer_PC\src\config\vehicle_v04.json"}))
    assert car == "v04" and "vehicle_v04.json" in why
    car, why = eab.car_for_tracker_json(
        write("v03.json", {"car_config_path": "/x/arm_poe_racket_center.json"}))
    assert car == "v03"
    car, why = eab.car_for_tracker_json(write("old.json", {"fps": 29.0}))
    assert car == "v03" and "0815" in why
    car, why = eab.car_for_tracker_json(
        write("v05.json", {"car_config_path": r"D:\robot\Ball_Tracer_PC\src\config\vehicle_v05.json"}))
    assert car == "v05" and "vehicle_v05.json" in why
    with pytest.raises(RuntimeError, match="认不出车型"):
        eab.car_for_tracker_json(write("new.json", {"car_config_path": "vehicle_v99.json"}))


def test_layout_configs_cover_every_tracker_car():
    """run_tracker 能启动的每一种 --car，这里都认得出它落下的布局文件；有臂车型 yaml 的现役车都在其中。"""
    import re

    source = (SRC_DIR.parent / "src" / "run_tracker.py").read_text(encoding="utf-8")
    block = re.search(r"CAR_LAYOUT_CONFIGS = \{(.*?)\}", source, re.S).group(1)
    layouts = dict(re.findall(r'"(v\d+)":\s*"([^"]+)"', block))
    assert layouts, "没从 run_tracker.py 读到 CAR_LAYOUT_CONFIGS"
    for car, layout in layouts.items():
        assert eab.CAR_BY_LAYOUT_CONFIG.get(layout) == car, f"{layout} 没登记成 {car}"
        assert car in eab.KNOWN_CARS
    assert set(eab.CARS) <= set(layouts)


@pytest.mark.parametrize("car", ["v04", "v05"])
def test_extractor_slots_are_the_production_slots(car):
    """抽取在 ROS2 解释器里按臂槽位规则排槽（调不了臂 C 库）；必须与臂生产代码 ArmEndpoint.slot_names 逐槽一致。"""
    _production()
    assert eab.slot_joint_names(car) == ap.slot_names(car)
    assert ap.slot_names(car) == {"v04": SIX_SLOTS, "v05": V05_SLOTS}[car]


def test_ordered_puts_v05_roll_in_slot5():
    """v0.5 的消息只有 5 个名字：按名字进槽、槽 4 补 0；不带名字按电机顺序；字段整个缺失给全 None。"""
    names = ["joint5", "joint1", "joint2", "joint3", "joint4"]   # 顺序无所谓，按名字取
    assert eab._ordered([0.5, 0.1, 0.2, 0.3, 0.4], names, V05_SLOTS) == [0.1, 0.2, 0.3, 0.4, 0.0, 0.5]
    assert eab._ordered([0.1, 0.2, 0.3, 0.4, 0.5], [], V05_SLOTS) == [0.1, 0.2, 0.3, 0.4, 0.0, 0.5]
    assert eab._ordered([], names, V05_SLOTS) == [None] * 6
    assert eab._ordered([1, 2, 3, 4, 5, 6], [], SIX_SLOTS) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert eab._ordered([1, 2, 3, 4, 5], ["joint1", "joint2", "joint3", "joint4", "joint5"], SIX_SLOTS)[5] is None


def test_joint_names_must_match_car():
    """v0.4 的包按 v0.5 读，腕转 joint5 会被当成滚转、算出一条看似正常的错 TCP——名字对不上必须直接失败。"""
    v04_names = list(SIX_SLOTS)
    v05_names = v04_names[:5]
    eab.check_joint_names("/joint_states", v05_names, V05_SLOTS, "v05")
    eab.check_joint_names("/joint_states", v04_names, SIX_SLOTS, "v04")
    with pytest.raises(SystemExit, match="车型选错"):
        eab.check_joint_names("/joint_states", v04_names, V05_SLOTS, "v05")
    with pytest.raises(SystemExit, match="车型选错"):
        eab.check_joint_names("/joint_states", v05_names, SIX_SLOTS, "v04")


def test_car_for_session_precedence(tmp_path):
    """显式 > arm JSON 自述 > tracker JSON；三条都没有就抛。"""
    tracker = tmp_path / "t.json"
    tracker.write_text(json.dumps({"config": {"car_config_path": "vehicle_v04.json"}}),
                       encoding="utf-8")
    assert eab.car_for_session({"car": "v03"}, tracker, explicit="v04")[0] == "v04"
    assert eab.car_for_session({"car": "v03"}, tracker)[0] == "v03"
    assert eab.car_for_session({}, tracker)[0] == "v04"
    with pytest.raises(RuntimeError, match="推不出车型"):
        eab.car_for_session({}, None)


def test_extractor_records_car_and_computes_nothing():
    """出 arm JSON 必须把车型与臂起栈 run_config 行写进去，报告端才不用再猜；臂派生量一个都不在抽取里算。"""
    source = (SRC_DIR / "extract_arm_bag.py").read_text(encoding="utf-8")
    assert '"car": car,' in source
    assert '"car_source": car_source,' in source
    assert '"run_configs": run_configs,' in source
    assert '"--car"' in source and "choices=CARS" in source
    for gone in ("def fk(", "compact_arm_kinematics", "standard_kinematics", "J1_TILT", "recompute_tcp", "use_car(",
                 "face_normal", "fk_source"):
        assert gone not in source, gone


def test_run_tracker_passes_car_to_extractor():
    """run_tracker 把启动时选的车型透给 extract_arm_bag（--car-config 直给时留空、由 JSON 推）。"""
    source = (SRC_DIR.parent / "src" / "run_tracker.py").read_text(encoding="utf-8")
    assert 'arm_command.extend(["--car", car])' in source
    assert "car=args.car," in source


def test_report_chain_has_no_second_arm_implementation():
    """报告链（抽取 / 生成页面 / 黑标测量 / 在线黑标 / swing_hs 几何）不再有臂 FK、J1 倾斜、拍头模型的第二份实现。"""
    files = [SRC_DIR / "extract_arm_bag.py", SRC_DIR / "extract_rk_tracking_bag.py", SRC_DIR / "generate_curve3_html.py",
             SRC_DIR / "racket_ht_black_marker.py", SRC_DIR.parent / "src" / "racket_marker_online.py",
             SRC_DIR.parent / "swing_hs" / "geometry.py", SRC_DIR.parent / "swing_hs" / "camera_proc.py"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for gone in ("compact_arm_kinematics", "standard_kinematics", "eab.fk(", "J1_TILT", "_frozen_head_model",
                     "CAR_MODELS", "tcp_distance"):
            assert gone not in text, (path.name, gone)


# ───────────────────────── 胶水：零位帧 / 判不了 / 逐值 = 生产 ─────────────────────────

def _run_config(frame: str) -> dict:
    return {"recv": 0.0, "text": f"run_config: car=v05 config=x | zero_frame={frame} offset_rad=[0,0,0,0,0,0] "
                                 "direction=[1,-1,1,-1,0,-1] face_table_fnv1a64=0 arm_calibration_fingerprint=0"}


def test_recorded_frame_prefers_the_run_config_line_then_the_registry():
    m = _production()
    current = m.endpoint.zero_frame("v05")[0]
    assert ap.recorded_frame("v05", "whatever", [_run_config(current)])[0] == current
    # 登记表里的录包名（臂 config/zero_frames.yaml）
    assert ap.recorded_frame("v05", "tracker_20261013_000109", None) == (
        "8344c70", "臂零位帧登记表（config/zero_frames.yaml 录包名 tracker_20261013_000109）")
    with pytest.raises(ap.Unavailable, match="未登记"):
        ap.recorded_frame("v05", "tracker_19990101_000000", [])
    with pytest.raises(ap.Unavailable, match="unregistered"):
        ap.recorded_frame("v05", "x", [{"recv": 0.0, "text": "run_config: zero_frame=unregistered -- the car yaml"}])
    with pytest.raises(ap.Unavailable, match="零位帧不同"):
        ap.recorded_frame("v05", "x", [_run_config(current), _run_config("8344c70")])


def _synthetic_arm(n: int = 300, frame: str | None = None) -> dict:
    """100 Hz 的五轴 /joint_states（面板直驱节拍：每拍五轴全发）+ 同刻命令；J1 在转、J2/J3 有力矩。"""
    t = 50.0 + 0.01 * np.arange(n)
    q = np.zeros((n, 6))
    q[:, 0] = -0.8 + 1.6 * (t - t[0]) / (t[-1] - t[0])
    q[:, 1], q[:, 2], q[:, 3], q[:, 5] = -0.2, 1.9, 1.7, 0.05
    qd = np.zeros((n, 6))
    qd[:, 0] = 1.6 / (t[-1] - t[0])
    e = np.zeros((n, 6))
    e[:, 0] = 25.0 * np.sin(4.0 * (t - t[0]))
    e[:, 1], e[:, 2], e[:, 3] = -30.0, 8.0, 2.0
    rows = [{"t": float(t[i]), "position": q[i].tolist(), "velocity": qd[i].tolist(), "effort": e[i].tolist()}
            for i in range(n)]
    arm = {"car": "v05", "states": rows, "commands": [dict(r) for r in rows], "events": [],
           "state_joint_names_raw": ["joint1", "joint2", "joint3", "joint4", "joint5"]}
    if frame:
        arm["run_configs"] = [_run_config(frame)]
    return arm


def test_endpoint_for_arm_is_the_production_model_value_for_value():
    """实测行 = 生产 eval_series（逐关节采样时刻同端点标定读包工具），命令行 = 生产刚体位形 pose；录包那一帧的读数先平移到当前帧。"""
    m = _production()
    arm = _synthetic_arm(frame="8344c70")
    info = ap.endpoint_for_arm(arm, "v05", "x")
    assert info["status"] == "ok" and info["frame"]["recorded"] == "8344c70"
    shift = m.endpoint.zero_shift("v05", "8344c70")
    assert info["frame"]["shift_rad"] == pytest.approx(shift.tolist(), abs=1e-9)
    assert abs(shift[0]) > 1e-3   # 8344c70 → 当前帧 J1 零位真的动过：这条用例确实在验平移

    states = arm["states"]
    t = np.array([s["t"] for s in states])
    q = np.array([s["position"] for s in states])
    e = np.array([s["effort"] for s in states])
    slots = list(m.bag.SLOTS)
    per = m.bag.joint_samples(m.bag.rows_of(t, q[:, slots], e[:, slots]), "v05", "8344c70")
    ep = ap.endpoint("v05")
    out = ep.eval_series([p["t"] for p in per], [p["q"] for p in per], [p["e"] for p in per], t)
    f = m.endpoint.field
    for name, key, digits in (("marker", "marker", 1e-5), ("sweet", "sweet", 1e-5), ("sweet_vel", "sweet_vel", 1e-4)):
        got = np.array([s[key] for s in states])
        assert got == pytest.approx(f(out, name), abs=digits), key
    assert [s["marker_h"] for s in states] == pytest.approx(f(out, "marker_height"), abs=1e-5)
    assert [s["face_yaw"] for s in states] == pytest.approx(np.degrees(f(out, "face_yaw")), abs=1e-3)
    assert [s["face_pitch"] for s in states] == pytest.approx(np.degrees(f(out, "face_pitch")), abs=1e-3)
    # 力矩非零 + 在动：动态项（柔度 / 间隙 / 时延 / 力矩轴倾斜）确实把黑标从刚体位形挪开了
    rigid = f(ep.pose(q + shift), "marker", m.endpoint.POSE)
    assert np.max(np.linalg.norm(np.array([s["marker"] for s in states]) - rigid, axis=1)) > 1e-3
    # 命令行 = 平移后的刚体位形
    sweet_cmd = f(ep.pose(q + shift), "sweet", m.endpoint.POSE)
    assert np.array([c["sweet"] for c in arm["commands"]]) == pytest.approx(sweet_cmd, abs=1e-5)
    assert all(math.isfinite(c["j1_speed"]) and c["j1_speed"] > 0 for c in arm["commands"])


def test_endpoint_for_arm_refuses_what_production_cannot_compute():
    _production()
    v04 = _synthetic_arm(frame="8344c70")
    with pytest.raises(ap.Unavailable, match="五轴"):
        ap.endpoint_for_arm(v04, "v04", "x")
    wrong_names = _synthetic_arm(frame="8344c70")
    wrong_names["state_joint_names_raw"] = list(SIX_SLOTS)
    with pytest.raises(ap.Unavailable, match="关节名"):
        ap.endpoint_for_arm(wrong_names, "v05", "x")
    with pytest.raises(ap.Unavailable, match="未登记"):
        ap.endpoint_for_arm(_synthetic_arm(), "v05", "tracker_19990101_000000")
    with pytest.raises(ap.Unavailable, match="连续"):
        ap.endpoint_for_arm(_synthetic_arm(n=50, frame="8344c70"), "v05", "x")
    # 两相调度认不出（5 ms 均匀节拍不是 two_phase_j1_j4）：读包工具的原话带出来
    odd = _synthetic_arm(frame="8344c70")
    for i, s in enumerate(odd["states"]):
        s["t"] = 50.0 + 0.005 * i
    with pytest.raises(ap.Unavailable, match="调度"):
        ap.endpoint_for_arm(odd, "v05", "x")
