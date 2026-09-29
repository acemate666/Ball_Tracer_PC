# -*- coding: utf-8 -*-
"""臂 FK 的车型合同：v0.3 / v0.4 / v0.5 是三台不同的臂，报告必须按本场车型算 TCP。

黄金向量取自臂端自己导出的资产（arm_controller `tools/export_cpp_assets.py` →
`cpp/arm_controller_cpp/assets/<car>/test_vectors.json` 的 ik_hit / face_lookup 组）：
每条是「臂端 IK 解出的位形 q」+「它本该把拍心放到的 (x, z)」。我们这边的 FK 独立算一遍，
必须落回同一个点——这才是跨实现对账，而不是拿本文件的常数自证。
v0.3 已随车从 arm_controller 删除（0923，连同 assets/v03），它的链是 extract_arm_bag 里冻结的抄本，
不再有臂端黄金向量；只作「拿错车」的对照。

⚠ 这组测试守的是 0816 那次事故：报告端 FK 写死 v0.3 链，v0.4 场次整场 TCP 偏
(−5.4, −8.6)cm、FK 拍速低报 5%，而拍面 yaw/pitch 两车逐拍恒等——页面上没有任何征兆。
所以既要验「各车自己对」，也要验「拿错车一定偏得看得见」，还要验「车型推断/缺省不猜」。
v0.5（五轴臂）另有一条：/joint_states 只有 5 个名字，按名字进控制器的 6 个槽，槽 4 空。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import extract_arm_bag as eab  # noqa: E402

# 黄金向量直接读臂端标准 checkout 导出的资产（cpp/arm_controller_cpp/assets/<car>/test_vectors.json，
# ik_hit / face_lookup 两组）：每条 = (x, z, q6)，臂端声称该位形把拍心放在 (x, y≈0, z)。
# 不再逐条抄进来——2026-09-05 用户定：报告端只认那一份标准文件，改标定不用来这里同步常数。
ASSETS_ROOT = eab.ARM_CONTROLLER_ROOT / "cpp" / "arm_controller_cpp" / "assets"


def _golden(car: str) -> list[tuple[float, float, list[float]]]:
    vectors = json.loads((ASSETS_ROOT / car / "test_vectors.json").read_text(encoding="utf-8"))
    rows = [(r["x"], r["z"], list(r["q5"]) + [0.0]) for r in vectors["ik_hit"]]
    rows += [(r["x"], r["z"], list(r["q"])) for r in vectors["face_lookup"]]
    assert rows, f"{car}: test_vectors.json 没有 ik_hit/face_lookup 向量"
    return rows


GOLDEN = {"v04": _golden("v04"), "v05": _golden("v05")}


@pytest.mark.parametrize("car", sorted(GOLDEN))
def test_tcp_distance_matches_current_calibration(car):
    cak = eab.standard_kinematics(car)
    yaml_tool_x = cak.read_car_kinematics(car)["tool_x"]
    j6_x = float(cak.JOINTS[-1]["local0"][0, 3])
    assert eab.CAR_MODELS[car].tcp_distance == pytest.approx(cak.TCP_DISTANCE, abs=1e-12)
    assert cak.TCP_DISTANCE == pytest.approx(float(yaml_tool_x) - j6_x, abs=1e-12)  # 真值 = yaml tool_x


@pytest.mark.parametrize("car", sorted(GOLDEN))
def test_fk_matches_arm_side_golden_vectors(car):
    """各车 FK 落回臂端 IK 声称的击球点（含 y≈0：拍心该落在击球平面上）。

    五轴臂（v0.5，没有腕转）的拍面 yaw 只能由 J1 给，臂端 IK 约束的是 J1 系里的 (x, z)
    （J1 轴在世界原点），世界 y 随 J1 转走、不约束——见 compact_arm_kinematics.ik_hit_face。
    """
    pytest.importorskip("numpy")
    eab.use_car(car)
    five_axis = not eab.standard_kinematics(car).HAS_WRIST_YAW
    for x, z, q in GOLDEN[car]:
        tcp = eab.fk(q)["tcp"]
        if five_axis:
            c, s = math.cos(q[0]), math.sin(q[0])
            tcp = [c * tcp[0] + s * tcp[1], -s * tcp[0] + c * tcp[1], tcp[2]]
        assert tcp[0] == pytest.approx(x, abs=1e-3), f"{car} x @({x},{z})"
        assert tcp[2] == pytest.approx(z, abs=1e-3), f"{car} z @({x},{z})"
        assert abs(tcp[1]) < 0.01, f"{car} 拍心不在击球平面上 @({x},{z})"


@pytest.mark.parametrize("car", sorted(GOLDEN))
def test_fk_matches_standard_module(car):
    """本文件的 fk 与臂端标准模块自己的 fk 是同一条链（关节表、基座变换、工具轴、甜点距离都是搬来的）。"""
    np = pytest.importorskip("numpy")
    cak = eab.standard_kinematics(car)
    rng = np.random.default_rng(5)
    for _ in range(50):
        q = rng.uniform(-1.5, 1.5, 6)
        if not cak.HAS_WRIST_YAW:
            q[4] = 0.0  # 空槽恒 0
        ours, theirs = eab.fk(q, car=car), cak.fk(q)
        assert np.allclose(ours["tcp"], theirs["tcp"], atol=1e-12)
        assert np.allclose(ours["handle_axis"], theirs["handle_axis"], atol=1e-12)


def test_wrong_car_is_off_by_centimetres_not_millimetres():
    """拿错车不是小数点级误差：当前 v0.4 黄金位形用 v0.3 链算会低约 12cm。"""
    pytest.importorskip("numpy")
    for x, z, q in GOLDEN["v04"]:
        good = eab.fk(q, car="v04")["tcp"]
        wrong = eab.fk(q, car="v03")["tcp"]
        assert wrong[2] - good[2] < -0.05, "v0.3 链算 v0.4 位形应显著偏低"
        assert math.hypot(wrong[0] - good[0], wrong[2] - good[2]) > 0.08


def test_face_angles_are_identical_across_cars():
    """拍面法向在 v0.3 / v0.4 下逐条恒等——所以角度列看不出选错车，位置列才是判据。
    （0816 实测 11 拍逐拍相同；这条固化那个观察，免得有人拿角度列去"验证"车型。）"""
    pytest.importorskip("numpy")
    for _, _, q in GOLDEN["v04"]:
        n3 = eab.fk(q, car="v03")["face_normal"]
        n4 = eab.fk(q, car="v04")["face_normal"]
        assert all(abs(a - b) < 1e-6 for a, b in zip(n3, n4))


def test_no_default_car():
    """一次都没选过车型就调 fk() 必须抛——静默用另一台车的链是这套代码最贵的错误。"""
    pytest.importorskip("numpy")
    eab._ACTIVE = None
    with pytest.raises(RuntimeError, match="还没选车型"):
        eab.fk([0.0] * 6)
    eab.use_car("v04")  # 复位，免得影响同进程其它用例


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
    """run_tracker 能启动的每一种 --car，这里都认得出它落下的布局文件，而且都有 FK 链。"""
    import re

    source = (SRC_DIR.parent / "src" / "run_tracker.py").read_text(encoding="utf-8")
    block = re.search(r"CAR_LAYOUT_CONFIGS = \{(.*?)\}", source, re.S).group(1)
    layouts = dict(re.findall(r'"(v\d+)":\s*"([^"]+)"', block))
    assert layouts, "没从 run_tracker.py 读到 CAR_LAYOUT_CONFIGS"
    for car, layout in layouts.items():
        assert eab.CAR_BY_LAYOUT_CONFIG.get(layout) == car, f"{layout} 没登记成 {car}"
        assert car in eab.CAR_MODELS, f"{car} 没有 FK 链"


def test_slot_joint_names_follow_arm_car_yaml():
    """各槽的 ROS 关节名 = 臂车型 yaml 的 motors（控制器发 /joint_states 用的就是它们）。
    v0.5 五轴臂 5 个电机进槽 0,1,2,3,5：名叫 joint5 的是拍柄滚转（槽 5），槽 4（腕转）空。"""
    six = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    assert eab.CAR_MODELS["v03"].slot_joint_names == six
    assert eab.CAR_MODELS["v04"].slot_joint_names == six
    assert eab.CAR_MODELS["v05"].slot_joint_names == ("joint1", "joint2", "joint3", "joint4", None, "joint5")


def test_ordered_puts_v05_roll_in_slot5():
    """v0.5 的消息只有 5 个名字：按名字进槽、槽 4 补 0；不带名字按电机顺序；字段整个缺失给全 None。"""
    slots = eab.CAR_MODELS["v05"].slot_joint_names
    names = ["joint5", "joint1", "joint2", "joint3", "joint4"]   # 顺序无所谓，按名字取
    assert eab._ordered([0.5, 0.1, 0.2, 0.3, 0.4], names, slots) == [0.1, 0.2, 0.3, 0.4, 0.0, 0.5]
    assert eab._ordered([0.1, 0.2, 0.3, 0.4, 0.5], [], slots) == [0.1, 0.2, 0.3, 0.4, 0.0, 0.5]
    assert eab._ordered([], names, slots) == [None] * 6
    six = eab.CAR_MODELS["v04"].slot_joint_names
    assert eab._ordered([1, 2, 3, 4, 5, 6], [], six) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert eab._ordered([1, 2, 3, 4, 5], ["joint1", "joint2", "joint3", "joint4", "joint5"], six)[5] is None


def test_joint_names_must_match_car():
    """v0.4 的包按 v0.5 读，腕转 joint5 会被当成滚转、算出一条看似正常的错 TCP——名字对不上必须直接失败。"""
    v04_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    v05_names = v04_names[:5]
    eab.check_joint_names("/joint_states", v05_names, eab.CAR_MODELS["v05"].slot_joint_names, "v05")
    eab.check_joint_names("/joint_states", v04_names, eab.CAR_MODELS["v04"].slot_joint_names, "v04")
    with pytest.raises(SystemExit, match="车型选错"):
        eab.check_joint_names("/joint_states", v04_names, eab.CAR_MODELS["v05"].slot_joint_names, "v05")
    with pytest.raises(SystemExit, match="车型选错"):
        eab.check_joint_names("/joint_states", v05_names, eab.CAR_MODELS["v04"].slot_joint_names, "v04")


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


def test_recompute_tcp_rewrites_derived_field():
    """arm JSON 的 tcp 只是派生量：换车重算即可，不必重跑 rosbag 提取。
    关节残缺的行置 None（报告列显示 —），不能留着别的车算出来的旧值。"""
    pytest.importorskip("numpy")
    q = [0.1, 0.2, 1.4, 1.6, 0.0, 0.3]
    rows = [{"position": q, "tcp": [9.0, 9.0, 9.0]},
            {"position": [0.1, None, 1.4, 1.6, 0.0, 0.3], "tcp": [9.0, 9.0, 9.0]}]
    eab.use_car("v04")
    assert eab.recompute_tcp(rows) == 2
    assert rows[0]["tcp"] == [round(float(v), 4) for v in eab.fk(q)["tcp"]]
    assert rows[1]["tcp"] is None


def test_extractor_records_car_in_output():
    """出 arm JSON 必须把车型写进去（car/car_source/fk_source），报告端才不用再猜。"""
    source = (SRC_DIR / "extract_arm_bag.py").read_text(encoding="utf-8")
    assert '"car": car,' in source
    assert '"car_source": car_source,' in source
    assert '"fk_source": f"extract_arm_bag.fk({car})",' in source
    assert '"--car"' in source and "choices=sorted(CAR_MODELS)" in source


def test_run_tracker_passes_car_to_extractor():
    """run_tracker 把启动时选的车型透给 extract_arm_bag（--car-config 直给时留空、由 JSON 推）。"""
    source = (SRC_DIR.parent / "src" / "run_tracker.py").read_text(encoding="utf-8")
    assert 'arm_command.extend(["--car", car])' in source
    assert "car=args.car," in source
