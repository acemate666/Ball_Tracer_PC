# -*- coding: utf-8 -*-
"""PC 报告 / 离线工具调臂生产代码的唯一入口（2026-10-13 用户「请你把 report 也修了把，也只用一份代码，做 fk 或者其他相关的时候，
都去引用现在的生产代码，这样生产代码变了，report 使用也是一样的」）。

本仓库不写臂的公式、判据与参数，全部经臂仓库（环境变量 TENNIS_MAN_ARM_CONTROLLER，缺省 D:/tennis-man/arm_controller）的 Python
包装调 C++（臂 arm_endpoint.library_path 按源码指纹现场编 C 接口库；要 WinLibs g++，缺了报安装提示）：
  - /predict_hit_pos 收不收、不收的原因：arm_controller.hit_pos = 臂节点收消息那一段 node::receive_hit_pos（认出 bot 求解失败报告
    就丢 → parse_hit_pos 全部合同门、臂标定指纹、击球侧）；不收的文本与臂发的 /tennis/status 逐字同。
  - 实测关节 → 黑标 / 甜点 / 拍速 / 拍面姿态：端点动态模型 ArmEndpoint.eval_series（control/arm_endpoint.hpp：四电机共用时延、
    J1 柔度、J1–J4 间隙开关、J1 力矩致轴倾斜、静态 J1 轴倾斜、黑标 / 甜点偏置；端点标定与 RL 训练器同一份）。逐关节真实采样时刻由
    端点标定同一个读包工具 arm_controller.tools.arm_endpoint_bag（rows_of → joint_samples：两相 CAN 调度认相 + 零位平移）还原。
  - 命令关节 → 甜点、J1 杠杆：同一端点模型的刚体位形 pose / pose_vjp（生产里命令侧本来就是刚体 FK）。
  - 录包关节角零位平移：臂零位帧登记表（config/zero_frames.yaml；录包自带的 run_config 行 zero_frame 优先，其次登记表 session_frame）。
  - /joint_states 关节名 → 控制器槽：ArmEndpoint.slot_names（车型 yaml + carcfg 槽位规则）。
判不了的（车型不在端点动态模型里、录包零位帧未登记、/joint_states 调度当前读包工具不认、方案是旧契约 / 旧臂标定）抛 Unavailable
或标 old_version，原因写进报告；不另写旧逻辑。

注意：ROS2 解释器（ros2/run_ros2.bat）的 PATH 里没有 g++，臂的 C 接口库只能在 tracker venv（generate_curve3_html /
racket_ht_black_marker / 在线黑标）里调；rosbag 抽取脚本（extract_arm_bag / extract_rk_tracking_bag）只抽原始数据。
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ARM_CONTROLLER_ROOT = Path(os.environ.get("TENNIS_MAN_ARM_CONTROLLER", "D:/tennis-man/arm_controller"))
_NEEDED = ("arm_controller/arm_endpoint.py", "arm_controller/hit_pos.py", "arm_controller/tools/arm_endpoint_bag.py")
# 端点动态模型的五个关节 J1 J2 J3 J4 J6 在控制器 6 槽里的位置（arm_endpoint_bag.SLOTS = endpoint::kSlot）
_MODS = None
_ENDPOINTS: dict = {}


class Unavailable(RuntimeError):
    """当前生产代码判不了 / 算不了这一场（原因写进报告页），不是程序错误。"""


def arm():
    """臂仓库的 Python 包：endpoint（arm_endpoint）、hit_pos、bag（tools.arm_endpoint_bag）。checkout 缺文件抛 Unavailable。"""
    global _MODS
    if _MODS is None:
        src = ARM_CONTROLLER_ROOT / "src"
        missing = [rel for rel in _NEEDED if not (src / rel).is_file()]
        if missing:
            raise Unavailable(
                f"臂仓库 {ARM_CONTROLLER_ROOT} 没有 {'、'.join(missing)}——报告要调的生产代码不在这个 checkout 里"
                "（臂生产分支还没含它们时，设 TENNIS_MAN_ARM_CONTROLLER 指向含它们的臂工作树）")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from arm_controller import arm_endpoint, hit_pos
        from arm_controller.tools import arm_endpoint_bag

        got = Path(arm_endpoint.__file__).resolve()
        if got.parents[2] != ARM_CONTROLLER_ROOT.resolve():
            raise Unavailable(f"import 到的 arm_controller 在 {got}，不是 {ARM_CONTROLLER_ROOT}（sys.path 上另有一份臂代码）")
        _MODS = SimpleNamespace(endpoint=arm_endpoint, hit_pos=hit_pos, bag=arm_endpoint_bag)
    return _MODS


def checkout() -> dict:
    """报告页写出「本页按哪一版臂生产代码算的」：路径、提交、工作树有无未提交改动（git 不可用时只给路径）。"""
    info: dict = {"path": str(ARM_CONTROLLER_ROOT)}
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(ARM_CONTROLLER_ROOT), *args], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=20).stdout.strip()
        info["commit"] = git("rev-parse", "--short=12", "HEAD") or None
        info["branch"] = git("rev-parse", "--abbrev-ref", "HEAD") or None
        info["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except Exception:  # noqa: BLE001 —— 溯源信息缺了不影响计算
        pass
    return info


def endpoint(car: str):
    """车型的端点模型（ArmEndpoint：车型 yaml + assets/<car>/fk_chain.json）。车型不在现役臂里抛 Unavailable。"""
    if car not in _ENDPOINTS:
        try:
            _ENDPOINTS[car] = arm().endpoint.ArmEndpoint(car)
        except Unavailable:
            raise
        except Exception as exc:  # noqa: BLE001 —— 车型 yaml / fk_chain 不在、或不合当前代码
            raise Unavailable(f"当前臂生产代码载不了车型 {car} 的端点模型：{exc}") from exc
    return _ENDPOINTS[car]


def slot_names(car: str) -> tuple:
    """控制器 6 槽的 ROS 关节名（None = 空槽）。"""
    return endpoint(car).slot_names


# ───────────────────────── 录包零位帧 ─────────────────────────
_RUN_CONFIG_KEYS = ("zero_frame", "face_table_fnv1a64", "arm_calibration_fingerprint", "rl_fingerprint", "ff_model_fingerprint")


def parse_run_config(text: str) -> dict:
    """臂起栈 /rosout 的 run_config 行（node/run_config.hpp run_identity_text：' | ' 之后空格分隔的 键=值）→ 键值。只读日志，不判。"""
    out: dict = {"text": text}
    tail = text.split(" | ", 1)[1] if " | " in text else text
    for key, value in re.findall(r"(\w+)=(\S+)", tail):
        if key in _RUN_CONFIG_KEYS:
            out[key] = value
    return out


def recorded_frame(car: str, run: str, run_configs: list | None) -> tuple[str, str]:
    """录包关节角是在哪个零位帧下录的 → (帧 id, 依据)。录包里的臂起栈 run_config 行（E0）优先，其次登记表的录包名；都没有抛 Unavailable。
    run_configs = extract_arm_bag 抽的原始行 {"recv", "text"}（这里按 parse_run_config 解；已解好的带 zero_frame 原样用）。"""
    m = arm()
    parsed = [rc if rc.get("zero_frame") else parse_run_config(str(rc.get("text") or ""))
              for rc in (run_configs or []) if isinstance(rc, dict)]
    frames = {rc.get("zero_frame") for rc in parsed if rc.get("zero_frame")}
    if len(frames) > 1:
        raise Unavailable(f"录包里臂起栈了 {len(parsed)} 次、零位帧不同（{'、'.join(sorted(frames))}），一场一个平移量算不了")
    if frames:
        frame = frames.pop()
        if frame == "unregistered":
            raise Unavailable("录包的臂起栈 run_config 行写着 zero_frame=unregistered：那组零位没登记，在臂 config/zero_frames.yaml 补登记")
        try:
            m.endpoint.zero_shift(car, frame)
        except Exception as exc:  # noqa: BLE001
            raise Unavailable(f"run_config 行的零位帧 {frame} 在当前臂登记表里查不到：{exc}") from exc
        return frame, "录包 run_config 行（臂起栈时按登记表反查）"
    try:
        return m.endpoint.session_frame(car, run), f"臂零位帧登记表（config/zero_frames.yaml 录包名 {run}）"
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(
            f"录包零位帧未登记：录包没有臂起栈 run_config 行，臂 config/zero_frames.yaml 也没有 {run}（{exc}）——"
            "按登记表规矩带证据补登记后重出") from exc


# ───────────────────────── /predict_hit_pos 判决与收拒对账 ─────────────────────────
_PLAN_ID_RE = re.compile(r"(?:^|\s)plan_id=(\S+)")
_HINT_ID_RE = re.compile(r"(?:^|\s)hint_id=(\S+)")
_REVISION_RE = re.compile(r"(?:^|\s)revision=(\d+)")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _status_outcome(text: str) -> str:
    for prefix, name in (("accepted ", "accepted"), ("reject ", "reject"), ("error hit_pos", "error")):
        if text.startswith(prefix):
            return name
    return "other"


def plan_audit(arm_json: dict, car: str) -> dict:
    """逐条 /predict_hit_pos 按臂受理口径判（event["check"]），并和录包里臂自己的 /tennis/status 逐条对账。

    verdict：executable（解析通过，臂交给路由）/ solve_failed（bot 求解失败报告，臂认出就丢、不发 status）/
    rejected（当前臂契约不收，text = 臂 status 原文）/ old_version（旧契约或按另一份臂标定解的方案：当前生产代码判不了它当时该不该收，
    只看录包里臂当时的 status；报告照常用臂当时收下的方案）。判决时刻 t_recv = generated_at（即时送达）；臂 status 是
    "error hit_pos" 而判决是 executable 时，再按臂 status 的发布时刻（= 收到时刻）重判，同文 = 送达晚造成的拒收。
    """
    m = arm()
    try:
        own = m.hit_pos.fingerprint(car)
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(f"当前臂生产代码装不了车型 {car}（方案判决）：{exc}") from exc
    statuses: dict = {}
    for e in arm_json.get("events") or []:
        if e.get("topic") != "/tennis/status":
            continue
        text = str(e.get("text") or "")
        ident, rev = _PLAN_ID_RE.search(text) or _HINT_ID_RE.search(text), _REVISION_RE.search(text)
        if ident and rev:
            statuses.setdefault((ident.group(1), int(rev.group(1))), []).append(e)
    counts = {"executable": 0, "solve_failed": 0, "rejected": 0, "old_version": 0}
    kinds: dict = {}
    fail_reasons: dict = {}
    old_reasons: dict = {}
    reject_texts: dict = {}
    arm_status = {"accepted": 0, "reject": {}, "error": {}, "missing": 0}
    agree = {"agree": 0, "arrival": 0, "disagree": 0, "missing": 0, "not_compared": 0}
    disagreements: list = []
    for e in arm_json.get("events") or []:
        if e.get("topic") != "/predict_hit_pos":
            continue
        text = str(e.get("text") or "")
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        parsed = isinstance(payload, dict)
        payload = payload if parsed else {}
        kind = payload.get("kind")
        t_recv = float(payload["generated_at"]) if _finite(payload.get("generated_at")) else 0.0
        v = m.hit_pos.check(car, text, t_recv)
        verdict, reason = v.verdict, None
        # 旧合同只认「是 JSON 对象、没有 kind」的历史载荷；解析不了的 JSON 哪一代臂都不收，照臂原文算不收
        if verdict == "rejected" and parsed and kind is None:
            verdict, reason = "old_version", "无 kind 的旧合同（final_hit_plan/v1 之前）"
        elif verdict == "rejected" and kind == "final_hit_plan" and payload.get("arm_calibration_fingerprint") != own:
            got = payload.get("arm_calibration_fingerprint")
            verdict = "old_version"
            reason = ("方案没有 arm_calibration_fingerprint（1012 前的契约）" if got is None
                      else f"方案按臂标定 {got} 解，当前臂是 {own}")
        check = {"verdict": verdict, "kind": kind if isinstance(kind, str) else None}
        if v.text:
            check["text"] = v.text
        if reason:
            check["reason"] = reason
        if v.side is not None and kind == "final_hit_plan":
            check["side"] = v.side
        if verdict == "solve_failed":
            check["fail_reason"] = payload.get("fail_reason")
            check["fail_detail"] = payload.get("fail_detail")
            key = f"{payload.get('fail_reason')}/{payload.get('fail_detail')}"
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
        if verdict == "old_version":
            old_reasons[reason] = old_reasons.get(reason, 0) + 1
        if verdict == "rejected":
            short = re.sub(r" (plan_id|hint_id)=.*$", "", v.text)
            reject_texts[short] = reject_texts.get(short, 0) + 1
        counts[verdict] += 1
        kinds[str(kind)] = kinds.get(str(kind), 0) + 1
        e["check"] = check
        if kind not in ("final_hit_plan", "preaim_hint"):
            continue
        ident = payload.get("plan_id") if kind == "final_hit_plan" else payload.get("hint_id")
        rev = payload.get("revision")
        got = statuses.get((ident, rev), []) if isinstance(ident, str) and isinstance(rev, int) else []
        if verdict == "solve_failed":
            if got:
                agree["disagree"] += 1
                disagreements.append({"id": ident, "revision": rev, "verdict": verdict, "arm": got[0]["text"]})
            else:
                agree["agree"] += 1
            continue
        if kind == "final_hit_plan":
            if not got:
                arm_status["missing"] += 1
            else:
                outcome = _status_outcome(str(got[0]["text"]))
                short = re.sub(r" (plan_id|hint_id)=.*$", "", str(got[0]["text"]))
                if outcome == "accepted":
                    arm_status["accepted"] += 1
                elif outcome in ("reject", "error"):
                    arm_status[outcome][short] = arm_status[outcome].get(short, 0) + 1
        if verdict == "old_version":
            agree["not_compared"] += 1
            continue
        if not got:
            agree["missing"] += 1
            continue
        status_text, outcome = str(got[0]["text"]), _status_outcome(str(got[0]["text"]))
        if verdict == "executable" and outcome in ("accepted", "reject"):
            agree["agree"] += 1
        elif verdict == "rejected" and outcome == "error" and status_text == v.text:
            agree["agree"] += 1
        elif verdict == "executable" and outcome == "error" and _finite(got[0].get("t")) and \
                m.hit_pos.check(car, text, float(got[0]["t"])).text == status_text:
            agree["arrival"] += 1
            check["arrival_reject"] = status_text
        else:
            agree["disagree"] += 1
            disagreements.append({"id": ident, "revision": rev, "verdict": verdict, "verdict_text": v.text,
                                  "arm": status_text})
    return {
        "arm_checkout": checkout(),
        "car": car,
        "arm_calibration_fingerprint": own,
        "decider": "arm_controller.hit_pos → C++ node::receive_hit_pos（t_recv = generated_at）",
        "verdicts": counts,
        "kinds": kinds,
        "solve_failed_reasons": fail_reasons,
        "old_version_reasons": old_reasons,
        "rejected_texts": reject_texts,
        "arm_status_final_plans": arm_status,
        "arm_status_vs_verdict": agree,
        "disagreements": disagreements[:50],
    }


# ───────────────────────── 端点：实测关节动态量 + 命令刚体量 ─────────────────────────
_LEGACY_DERIVED = ("tcp", "marker", "fy", "fp", "p_head", "v_tcp_arm", "sweet", "sweet_vel", "sweet_h", "marker_h",
                   "face_yaw", "face_pitch", "j1_speed")
# 拍头椭圆半轴不传（0）：报告只要黑标 / 甜点 / 拍面，不要拍框最低点
_DIGITS_M, _DIGITS_V, _DIGITS_DEG = 5, 4, 3


def _six(v) -> bool:
    return isinstance(v, list) and len(v) == 6 and all(_finite(x) for x in v)


def _j1_lever(ep, q6: np.ndarray) -> np.ndarray:
    """刚体位形下甜点对 q1 的杠杆 |∂甜点/∂q1|（m/rad；端点 pose_vjp，J1 轴倾斜在内）。"""
    pose = arm().endpoint.POSE
    cols = range(pose["sweet"].start, pose["sweet"].stop)
    grads = []
    for c in cols:
        cot = np.zeros((len(q6), 10))
        cot[:, c] = 1.0
        grads.append(ep.pose_vjp(q6, cot)[:, 0])
    return np.sqrt(np.sum(np.square(grads), axis=0))


def endpoint_for_arm(arm_json: dict, car: str, run: str) -> dict:
    """就地给 arm JSON 写端点量，返回溯源（status ok；判不了抛 Unavailable，调用方写进页面）。

    states（实测 /joint_states）：端点动态模型 eval_series 在各行戳时刻求值——sweet / marker（车体系，x 右 y 前、原点 = J1 轴，
    z = 臂模型 z）、sweet_h / marker_h（离地高）、sweet_vel（车体系 m/s，±5 ms 中心差分）、face_yaw / face_pitch（°，拍面法向
    朝 +y 那支：yaw = atan2(nx, ny)、pitch = asin(nz)，车体系）；j1_speed = |q̇1|·|∂甜点/∂q1|（刚体位形雅可比，命令侧口径的对照量）。
    commands（/tennis/motor_command）：刚体位形 pose 的 sweet（同车体系）与 j1_speed。读数先从录包那一帧平移到当前车型 yaml 的帧。
    """
    for rows in (arm_json.get("states") or [], arm_json.get("commands") or []):
        for row in rows:
            if isinstance(row, dict):
                for key in _LEGACY_DERIVED:
                    row.pop(key, None)
    m = arm()
    ep = endpoint(car)
    if ep.slot_names[4] is not None:
        raise Unavailable(f"端点动态模型只覆盖五轴臂（J1 J2 J3 J4 J6，endpoint::kSlot）；{car} 有腕转槽，当前生产代码不算它的端点")
    raw = sorted(arm_json.get("state_joint_names_raw") or [])
    want = sorted(n for n in ep.slot_names if n is not None)
    if raw and raw != want:
        raise Unavailable(f"录包 /joint_states 关节名 {raw} 不是当前车型 {car} 的 {want}")
    frame, frame_source = recorded_frame(car, run, arm_json.get("run_configs"))
    shift = m.endpoint.zero_shift(car, frame)
    states = [s for s in arm_json.get("states") or []
              if isinstance(s, dict) and _finite(s.get("t")) and _six(s.get("position")) and _six(s.get("effort"))]
    if len(states) < 100:
        raise Unavailable(f"/joint_states 只有 {len(states)} 行带位置 + 力矩，端点动态模型要连续的关节序列")
    slots = list(m.bag.SLOTS)
    t = np.array([s["t"] for s in states], dtype=float)
    q = np.array([s["position"] for s in states], dtype=float)
    e = np.array([s["effort"] for s in states], dtype=float)
    rows = m.bag.rows_of(t, q[:, slots], e[:, slots])
    try:
        per = m.bag.joint_samples(rows, car, frame)
    except ValueError as exc:
        raise Unavailable(f"/joint_states 发送调度当前读包工具认不出（arm_endpoint_bag.joint_samples：{exc}）") from exc
    out = ep.eval_series([p["t"] for p in per], [p["q"] for p in per], [p["e"] for p in per], t)
    f = m.endpoint.field
    sweet, marker, vel = f(out, "sweet"), f(out, "marker"), f(out, "sweet_vel")
    sweet_h, marker_h = f(out, "sweet_height"), f(out, "marker_height")
    yaw, pitch = np.degrees(f(out, "face_yaw")), np.degrees(f(out, "face_pitch"))
    q_now = q + shift
    j1_state = np.abs(np.array([s["velocity"][0] if _six(s.get("velocity")) else np.nan for s in states])) * _j1_lever(ep, q_now)
    r5 = lambda a: [round(float(x), _DIGITS_M) for x in a]  # noqa: E731
    for i, s in enumerate(states):
        s["sweet"], s["marker"] = r5(sweet[i]), r5(marker[i])
        s["sweet_h"], s["marker_h"] = round(float(sweet_h[i]), _DIGITS_M), round(float(marker_h[i]), _DIGITS_M)
        s["sweet_vel"] = [round(float(x), _DIGITS_V) for x in vel[i]]
        s["face_yaw"], s["face_pitch"] = round(float(yaw[i]), _DIGITS_DEG), round(float(pitch[i]), _DIGITS_DEG)
        if math.isfinite(j1_state[i]):
            s["j1_speed"] = round(float(j1_state[i]), _DIGITS_V)
    commands = [c for c in arm_json.get("commands") or [] if isinstance(c, dict) and _six(c.get("position"))]
    if commands:
        qc = np.array([c["position"] for c in commands], dtype=float) + shift
        sweet_c = f(ep.pose(qc), "sweet", m.endpoint.POSE)
        vc = np.abs(np.array([c["velocity"][0] if _six(c.get("velocity")) else np.nan for c in commands]))
        j1_cmd = vc * _j1_lever(ep, qc)
        for i, c in enumerate(commands):
            c["sweet"] = r5(sweet_c[i])
            if math.isfinite(j1_cmd[i]):
                c["j1_speed"] = round(float(j1_cmd[i]), _DIGITS_V)
    return {
        "status": "ok",
        "car": car,
        "arm_checkout": checkout(),
        "states": "端点动态模型 ArmEndpoint.eval_series（arm_endpoint.hpp：时延 / J1 柔度 / J1–J4 间隙 / J1 力矩轴倾斜 / 静态 J1 轴倾斜），"
                  "逐关节采样时刻 = arm_endpoint_bag.joint_samples（两相 CAN 调度认相）",
        "commands": "端点刚体位形 ArmEndpoint.pose（命令侧，同生产）",
        "j1_speed": "|q̇1|·|∂甜点/∂q1|（端点刚体位形 pose_vjp；实测行用实测 q / q̇1，命令行用命令 q / q̇1）",
        "frame": {"recorded": frame, "source": frame_source, "current": m.endpoint.zero_frame(car)[0],
                  "shift_rad": [round(float(x), 9) for x in shift]},
        "dynamics": {k: float(v) for k, v in ep.dynamics.items()},
        "tilt_deg": [round(math.degrees(a), 4) for a in ep.tilt_angles],
        "z_offset_m": ep.z_offset,
        "rows": len(states),
        "commands_rows": len(commands),
        "velocity_half_window_ms": 5.0,
        "s_init": "J1 游隙 / J2–J4 间隙在录包开头之前的状态按 0（第一拍大力矩之前的段可能差半个间隙）",
    }
