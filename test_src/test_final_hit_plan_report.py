# -*- coding: utf-8 -*-
"""FinalHitPlan / PreAimHint 报告合同测试。

2026-10-13 起方案收不收由臂生产代码判（test_src/arm_production.py → 臂 arm_controller.hit_pos → C++ node::receive_hit_pos，
臂节点收消息那一段）：报告的 Python / JS 不再复刻 bot 的方案物理（旧的 car→arm 刚体变换 / 球心→拍心执行点 / 水平触球平面 /
碰撞前向复算 / 上旋 / 五轴切向对账整套删了，连同这里照 bot 公式造自洽方案的夹具——那是同一套公式的又一份实现）。这里守：
  - 抽取（ROS2 解释器，调不了臂 C 库）只映射、不判；
  - 页面只读 Python 写进 event.check 的判决，不判、不对账；
  - ACK 按 (plan_id, revision) 原子回配、contact_ht 同源校验；
  - plan_audit 的判决与录包里臂自己的 /tennis/status 逐字对账（要臂 checkout 与 g++，缺了 skip）。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_src.extract_arm_bag import _event_payload_time, _event_text
from test_src.extract_rk_tracking_bag import _payload_time, _report_prediction_payload


SRC = Path(__file__).with_name("generate_curve3_html.py")
RK_EXTRACTOR = Path(__file__).with_name("extract_rk_tracking_bag.py")
NODE = shutil.which("node")
STATUS_NUM_JS = ("\nconst statusNum=(text,key)=>{const m=new RegExp('(?:^|\\\\s)'+key+"
                 "'=(-?[0-9]+(?:\\\\.[0-9]+)?)').exec(text||'');return m?Number(m[1]):null;};\n")


def _core(begin: str, end: str) -> str:
    source = SRC.read_text(encoding="utf-8")
    match = re.search(
        rf"// \[\[{re.escape(begin)}\]\].*?\n(.*)// \[\[{re.escape(end)}\]\]",
        source,
        re.S,
    )
    assert match
    return match.group(1)


def _plan(plan_id: str = "throw42", revision: int = 3, contact_ht: float = 100.6) -> dict:
    """一条字段齐全的 final_hit_plan。数值只求形状对、不求物理自洽：报告不再对账方案物理。"""
    return {
        "kind": "final_hit_plan",
        "contract": "final_hit_plan/v1",
        "plan_id": plan_id,
        "revision": revision,
        "source_ct": 100.0,
        "generated_at": 100.012,
        "contact_ht": contact_ht,
        "n_points": 18,
        "contact_rel_x": 1.0,
        "contact_rel_y": 0.04,
        "contact_rel_z": 1.1,
        "arm_target_rel_x": 0.99,
        "arm_target_rel_y": 0.01,
        "arm_target_rel_z": 1.08,
        "contact_x": 1.2,
        "contact_y": 2.04,
        "contact_z": 1.1,
        "arm_center_x": 0.2,
        "arm_center_y": 2.0,
        "car_center_x": 0.2,
        "car_center_y": 1.955,
        "car_yaw_at_ht": 0.05,
        "face_normal_yaw_world": -0.08,
        "face_pitch": 0.31,
        "compensated_speed": 5.4,
        "apex_z_world": 2.55,
        "net_clearance_m": 0.24,
        "collision_en": 0.36,
        "collision_kt": 1.15,
        "solve_iterations": 2,
        "solve_ms": 1.38,
        "model_fingerprint": "world3d_topspin_v2_five_axis_sweetspot:0123456789abcdef",
        "arm_calibration_fingerprint": "dc47f4de4b72bebf",
    }


def _preaim() -> dict:
    return {
        "kind": "preaim_hint",
        "contract": "preaim_hint/v1",
        "hint_id": "throw42",
        "revision": 2,
        "source_ct": 99.7,
        "generated_at": 99.712,
        "estimate_ht": 100.6,
        "n_points": 12,
        "estimate_rel_x": 0.9,
        "estimate_rel_y": 1.1,
        "estimate_rel_z": 1.0,
        # These legacy-looking fields must not make a pre-aim message a FinalHT.
        "ct": 99.7,
        "ht": 100.6,
        "rel_x": 999.0,
        "rel_z": 999.0,
    }


def _solve_failed() -> dict:
    return {**_plan("throw43", 1), "plan_status": "solve_failed", "fail_reason": "no_meeting_root",
            "fail_detail": "meeting"}


def test_arm_extractor_preserves_full_json_and_uses_generated_at():
    long_value = "x" * 20_000
    payload = {**_plan(), "diagnostic": long_value}
    text = json.dumps(payload)
    assert _event_text(text, object()) == text
    assert json.loads(_event_text(text, object()))["diagnostic"] == long_value
    assert _event_payload_time("/predict_hit_pos", text) == pytest.approx(100.012)
    assert _event_payload_time("/predict_hit_pos", json.dumps({"ct": 8.0})) == 8.0
    # Declaring a new kind disables the legacy ct fallback.
    assert _event_payload_time(
        "/predict_hit_pos", json.dumps({"kind": "final_hit_plan", "ct": 8.0})
    ) is None
    source = Path(_event_text.__code__.co_filename).read_text(encoding="utf-8")
    assert "EVENT_TEXT_MAX" not in source
    assert "text = text[:" not in source


def test_rk_extractor_maps_plans_raw_and_skips_preaim_and_solve_failures():
    preaim = _preaim()
    plan = _plan()
    assert _payload_time("/predict_hit_pos", preaim) == pytest.approx(preaim["generated_at"])
    assert _report_prediction_payload(preaim) is None
    # bot 求解失败报告没有触球点，不是预测
    assert _report_prediction_payload(_solve_failed()) is None
    mapped = _report_prediction_payload(plan)
    assert mapped is not None
    assert mapped["ct"] == plan["source_ct"]
    assert mapped["ht"] == plan["contact_ht"]
    assert mapped["duration"] == pytest.approx(plan["contact_ht"] - plan["source_ct"])
    assert [mapped[k] for k in ("x", "y", "z")] == pytest.approx(
        [plan["contact_x"], plan["contact_y"], plan["contact_z"]]
    )
    assert [mapped[k] for k in ("rel_x", "rel_y", "rel_z")] == pytest.approx(
        [plan["contact_rel_x"], plan["contact_rel_y"], plan["contact_rel_z"]]
    )
    assert (mapped["arm_pred_x"], mapped["arm_pred_y"]) == (plan["arm_center_x"], plan["arm_center_y"])
    # 抽取不判：契约错、字段坏的方案照样映射（收不收报告生成时由臂生产代码判，见 arm_production.plan_audit）
    malformed = {**plan, "contract": "final_hit_plan/v0", "arm_forward_offset_m": float("nan")}
    assert _report_prediction_payload(malformed)["contract"] == "final_hit_plan/v0"
    no_time = {k: v for k, v in plan.items() if k != "source_ct"}
    assert _report_prediction_payload(no_time)["duration"] is None
    legacy = {"ct": 8.0, "ht": 8.5, "rel_x": 0.2}
    assert _report_prediction_payload(legacy) is legacy


def test_report_and_extractors_carry_no_plan_physics():
    """方案物理只在 bot（解）与臂（指纹核对）里：报告链不再有第二份对账。"""
    for path in (SRC, RK_EXTRACTOR):
        source = path.read_text(encoding="utf-8")
        for gone in ("_valid_final_hit_plan_physics", "spin_cross_v1", "碰撞前向复算", "car→arm刚体变换",
                     "球心→拍心执行点", "水平触球平面", "拍心世界水平速度", "_FINAL_HIT_PLAN_FINGERPRINT_RE"):
            assert gone not in source, (path.name, gone)
    source = SRC.read_text(encoding="utf-8")
    assert "return structured?null:chassisTargetForThrow" in source
    assert "armPreds.filter(p=>p.kind==='final_hit_plan')" in source
    for field in (
        "collision_en", "collision_kt", "apex_z_world", "net_clearance_m", "solve_iterations", "solve_ms",
        "model_fingerprint", "contact_rel_x", "arm_target_rel_x",
    ):
        assert field in source
    assert "碰撞 / 求解" in source


def _parse_harness(events: list, tail: str) -> str:
    return (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:100};\n"
        f"const ARM={{events:{json.dumps(events)}}};\n"
        + _core("final-hit-plan-parse-core-begin", "final-hit-plan-parse-core-end")
        + STATUS_NUM_JS
        + _core("final-hit-plan-ack-core-begin", "final-hit-plan-ack-core-end")
        + tail
    )


def _run(tmp_path: Path, body: str) -> dict:
    script = tmp_path / "final_plan_contract.js"
    script.write_text(body, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _event(t: float, payload: dict, check: dict | None) -> dict:
    event = {"t": t, "topic": "/predict_hit_pos", "text": json.dumps(payload)}
    if check is not None:
        event["check"] = check
    return event


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_malformed_declared_new_message_still_disables_legacy_fallback(tmp_path):
    body = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const ARM={events:[{t:0,topic:'/predict_hit_pos',"
        "text:'{\\\"kind\\\":\\\"final_hit_plan\\\"'}]};\n"
        + _core("final-hit-plan-parse-core-begin", "final-hit-plan-parse-core-end")
        + "\nconsole.log(JSON.stringify({structured:hasStructuredHitMessages,"
          "bad:armPredParseBad,preds:armPreds.length}));\n"
    )
    assert _run(tmp_path, body) == {"structured": True, "bad": 1, "preds": 0}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_page_reads_the_production_verdict_and_never_judges(tmp_path):
    """event.check 是 Python 用臂生产代码判好的：executable / old_version / 未判 进 FinalHT 候选，rejected 原文进
    合同错误（= 臂 /tennis/status 原文），solve_failed 单独记原因、不算合同错误；页面自己不看方案物理。"""
    rejected_text = ("error hit_pos: final plan arm calibration fingerprint 00000000deadbeef differs from this arm's "
                     "dc47f4de4b72bebf (bot loaded another arm calibration) plan_id=bad revision=1 contact_ht=100.600000")
    # 物理上乱写的方案：页面不该看出任何问题（判决只看 check）
    nonsense = {**_plan("nonsense", 1), "outgoing_vz": 99.0, "car_center_x": 7.0}
    events = [
        _event(-0.3, _preaim(), {"verdict": "executable", "kind": "preaim_hint"}),
        _event(0.0, _plan("good", 1), {"verdict": "executable", "kind": "final_hit_plan", "side": 1}),
        _event(0.01, _plan("old", 1), {"verdict": "old_version", "kind": "final_hit_plan",
                                       "reason": "方案没有 arm_calibration_fingerprint（1012 前的契约）"}),
        _event(0.02, nonsense, None),
        _event(0.03, _plan("bad", 1), {"verdict": "rejected", "kind": "final_hit_plan", "text": rejected_text}),
        _event(0.04, _solve_failed(), {"verdict": "solve_failed", "kind": "final_hit_plan",
                                       "fail_reason": "no_meeting_root", "fail_detail": "meeting"}),
        _event(0.05, {**_plan("noid", 1), "plan_id": ""}, {"verdict": "executable", "kind": "final_hit_plan"}),
    ]
    result = _run(tmp_path, _parse_harness(events, (
        "console.log(JSON.stringify({preds:armPreds.map(p=>[p.planId,p.verdict]),preaim:armPreaimCount,"
        "errors:armPlanContractErrors,failed:armSolveFailed,unusable:armPlanUnusable,counts:armPlanVerdictCounts}));\n")))
    assert result["preds"] == [["good", "executable"], ["old", "old_version"], ["nonsense", "unchecked"]]
    assert result["preaim"] == 1
    assert result["errors"] == [rejected_text]
    assert result["failed"] == [{"t": 0.04, "planId": "throw43", "revision": 1, "reason": "no_meeting_root",
                                 "detail": "meeting"}]
    assert result["unusable"] == ["?#1"]
    assert result["counts"] == {"executable": 1, "old_version": 1, "unchecked": 1}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_report_matches_only_exact_plan_identity_and_contact_ht(tmp_path):
    good = _plan(revision=3, contact_ht=100.6)
    wrong_ht = _plan(revision=4, contact_ht=100.62)
    conflict_a = _plan("conflict", 5)
    conflict_b = {**conflict_a, "solve_ms": conflict_a["solve_ms"] + 0.01}
    ok = {"verdict": "executable", "kind": "final_hit_plan", "side": 1}
    events = [
        _event(-0.3, _preaim(), {"verdict": "executable", "kind": "preaim_hint"}),
        _event(0.0, good, ok),
        _event(0.01, wrong_ht, ok),
        _event(0.023, conflict_a, ok),
        _event(0.024, conflict_b, ok),
        {"t": 0.05, "topic": "/tennis/status", "text":
         "accepted hit x=1.0120 z=1.0800 duration=0.5500 "
         "plan_id=throw42 revision=3 contact_ht=100.600000 solve_ms=1.380"},
        {"t": 0.06, "topic": "/tennis/status", "text":
         "accepted hit x=1.0120 z=1.0800 duration=0.5600 "
         "plan_id=throw42 revision=4 contact_ht=100.610000 solve_ms=1.400"},
        {"t": 0.07, "topic": "/tennis/status", "text":
         "accepted hit x=1.0120 z=1.0800 duration=0.5600 "
         "plan_id=conflict revision=5 contact_ht=100.600000 solve_ms=1.400"},
    ]
    result = _run(tmp_path, _parse_harness(events, (
        "const rec=acceptedRecordForPlanAck(armPlanAcks[0]);\n"
        "console.log(JSON.stringify({structured:hasStructuredHitMessages,preaim:armPreaimCount,"
        "preds:armPreds.length,plans:armFinalPlans.length,acks:armPlanAcks.length,ackErrors:armPlanAckErrors,"
        "contact:[rec.plan.rel_x,rec.plan.rel_y,rec.plan.rel_z],"
        "target:[rec.wx,rec.wy,rec.wz],id:rec.plan.planId,revision:rec.plan.revision}));\n")))
    assert result["structured"] is True
    assert result["preaim"] == 1
    assert result["preds"] == result["plans"] == 4
    assert result["acks"] == 1
    assert any("payload 不一致" in err for err in result["ackErrors"])
    assert any("contact_ht" in err for err in result["ackErrors"])
    assert any("conflict#5" in err and "找不到同身份" in err for err in result["ackErrors"])
    expected = _plan()
    assert result["contact"] == pytest.approx([
        expected["contact_rel_x"], expected["contact_rel_y"], expected["contact_rel_z"],
    ])
    assert result["target"] == pytest.approx([
        expected["arm_target_rel_x"], expected["arm_target_rel_y"], expected["arm_target_rel_z"],
    ])
    assert (result["id"], result["revision"]) == ("throw42", 3)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_final_plan_session_calibrates_arm_z_offset_from_plan_acks(tmp_path):
    """FinalHitPlan 场没有旧状态可回配，zOff（旧状态回配的值键与悬停溯源用）从 plan_id+revision 原子回配的 ACK 标定：
    臂端 z = arm_target_rel_z + hit_pos_z_offset_m。"""
    z_offset = -0.193109308
    events = []
    for k in range(3):
        plan = _plan(plan_id=f"throw{k}", revision=1)
        events.append(_event(0.02 + 0.01 * k, plan, {"verdict": "executable", "kind": "final_hit_plan"}))
        events.append({"t": 0.05 + 0.01 * k, "topic": "/tennis/status", "text":
                       f"accepted hit x={plan['arm_target_rel_x']:.4f} "
                       f"z={plan['arm_target_rel_z'] + z_offset:.4f} duration=0.5500 "
                       f"plan_id=throw{k} revision=1 contact_ht=100.600000 solve_ms=1.380"})
    body = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:100};\n"
        "let HIT_TIME_ADVANCE_SEC=0.0;\n"
        f"const ARM={{events:{json.dumps(events)}}};\n"
        + _core("arm-prediction-match-core-begin", "arm-prediction-match-core-end")
        + _core("final-hit-plan-parse-core-begin", "final-hit-plan-parse-core-end")
        + _core("arm-pred-align-core-begin", "arm-pred-align-core-end")
        + STATUS_NUM_JS
        + _core("final-hit-plan-ack-core-begin", "final-hit-plan-ack-core-end")
        + _core("arm-const-cal-core-begin", "arm-const-cal-core-end")
        + "console.log(JSON.stringify({acks:armPlanAcks.length,cal:armConstCal,z:ARM_HIT_Z_OFFSET}));\n"
    )
    result = _run(tmp_path, body)
    assert result["acks"] == 3
    assert result["cal"]["zOff"] == pytest.approx(-0.1931, abs=1e-9)
    assert result["cal"]["n"] == 3
    assert result["z"] == pytest.approx(-0.1931, abs=1e-9)


# ───────────────────────── plan_audit：臂生产代码判决 + 与臂 /tennis/status 逐字对账 ─────────────────────────

def _production():
    """臂生产代码（判决库）+ 臂自己测试里那条字段齐全、在各合同门内的方案夹具（合法方案长什么样只有臂说了算，这里不另造）。"""
    import importlib.util

    import test_src.arm_production as ap

    try:
        own = ap.arm().hit_pos.fingerprint("v05")
    except Exception as exc:  # noqa: BLE001 —— 没有臂 checkout / g++ 的机器
        pytest.skip(f"臂生产代码不可用：{exc}")
    path = ap.ARM_CONTROLLER_ROOT / "tests" / "test_hit_pos_lib.py"
    if not path.is_file():
        pytest.skip(f"臂 checkout 没有 {path}")
    spec = importlib.util.spec_from_file_location("arm_tests_hit_pos_lib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return ap, own, module.plan


def _status(t: float, text: str) -> dict:
    return {"t": t, "topic": "/tennis/status", "text": text}


def _ident(plan_id: str) -> str:
    return f" plan_id={plan_id} revision=3 contact_ht=100.500000"


def test_plan_audit_uses_the_arm_verdict_and_reconciles_the_arm_status():
    ap, own, arm_plan = _production()
    good = arm_plan("v05", plan_id="good")
    backhand = arm_plan("v05", plan_id="back", arm_target_rel_x=-1.0, contact_rel_x=-0.97, contact_x=-0.97)
    stale = arm_plan("v05", plan_id="stale", arm_calibration_fingerprint="00000000deadbeef")
    old = arm_plan("v05", plan_id="old", arm_calibration_fingerprint=None)
    late = arm_plan("v05", plan_id="late")
    legacy = {"ct": 99.0, "ht": 99.5, "rel_x": 1.0, "rel_z": 1.0}
    late_text = "error hit_pos: contact_ht must be after generation and receipt" + _ident("late")
    events = [
        {"t": 100.01, "topic": "/predict_hit_pos", "text": json.dumps(p)}
        for p in (good, backhand, stale, old, late, _solve_failed(), legacy)
    ] + [
        {"t": 100.02, "topic": "/predict_hit_pos", "text": "{not json"},
        _status(100.05, "accepted hit x=1.0000 z=0.8127 duration=0.4500" + _ident("good")),
        _status(100.05, "accepted hit x=1.0000 z=0.8127 duration=0.4500" + _ident("back")),
        _status(100.05, "accepted hit x=1.0000 z=0.8127 duration=0.4500" + _ident("old")),
        # 臂收到时 HT 已过（送达晚）：按臂 status 的发布时刻重判同文 = 送达晚造成的拒收，不算不一致
        _status(100.6, late_text),
    ]
    audit = ap.plan_audit({"events": events}, "v05")
    check = {json.loads(e["text"]).get("plan_id", "legacy") if e["text"].startswith("{\"") else "garbage": e["check"]
             for e in events if e["topic"] == "/predict_hit_pos"}
    assert check["good"] == {"verdict": "executable", "kind": "final_hit_plan", "side": 1}
    assert check["back"] == {"verdict": "executable", "kind": "final_hit_plan", "side": -1}
    # 指纹不同 = 按另一份臂标定解的方案：当前代码判不了它当时该不该收 → old_version，臂原文与原因都留下
    assert check["stale"]["verdict"] == "old_version" and own in check["stale"]["reason"]
    assert check["stale"]["text"] == ("error hit_pos: final plan arm calibration fingerprint 00000000deadbeef differs "
                                      f"from this arm's {own} (bot loaded another arm calibration)" + _ident("stale"))
    assert check["old"]["verdict"] == "old_version" and "1012 前" in check["old"]["reason"]
    assert check["late"]["verdict"] == "executable" and check["late"]["arrival_reject"] == late_text
    assert check["throw43"] == {"verdict": "solve_failed", "kind": "final_hit_plan",
                                "fail_reason": "no_meeting_root", "fail_detail": "meeting"}
    assert check["legacy"]["verdict"] == "old_version" and "无 kind" in check["legacy"]["reason"]
    assert check["garbage"]["verdict"] == "rejected" and check["garbage"]["text"].startswith("error hit_pos: json")
    assert audit["verdicts"] == {"executable": 3, "solve_failed": 1, "rejected": 1, "old_version": 3}
    assert audit["solve_failed_reasons"] == {"no_meeting_root/meeting": 1}
    assert audit["arm_calibration_fingerprint"] == own
    # 求解失败报告臂认出就丢、不发 status = 一致；旧版本不比
    assert audit["arm_status_vs_verdict"] == {"agree": 3, "arrival": 1, "disagree": 0, "missing": 0, "not_compared": 2}
    assert audit["arm_status_final_plans"] == {
        "accepted": 3, "reject": {}, "error": {"error hit_pos: contact_ht must be after generation and receipt": 1},
        "missing": 1}
    assert audit["disagreements"] == []


def test_plan_audit_flags_a_status_the_verdict_cannot_explain():
    ap, _, arm_plan = _production()
    # 当前臂契约不收的方案（缺字段），录包里臂却说 accepted ⇒ 不一致，两边原文都留下
    broken = arm_plan("v05", plan_id="broken", landing_x=None)
    events = [
        {"t": 100.01, "topic": "/predict_hit_pos", "text": json.dumps(broken)},
        _status(100.05, "accepted hit x=1.0000 z=0.8127 duration=0.4500" + _ident("broken")),
    ]
    audit = ap.plan_audit({"events": events}, "v05")
    check = events[0]["check"]
    assert check["verdict"] == "rejected"
    assert check["text"] == "error hit_pos: json: missing key landing_x" + _ident("broken")
    assert audit["arm_status_vs_verdict"]["disagree"] == 1
    assert audit["disagreements"][0]["arm"].startswith("accepted hit")
    assert audit["disagreements"][0]["verdict_text"] == check["text"]
