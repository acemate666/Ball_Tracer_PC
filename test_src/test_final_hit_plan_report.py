# -*- coding: utf-8 -*-
"""FinalHitPlan/PreAimHint report contract tests."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_src.extract_arm_bag import _event_payload_time, _event_text
from test_src.extract_rk_tracking_bag import _payload_time, _report_prediction_payload


SRC = Path(__file__).with_name("generate_curve3_html.py")
NODE = shutil.which("node")


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
    car_yaw = 0.05
    car_yaw_rate = -0.1
    arm_forward_offset = 0.045
    ball_radius = 0.033
    target_x_bias = 0.0
    target_z_bias = -0.02
    arm_center = (0.2, 2.0)
    arm_center_v = (0.1, 0.3)
    offset = (-arm_forward_offset * math.sin(car_yaw),
              arm_forward_offset * math.cos(car_yaw))
    car_center = (arm_center[0] - offset[0], arm_center[1] - offset[1])
    car_center_v = (
        arm_center_v[0] + car_yaw_rate * offset[1],
        arm_center_v[1] - car_yaw_rate * offset[0],
    )
    face_yaw = -0.08
    face_pitch = 0.31
    horizontal_normal = (-math.sin(face_yaw), math.cos(face_yaw))
    face_normal = (
        horizontal_normal[0] * math.cos(face_pitch),
        horizontal_normal[1] * math.cos(face_pitch),
        math.sin(face_pitch),
    )
    contact_rel = (
        1.0,
        (ball_radius - horizontal_normal[0]) / horizontal_normal[1],
        1.1,
    )
    arm_target_rel = (
        contact_rel[0] - ball_radius * horizontal_normal[0] + target_x_bias,
        contact_rel[1] - ball_radius * horizontal_normal[1],
        contact_rel[2] + target_z_bias,
    )
    incoming = (0.2, -7.0, -1.2)
    compensated_speed = 5.4
    racket_contact_v = (
        arm_center_v[0] - car_yaw_rate * arm_target_rel[1]
        + compensated_speed * horizontal_normal[0],
        arm_center_v[1] + car_yaw_rate * arm_target_rel[0]
        + compensated_speed * horizontal_normal[1],
        0.2,
    )
    collision_en = 0.36441583352036017
    collision_kt = 1.1528285465425736
    relative = tuple(vin - vr for vin, vr in zip(incoming, racket_contact_v))
    relative_normal = sum(u * n for u, n in zip(relative, face_normal))
    outgoing = tuple(
        vr + collision_kt * (u - relative_normal * n) - collision_en * relative_normal * n
        for vr, u, n in zip(racket_contact_v, relative, face_normal)
    )
    return {
        "kind": "final_hit_plan",
        "contract": "final_hit_plan/v1",
        "plan_id": plan_id,
        "revision": revision,
        "source_ct": 100.0,
        "generated_at": 100.012,
        "contact_ht": contact_ht,
        "n_points": 18,
        "contact_rel_x": contact_rel[0],
        "contact_rel_y": contact_rel[1],
        "contact_rel_z": contact_rel[2],
        "arm_target_rel_x": arm_target_rel[0],
        "arm_target_rel_y": arm_target_rel[1],
        "arm_target_rel_z": arm_target_rel[2],
        "contact_x": arm_center[0] + contact_rel[0],
        "contact_y": arm_center[1] + contact_rel[1],
        "contact_z": contact_rel[2],
        "incoming_vx": incoming[0],
        "incoming_vy": incoming[1],
        "incoming_vz": incoming[2],
        "arm_center_x": arm_center[0],
        "arm_center_y": arm_center[1],
        "arm_center_vx": arm_center_v[0],
        "arm_center_vy": arm_center_v[1],
        "car_center_x": car_center[0],
        "car_center_y": car_center[1],
        "car_center_vx": car_center_v[0],
        "car_center_vy": car_center_v[1],
        "arm_forward_offset_m": arm_forward_offset,
        "tennis_ball_radius_m": ball_radius,
        "arm_target_x_bias_m": target_x_bias,
        "arm_target_z_bias_m": target_z_bias,
        "car_yaw_at_ht": car_yaw,
        "car_yaw_rate_at_ht": car_yaw_rate,
        "face_normal_yaw_world": face_yaw,
        "face_pitch": face_pitch,
        "face_normal_nx": face_normal[0],
        "face_normal_ny": face_normal[1],
        "face_normal_nz": face_normal[2],
        "compensated_speed": compensated_speed,
        "racket_contact_vx": racket_contact_v[0],
        "racket_contact_vy": racket_contact_v[1],
        "racket_contact_vz": racket_contact_v[2],
        "outgoing_vx": outgoing[0],
        "outgoing_vy": outgoing[1],
        "outgoing_vz": outgoing[2],
        "landing_x": 0.0,
        "landing_y": 15.5,
        "apex_z_world": 2.55,
        "net_clearance_m": 0.24,
        "collision_en": collision_en,
        "collision_kt": collision_kt,
        "model_fingerprint": "world3d_effective_v3_arm_center_sweetspot:0123456789abcdef",
        "solve_iterations": 2,
        "solve_ms": 1.38,
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
        "car_yaw_estimate": 0.04,
        "face_normal_yaw_seed": -0.07,
        # These legacy-looking fields must not make a pre-aim message a FinalHT.
        "ct": 99.7,
        "ht": 100.6,
        "rel_x": 999.0,
        "rel_z": 999.0,
    }


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


def test_rk_extractor_excludes_preaim_and_maps_contact_coordinates():
    preaim = _preaim()
    plan = _plan()
    assert _payload_time("/predict_hit_pos", preaim) == pytest.approx(preaim["generated_at"])
    assert _report_prediction_payload(preaim) is None
    mapped = _report_prediction_payload(plan)
    assert mapped is not None
    assert mapped["ct"] == plan["source_ct"]
    assert mapped["ht"] == plan["contact_ht"]
    assert [mapped[k] for k in ("x", "y", "z")] == pytest.approx(
        [plan["contact_x"], plan["contact_y"], plan["contact_z"]]
    )
    assert [mapped[k] for k in ("rel_x", "rel_y", "rel_z")] == pytest.approx(
        [plan["contact_rel_x"], plan["contact_rel_y"], plan["contact_rel_z"]]
    )
    malformed = {**plan, "contract": "final_hit_plan/v0", "ct": 7.0, "ht": 9.0}
    assert _report_prediction_payload(malformed) is None
    assert _report_prediction_payload({**plan, "revision": 0}) is None
    assert _report_prediction_payload({**plan, "arm_forward_offset_m": float("nan")}) is None
    legacy = {"ct": 8.0, "ht": 8.5, "rel_x": 0.2}
    assert _report_prediction_payload(legacy) is legacy


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(model_fingerprint="world3d_effective_v3_arm_center_sweetspot:xyz"),
        lambda p: p.update(revision=0),
        lambda p: p.update(solve_iterations=0),
        lambda p: p.update(solve_iterations=5),
        lambda p: p.update(solve_iterations=2.0),
        lambda p: p.update(generated_at=p["source_ct"] - 0.001),
        lambda p: p.update(generated_at=p["contact_ht"]),
        lambda p: p.update(car_center_x=p["car_center_x"] + 0.001),
        lambda p: p.update(arm_center_vy=p["arm_center_vy"] + 0.001),
        lambda p: p.update(contact_rel_y=p["contact_rel_y"] + 0.001,
                           contact_y=p["contact_y"] + 0.001),
        lambda p: p.update(arm_target_rel_x=p["arm_target_rel_x"] + 0.001),
        lambda p: p.update(face_normal_nx=p["face_normal_nx"] + 0.001),
        lambda p: p.update(racket_contact_vx=p["racket_contact_vx"] + 0.001),
        lambda p: p.update(outgoing_vz=p["outgoing_vz"] + 0.001),
    ],
)
def test_rk_extractor_rejects_each_invalid_final_plan_invariant(mutate):
    plan = _plan()
    mutate(plan)
    assert _report_prediction_payload(plan) is None


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
    script = tmp_path / "malformed_final_plan.js"
    script.write_text(body, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"structured": True, "bad": 1, "preds": 0}


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_report_matches_only_exact_plan_identity_and_contact_ht(tmp_path):
    good = _plan(revision=3, contact_ht=100.6)
    wrong_ht = _plan(revision=4, contact_ht=100.62)
    malformed = {**_plan("bad", 1), "contract": "final_hit_plan/v0", "ct": 1.0, "ht": 2.0}
    bad_geometry = {**_plan("bad-geometry", 1), "car_center_x": 0.3}
    bad_sweetspot = {
        **_plan("bad-sweetspot", 1),
        "arm_target_rel_x": _plan("bad-sweetspot", 1)["arm_target_rel_x"] + 0.01,
    }
    bad_collision = {
        **_plan("bad-collision", 1),
        "outgoing_vx": _plan("bad-collision", 1)["outgoing_vx"] + 0.1,
    }
    bad_plane = _plan("bad-plane", 1)
    bad_plane["contact_rel_y"] += 0.01
    bad_plane["contact_y"] += 0.01
    bad_racket = {
        **_plan("bad-racket", 1),
        "racket_contact_vy": _plan("bad-racket", 1)["racket_contact_vy"] + 0.01,
    }
    bad_fingerprint = {
        **_plan("bad-fingerprint", 1),
        "model_fingerprint": "world3d_effective_v2:0123456789abcdef",
    }
    bad_iterations = {**_plan("bad-iterations", 1), "solve_iterations": 5}
    bad_time = {**_plan("bad-time", 1), "generated_at": 99.99}
    conflict_a = _plan("conflict", 5)
    conflict_b = {**conflict_a, "solve_ms": conflict_a["solve_ms"] + 0.01}
    events = [
        {"t": -0.3, "topic": "/predict_hit_pos", "text": json.dumps(_preaim())},
        {"t": 0.0, "topic": "/predict_hit_pos", "text": json.dumps(good)},
        {"t": 0.01, "topic": "/predict_hit_pos", "text": json.dumps(wrong_ht)},
        {"t": 0.02, "topic": "/predict_hit_pos", "text": json.dumps(malformed)},
        {"t": 0.021, "topic": "/predict_hit_pos", "text": json.dumps(bad_geometry)},
        {"t": 0.0215, "topic": "/predict_hit_pos", "text": json.dumps(bad_sweetspot)},
        {"t": 0.022, "topic": "/predict_hit_pos", "text": json.dumps(bad_collision)},
        {"t": 0.0221, "topic": "/predict_hit_pos", "text": json.dumps(bad_plane)},
        {"t": 0.0222, "topic": "/predict_hit_pos", "text": json.dumps(bad_racket)},
        {"t": 0.0223, "topic": "/predict_hit_pos", "text": json.dumps(bad_fingerprint)},
        {"t": 0.0224, "topic": "/predict_hit_pos", "text": json.dumps(bad_iterations)},
        {"t": 0.0225, "topic": "/predict_hit_pos", "text": json.dumps(bad_time)},
        {"t": 0.023, "topic": "/predict_hit_pos", "text": json.dumps(conflict_a)},
        {"t": 0.024, "topic": "/predict_hit_pos", "text": json.dumps(conflict_b)},
        {"t": 0.05, "topic": "/tennis/status", "text":
         "accepted hit x=1.0120 z=1.0800 duration=0.5500 "
         "plan_id=throw42 revision=3 contact_ht=100.600000 solve_ms=1.380"},
        {"t": 0.06, "topic": "/tennis/status", "text":
         "accepted hit x=1.0120 z=1.0800 duration=0.5600 "
         "plan_id=throw42 revision=4 contact_ht=100.610000 solve_ms=1.400"},
    ]
    body = (
        "const isNum=v=>typeof v==='number'&&Number.isFinite(v);\n"
        "const RK={t0:100};\n"
        f"const ARM={{events:{json.dumps(events)}}};\n"
        + _core("final-hit-plan-parse-core-begin", "final-hit-plan-parse-core-end")
        + "\nconst statusNum=(text,key)=>{const m=new RegExp('(?:^|\\\\s)'+key+'=(-?[0-9]+(?:\\\\.[0-9]+)?)').exec(text||'');return m?Number(m[1]):null;};\n"
        + _core("final-hit-plan-ack-core-begin", "final-hit-plan-ack-core-end")
        + "\nconst rec=acceptedRecordForPlanAck(armPlanAcks[0]);\n"
          "console.log(JSON.stringify({structured:hasStructuredHitMessages,preaim:armPreaimCount,"
          "preds:armPreds.length,plans:armFinalPlans.length,acks:armPlanAcks.length,"
          "contractErrors:armPlanContractErrors,ackErrors:armPlanAckErrors,"
          "contact:[rec.plan.rel_x,rec.plan.rel_y,rec.plan.rel_z],"
          "target:[rec.wx,rec.wy,rec.wz],id:rec.plan.planId,revision:rec.plan.revision}));\n"
    )
    script = tmp_path / "final_plan_contract.js"
    script.write_text(body, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["structured"] is True
    assert result["preaim"] == 1
    assert result["preds"] == result["plans"] == 4
    assert result["acks"] == 1
    assert any("bad#1" in err for err in result["contractErrors"])
    assert any("car→arm刚体变换" in err for err in result["contractErrors"])
    assert any("球心→拍心执行点" in err for err in result["contractErrors"])
    assert any("水平触球平面" in err for err in result["contractErrors"])
    assert any("拍心世界水平速度" in err for err in result["contractErrors"])
    assert any("碰撞前向复算" in err for err in result["contractErrors"])
    assert any("model_fingerprint" in err for err in result["contractErrors"])
    assert any("solve_iterations" in err for err in result["contractErrors"])
    assert any("generated_at>=source_ct" in err for err in result["contractErrors"])
    assert any("payload 不一致" in err for err in result["contractErrors"])
    assert any("contact_ht" in err for err in result["ackErrors"])
    expected = _plan()
    assert result["contact"] == pytest.approx([
        expected["contact_rel_x"], expected["contact_rel_y"],
        expected["contact_rel_z"],
    ])
    assert result["target"] == pytest.approx([
        expected["arm_target_rel_x"], expected["arm_target_rel_y"],
        expected["arm_target_rel_z"],
    ])
    assert (result["id"], result["revision"]) == ("throw42", 3)


def test_report_surfaces_plan_physics_and_disables_new_contract_fallback():
    source = SRC.read_text(encoding="utf-8")
    assert "return structured?null:chassisTargetForThrow" in source
    assert "p.kind==='preaim_hint'" in source
    assert "armPreds.filter(p=>p.kind==='final_hit_plan')" in source
    for field in (
        "collision_en", "collision_kt", "outgoing_vx", "landing_x",
        "apex_z_world", "net_clearance_m", "solve_iterations", "solve_ms",
        "model_fingerprint", "contact_rel_x", "arm_target_rel_x",
        "car_center_x", "arm_forward_offset_m", "tennis_ball_radius_m",
        "racket_contact_vx", "face_normal_nx",
    ):
        assert field in source
    assert "world3d_effective_v3_arm_center_sweetspot:[0-9a-f]{16}" in source
    assert "碰撞 / 求解" in source
