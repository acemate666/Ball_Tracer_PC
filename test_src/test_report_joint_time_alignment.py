"""/joint_states 关节时刻对齐（generate_curve3_html._time_align_joint_states）。

合成一段 arm_controller_cpp 两相调度的 /joint_states：tick 5ms，相表按车取（臂车型 yaml 的
control.tx_schedule）——v0.4 A 相 J1,J2,J3,J5,J4、B 相 J1,J5,J6；v0.5（五轴，槽 4 空）A 相 J1..J4、
B 相 J1,滚转（槽 5）。帧间 g=0.52ms，每帧反馈 0.15ms 后到达，header.stamp=最新一帧到达。各轴按匀速
运动生成真值，快照里非本相轴保留旧值（正如节点的 B 相快照里 J2-J4 是 5ms 前的值）；空槽恒 0
（extract_arm_bag 就这么写）。对齐后每行 position 应等于该行戳时刻的真值；识别不出两相结构、
或本车不是两相调度时必须原样不动。
"""
import copy
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent


def _load():
    sys.path.insert(0, str(SRC))
    try:
        from generate_curve3_html import _time_align_joint_states, _tx_phase_order, _REPLY_FRAME_MS
    finally:
        sys.path.pop(0)
    return _time_align_joint_states, _tx_phase_order, _REPLY_FRAME_MS


def _synth_two_phase(car="v04", n_ticks=400, g_ms=0.52, reply_ms=0.15, tick_ms=5.0, t0=1000.0):
    _, phase_order, _ = _load()
    _, order, _ = phase_order(car)
    present = set(order["A"]) | set(order["B"])
    on = [1.0 if j in present else 0.0 for j in range(6)]   # 空槽恒 0
    w = [v * s for v, s in zip([3.0, -0.8, 0.6, 0.4, -2.5, 1.7], on)]   # rad/s，各轴匀速
    a = [v * s for v, s in zip([-0.5, 0.3, 1.5, 1.8, 0.0, 0.3], on)]
    wd = [v * s for v, s in zip([0.2, -0.1, 0.05, 0.0, -0.3, 0.1], on)]  # velocity 也给个线性变化，验证同样被插值
    eff = [v * s for v, s in zip([10.0, -5.0, 2.0, 1.0, 0.5, 0.1], on)]
    truth = lambda j, t: a[j] + w[j] * (t - t0)
    vel_truth = lambda j, t: w[j] + wd[j] * (t - t0)
    last = [truth(j, t0) for j in range(6)]
    last_v = [vel_truth(j, t0) for j in range(6)]
    rows = []
    for k in range(n_ticks):
        phase = "A" if k % 2 == 0 else "B"
        seq = order[phase]
        tick = t0 + k * tick_ms * 1e-3
        stamp = None
        for i, j in enumerate(seq):
            ts = tick + i * g_ms * 1e-3            # 命令发出即取样（简化），反馈 reply_ms 后到达
            last[j] = truth(j, ts)
            last_v[j] = vel_truth(j, ts)
            stamp = ts + reply_ms * 1e-3
        rows.append({"t": stamp, "position": [round(v, 5) for v in last],
                     "velocity": [round(v, 5) for v in last_v], "effort": list(eff)})
    return rows, truth, vel_truth


@pytest.mark.parametrize("car,schedule,frames_a,frames_b,b_last", [
    ("v04", "two_phase_j1_j5", 5, 3, "J6"),
    ("v05", "two_phase_j1", 4, 2, "J5"),     # v0.5 的滚转叫 joint5，在槽 5
])
def test_two_phase_rows_are_aligned_to_stamp(car, schedule, frames_a, frames_b, b_last):
    align_fn, _, _ = _load()
    rows, truth, vel_truth = _synth_two_phase(car)
    effort = copy.deepcopy(rows[0]["effort"])
    arm = {"states": rows}
    info = align_fn(arm, car)
    assert info is not None and info["schedule"] == schedule
    assert info["tx_gap_ms"] == pytest.approx(0.52, abs=0.01)
    assert info["phase_rows"]["unknown"] <= 1
    # 中段（首尾各留几行给插值找邻居）逐行逐轴对齐到真值：J1 200Hz、J2-J4 100Hz（B 相原本是旧值）、槽 5 100Hz；空槽仍是 0
    for row in rows[5:-5]:
        for j in range(6):
            assert row["position"][j] == pytest.approx(truth(j, row["t"]), abs=2e-4), (j, row["t"])
            assert row["velocity"][j] == pytest.approx(vel_truth(j, row["t"]), abs=2e-4), (j, row["t"])
        assert row["effort"] == effort
    lag = info["sample_lag_before_stamp_ms"]
    assert lag["A"]["J1"] == pytest.approx((frames_a - 1) * 0.52 + 0.15, abs=0.05)
    assert lag["B"]["J1"] == pytest.approx((frames_b - 1) * 0.52 + 0.15, abs=0.05)
    assert lag["A"]["J4"] == pytest.approx(0.15, abs=0.01) and lag["B"][b_last] == pytest.approx(0.15, abs=0.01)
    assert len(lag["A"]) == frames_a and len(lag["B"]) == frames_b


def test_v05_rows_need_the_v05_phase_table():
    """v0.5 的包按 v0.4 相表对齐：两相间隔规律恰好一样（A、B 都差 2 帧），照样判成两相、照样对齐，
    但 J1-J3 的采样时刻会多挪一帧（~0.5ms）——所以相表必须按本场车型取。"""
    align_fn, _, _ = _load()
    rows, truth, _ = _synth_two_phase("v05")
    wrong = {"states": copy.deepcopy(rows)}
    assert align_fn(wrong, "v04") is not None
    worst_wrong = max(abs(r["position"][0] - truth(0, r["t"])) for r in wrong["states"][5:-5])
    assert worst_wrong > 0.5 * 3.0 * 0.52e-3      # |w1|=3 rad/s × 一帧 0.52ms
    right = {"states": copy.deepcopy(rows)}
    assert align_fn(right, "v05") is not None
    worst_right = max(abs(r["position"][0] - truth(0, r["t"])) for r in right["states"][5:-5])
    assert worst_right < 2e-4


def test_raw_rows_were_actually_wrong_before_alignment():
    """对照：不对齐时 B 相快照里的 J2 与真值差一整拍（w2·~6ms），J1 差 ~2ms——这就是本函数存在的理由。"""
    rows, truth, _ = _synth_two_phase()
    worst_j2 = max(abs(r["position"][1] - truth(1, r["t"])) for r in rows[5:-5])
    worst_j1 = max(abs(r["position"][0] - truth(0, r["t"])) for r in rows[5:-5])
    assert worst_j2 > 0.8 * 5.5e-3      # |w2|=0.8 rad/s × ≥5.5ms
    assert worst_j1 > 3.0 * 2.0e-3      # |w1|=3.0 rad/s × ≥2ms


def test_non_two_phase_data_is_left_untouched():
    align_fn, _, _ = _load()
    # 单相 100Hz 节点：每 10ms 六轴全刷新
    rows = [{"t": 5.0 + 0.01 * k, "position": [0.1 * k] * 6, "velocity": [1.0] * 6, "effort": [0.0] * 6}
            for k in range(600)]
    arm = {"states": copy.deepcopy(rows)}
    assert align_fn(arm, "v04") is None
    assert arm["states"] == rows
    # 行数太少
    arm2 = {"states": copy.deepcopy(rows[:50])}
    assert align_fn(arm2, "v04") is None
    assert arm2["states"] == rows[:50]
    # v0.3（单相节点、臂端已无车型 yaml）、未知车型：两相数据也不动
    rows_v04, _, _ = _synth_two_phase("v04")
    for car in ("v03", None, "v99"):
        arm4 = {"states": copy.deepcopy(rows_v04)}
        assert align_fn(arm4, car) is None
        assert arm4["states"] == rows_v04
    # 关节残缺的行不参与也不报错
    rows_bad, _, _ = _synth_two_phase("v04", n_ticks=300)
    rows_bad[10]["position"] = [0.1, None, 0.3, 0.15, -0.4, 0.25]
    rows_bad[11]["position"] = rows_bad[11]["position"][:5]
    rows_bad[12].pop("velocity")
    arm3 = {"states": rows_bad}
    info = align_fn(arm3, "v04")
    assert info is not None
    assert arm3["states"][10]["position"] == [0.1, None, 0.3, 0.15, -0.4, 0.25]
    assert len(arm3["states"][11]["position"]) == 5
    assert "velocity" not in arm3["states"][12]
