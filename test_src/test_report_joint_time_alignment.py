"""/joint_states 关节时刻对齐（generate_curve3_html._time_align_joint_states）。

合成一段 arm_controller_cpp 两相调度的 /joint_states：tick 5ms，相表逐项抄臂端 tx_phase——
v0.4 two_phase_j1_j5：A 相 J1,J2,J3,J5,J4、B 相 J1,J5,J6；v0.5（五轴，槽 4 空）1010 前 two_phase_j1：
A 相 J1..J4、B 相 J1,滚转（槽 5），1010 起 two_phase_j1_j4：A 相 J1..J4,滚转、B 相 J1..J4。帧间 g=0.52ms，
每帧反馈 0.15ms 后到达，header.stamp=最新一帧到达。各轴按匀速运动生成真值，快照里非本相轴保留旧值
（正如节点的 B 相快照里 J2-J4 是 5ms 前的值）；空槽恒 0（extract_arm_bag 就这么写）。对齐后每行
position 应等于该行戳时刻的真值；调度从数据认（不读当前 yaml），认不出两相结构时必须原样不动。
"""
import copy
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent


def _load():
    sys.path.insert(0, str(SRC))
    try:
        from generate_curve3_html import _time_align_joint_states, _tx_schedules, _REPLY_FRAME_MS
    finally:
        sys.path.pop(0)
    return _time_align_joint_states, _tx_schedules, _REPLY_FRAME_MS


def _table(car, schedule=None):
    _, tx_schedules, _ = _load()
    tables, _ = tx_schedules(car)
    return tables[schedule or next(iter(tables))]


def _synth(order, n_ticks=400, g_ms=0.52, reply_ms=0.15, tick_ms=5.0, t0=1000.0):
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


def _synth_two_phase(car="v04", schedule=None, **kw):
    return _synth(_table(car, schedule), **kw)


def _assert_rows_match_truth(rows, truth, vel_truth, effort):
    # 中段（首尾各留几行给插值找邻居）逐行逐轴对齐到真值；空槽仍是 0
    for row in rows[5:-5]:
        for j in range(6):
            assert row["position"][j] == pytest.approx(truth(j, row["t"]), abs=2e-4), (j, row["t"])
            assert row["velocity"][j] == pytest.approx(vel_truth(j, row["t"]), abs=2e-4), (j, row["t"])
        assert row["effort"] == effort


@pytest.mark.parametrize("car,schedule,frames_a,frames_b,a_last,b_last", [
    ("v04", "two_phase_j1_j5", 5, 3, "J4", "J6"),
    ("v05", "two_phase_j1", 4, 2, "J4", "J5"),       # v0.5 的滚转叫 joint5，在槽 5
    ("v05", "two_phase_j1_j4", 5, 4, "J5", "J4"),    # 1010 起：J1–J4 每拍、滚转只在 A 相（排最后）
])
def test_two_phase_rows_are_aligned_to_stamp(car, schedule, frames_a, frames_b, a_last, b_last):
    align_fn, _, _ = _load()
    rows, truth, vel_truth = _synth_two_phase(car, schedule)
    effort = copy.deepcopy(rows[0]["effort"])
    arm = {"states": rows}
    info = align_fn(arm, car)
    assert info is not None and info["schedule"] == schedule
    assert info["tx_gap_ms"] == pytest.approx(0.52, abs=0.01)
    assert info["phase_rows"]["unknown"] <= 1
    assert [s["schedule"] for s in info["segments"]] == [schedule]
    _assert_rows_match_truth(rows, truth, vel_truth, effort)
    lag = info["sample_lag_before_stamp_ms"]
    assert lag["A"]["J1"] == pytest.approx((frames_a - 1) * 0.52 + 0.15, abs=0.05)
    assert lag["B"]["J1"] == pytest.approx((frames_b - 1) * 0.52 + 0.15, abs=0.05)
    assert lag["A"][a_last] == pytest.approx(0.15, abs=0.01) and lag["B"][b_last] == pytest.approx(0.15, abs=0.01)
    assert len(lag["A"]) == frames_a and len(lag["B"]) == frames_b
    assert schedule in info["summary"]


def test_schedule_change_between_arm_runs_is_aligned_per_segment():
    """同一场里臂重启换了调度（1010：two_phase_j1 → two_phase_j1_j4）：按 /joint_states 断档分段，各段按自己的调度对齐。"""
    align_fn, _, _ = _load()
    rows1, truth1, vel1 = _synth_two_phase("v05", "two_phase_j1", t0=1000.0)
    rows2, truth2, vel2 = _synth_two_phase("v05", "two_phase_j1_j4", t0=1005.0)   # 断开 3s 后重起
    effort = copy.deepcopy(rows1[0]["effort"])
    arm = {"states": rows1 + rows2}
    info = align_fn(arm, "v05")
    assert info is not None and info["schedule"] == "two_phase_j1+two_phase_j1_j4"
    assert [(s["schedule"], s["rows"]) for s in info["segments"]] == [
        ("two_phase_j1", 400), ("two_phase_j1_j4", 400)]
    _assert_rows_match_truth(rows1, truth1, vel1, effort)
    _assert_rows_match_truth(rows2, truth2, vel2, effort)


def test_v05_rows_need_the_v05_phase_table():
    """v0.5 的包按 v0.4 相表对齐：两相间隔规律恰好一样（A、B 都差 2 帧），照样判成两相、照样对齐，
    但 J1-J3 的采样时刻会多挪一帧（~0.5ms）——所以相表必须按本场车型取。"""
    align_fn, _, _ = _load()
    rows, truth, _ = _synth_two_phase("v05", "two_phase_j1")
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


def test_non_two_phase_data_is_left_untouched(capsys):
    align_fn, _, _ = _load()
    # 单相 100Hz 节点：每 10ms 六轴全刷新
    rows = [{"t": 5.0 + 0.01 * k, "position": [0.1 * k] * 6, "velocity": [1.0] * 6, "effort": [0.0] * 6}
            for k in range(600)]
    arm = {"states": copy.deepcopy(rows)}
    assert align_fn(arm, "v04") is None
    assert arm["states"] == rows
    # 全轴 200Hz（full，每拍五帧 J1,J2,J3,滚转,J4）：没有两相结构，不动，但要提示"200Hz 命令流认不出调度"
    full = {"A": (0, 1, 2, 5, 3), "B": (0, 1, 2, 5, 3)}
    rows_full, _, _ = _synth(full)
    arm5 = {"states": copy.deepcopy(rows_full)}
    capsys.readouterr()
    assert align_fn(arm5, "v05") is None
    assert arm5["states"] == rows_full
    assert "认不出调度" in capsys.readouterr().err
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
