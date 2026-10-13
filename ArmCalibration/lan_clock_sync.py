#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASUS ↔ Jetson 局域网双向对时（用户 2026-10-06 定）：以网球对时为真值，量局域网对时相对真值的固定偏差。

用户原话：「让 jetson 和 asus 互相发一下对时信息（把基于网球的对时当作真值，通过局域网通信的对时来得到
他相对真值的 offset）。保存好这个值后，我们以后都可以用局域网通信来做对时了」。

两边各跑一个，Jetson 端只是 UDP 应答，不碰臂栈 / bot / 视觉：
  Jetson（RK 轴 = CLOCK_MONOTONIC，bot_center / arm_controller 的 steady_clock 就是它）：
      python3 lan_clock_sync.py responder [--port 47123]
  ASUS（PC 轴 = perf_counter，tracker 与专场录像的时间轴）：
      python lan_clock_sync.py initiator --peer 192.168.50.110 [--rate 20] [--out <csv>]

每次交换记四个时刻（同 NTP）：t1 = PC 发出，t2 = Jetson 收到，t3 = Jetson 回发，t4 = PC 收到。
  rtt          = (t4 − t1) − (t3 − t2)
  pc_minus_rk  = ((t1 − t2) + (t4 − t3)) / 2
pc_minus_rk 与 tracker 网球在线对时、/pc_rk_time_offset 同号同义：PC perf ≈ RK t + pc_minus_rk。
去程、回程不等长时，这个估计偏 (回程 − 去程)/2，这一项正是要拿网球真值标出来的偏差。
Wi-Fi 时延抖得厉害，所以只信往返最短的那部分样本（见 robust_offset）。

initiator 逐条写 CSV（seq,t1_pc,t2_rk,t3_rk,t4_pc），每 10 s 打一行最近 30 s 的稳健偏移与往返统计。
停止：Ctrl+C、--duration 到点，或建停止文件 <out 所在目录>/.stop_lan_sync（脱离 ssh 起的时候用）。

用法（用户 1006 定的公式）：
  差值 = 网球对时 offset − 局域网 offset（开场 tracker 40 fps 扔球时两者同时测，多抛取中位）
  ASUS 相机曝光时间 − Jetson 时间 = 局域网 offset（要对的那一刻）+ 差值
  python lan_clock_sync.py bias --lan-csv <csv> --tracker-json <tracker_*.json> [--save <json>]
1006 首测（18F，两头都走 Wi-Fi）：往返最短 2.6 ms、中位 5.4 ms，2 分钟收发 2400 条丢 1 条。
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

DEFAULT_PORT = 47123
PING = struct.Struct("<qd")        # seq, t1
PONG = struct.Struct("<qddd")      # seq, t1, t2, t3
STOP_FILE_NAME = ".stop_lan_sync"


def exchange_values(t1: float, t2: float, t3: float, t4: float) -> tuple[float, float]:
    """(rtt, pc_minus_rk)，单位秒。"""
    return (t4 - t1) - (t3 - t2), 0.5 * ((t1 - t2) + (t4 - t3))


def load_csv(path: str | Path) -> list[tuple[float, float, float, float]]:
    """initiator 写的 CSV → [(t1, t2, t3, t4)]。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 5:
                rows.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def offset_at(rows: list[tuple[float, float, float, float]], t_pc: float, *, half_window_s: float = 5.0,
              keep_fraction: float = 0.2) -> dict | None:
    """PC 时刻 t_pc 那一刻的局域网偏移：t_pc 前后 half_window_s 内往返最短的 keep_fraction 条，取偏移中位数。

    用的时候每个时刻都取自己那一刻的值（录像全程开着局域网对时），不做漂移估计。
    """
    sel = [r for r in rows if abs(r[0] - t_pc) <= half_window_s]
    est = robust_offset(sel, keep_fraction=keep_fraction)
    if est is None:
        return None
    return {"pc_minus_rk": est["pc_minus_rk"], "n": est["n"], "kept": est["kept"],
            "kept_spread_ms": est["kept_spread"] * 1e3, "rtt_min_ms": est["rtt_min"] * 1e3,
            "rtt_p50_ms": est["rtt_p50"] * 1e3}


def robust_offset(rows: list[tuple[float, float, float, float]], *, keep_fraction: float = 0.2,
                  min_keep: int = 5) -> dict | None:
    """一段样本的稳健偏移：只取往返最短的 keep_fraction（至少 min_keep 条），取它们 pc_minus_rk 的中位数。

    往返最短的那些交换两程都没排队，偏差只剩两程固有的不对称；均值会被 Wi-Fi 重传 / 省电唤醒拖走。
    """
    vals = [exchange_values(*r) for r in rows]
    vals = [v for v in vals if math.isfinite(v[0]) and v[0] > 0]
    if len(vals) < min_keep:
        return None
    vals.sort(key=lambda v: v[0])
    k = max(min_keep, int(len(vals) * keep_fraction))
    kept = sorted(v[1] for v in vals[:k])
    rtts = [v[0] for v in vals]
    return {
        "n": len(vals),
        "kept": k,
        "pc_minus_rk": kept[len(kept) // 2],
        "kept_spread": kept[-1] - kept[0],
        "rtt_min": rtts[0],
        "rtt_p50": rtts[len(rtts) // 2],
        "rtt_kept_max": rtts[k - 1],
    }


# ───────────────── Jetson 端 ─────────────────


def run_responder(port: int) -> int:
    mono = time.monotonic            # Linux 上 = CLOCK_MONOTONIC = bot_center / 臂的 steady_clock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"[lan_clock_sync] responder 监听 UDP {port}，时钟 = time.monotonic (CLOCK_MONOTONIC)", flush=True)
    n = 0
    while True:
        data, addr = sock.recvfrom(64)
        t2 = mono()
        if len(data) != PING.size:
            continue
        seq, t1 = PING.unpack(data)
        t3 = mono()
        sock.sendto(PONG.pack(seq, t1, t2, t3), addr)
        n += 1
        if n == 1 or n % 6000 == 0:
            print(f"[lan_clock_sync] 已应答 {n} 次（最近来自 {addr[0]}）", flush=True)


# ───────────────── ASUS 端 ─────────────────


def run_initiator(peer: str, port: int, rate: float, out: Path, duration: float) -> int:
    clock = time.perf_counter        # Windows 上 = QueryPerformanceCounter = tracker / 录像的 PC 轴
    out.parent.mkdir(parents=True, exist_ok=True)
    stop_file = out.parent / STOP_FILE_NAME
    if stop_file.exists():
        stop_file.unlink()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))
    sock.settimeout(0.5)
    rows: list[tuple[float, float, float, float]] = []
    lock = threading.Lock()
    stop = threading.Event()
    sent = [0]
    bad = [0]

    def receiver(f) -> None:
        while not stop.is_set():
            try:
                data = sock.recv(64)
            except socket.timeout:
                continue
            except OSError:
                if stop.is_set():
                    return
                continue
            t4 = clock()
            if len(data) != PONG.size:
                bad[0] += 1
                continue
            seq, t1, t2, t3 = PONG.unpack(data)
            f.write(f"{seq},{t1:.9f},{t2:.9f},{t3:.9f},{t4:.9f}\n")
            with lock:
                rows.append((t1, t2, t3, t4))

    period = 1.0 / rate
    print(f"[lan_clock_sync] initiator → {peer}:{port}，{rate:g} Hz，CSV {out}", flush=True)
    print(f"[lan_clock_sync] 停止：Ctrl+C 或创建 {stop_file}", flush=True)
    t_start = clock()
    with open(out, "w", encoding="utf-8", newline="\n", buffering=1 << 16) as f:
        f.write("seq,t1_pc,t2_rk,t3_rk,t4_pc\n")
        rx = threading.Thread(target=receiver, args=(f,), name="lan-sync-rx", daemon=True)
        rx.start()
        next_t = t_start
        last_report = t_start
        seq = 0
        try:
            while True:
                now = clock()
                if now - t_start >= duration or stop_file.exists():
                    break
                if now < next_t:
                    time.sleep(min(next_t - now, 0.05))
                    continue
                next_t += period
                if next_t < now:                     # 落后太多就别补发一串
                    next_t = now + period
                seq += 1
                t1 = clock()
                try:
                    sock.sendto(PING.pack(seq, t1), (peer, port))
                    sent[0] += 1
                except OSError as exc:
                    print(f"[lan_clock_sync] 发送失败: {exc}", flush=True)
                if now - last_report >= 10.0:
                    last_report = now
                    f.flush()
                    with lock:
                        recent = [r for r in rows if r[0] >= now - 30.0]
                        n_all = len(rows)
                    est = robust_offset(recent)
                    if est is None:
                        print(f"[{now - t_start:7.1f}s] 发 {sent[0]} 收 {n_all}：最近 30 s 样本不够", flush=True)
                    else:
                        print(f"[{now - t_start:7.1f}s] 发 {sent[0]} 收 {n_all} | 30 s: pc_minus_rk "
                              f"{est['pc_minus_rk']:+.6f} s（往返最短 {est['kept']}/{est['n']} 条，散布 "
                              f"{est['kept_spread'] * 1e3:.3f} ms）rtt min {est['rtt_min'] * 1e3:.2f} / "
                              f"p50 {est['rtt_p50'] * 1e3:.2f} ms", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            rx.join(timeout=2.0)
            sock.close()
    with lock:
        est = robust_offset(rows)
    print(f"[lan_clock_sync] 结束：发 {sent[0]}、收 {len(rows)}、坏包 {bad[0]}", flush=True)
    if est is not None:
        print(f"[lan_clock_sync] 全程 pc_minus_rk {est['pc_minus_rk']:+.6f} s，rtt min {est['rtt_min'] * 1e3:.2f} ms", flush=True)
    if stop_file.exists():
        stop_file.unlink()
    return 0


# ───────────────── 离线：局域网对时 vs 网球真值 ─────────────────


def run_bias(lan_csvs: list[str], tracker_jsons: list[str], save: str | None) -> int:
    """逐次比较 tracker 网球在线对时（accepted，真值）与同一 PC 时刻的局域网偏移：差值 = 网球 − 局域网。

    以后的用法：pc_minus_rk = 那一刻的局域网偏移（offset_at）+ 差值。
    网球单次对时散布 ~±2 ms（181235：8 次通过的跨 3.8 ms），差值取多抛中位。
    """
    import json

    import numpy as np

    rows: list[tuple[float, float, float, float]] = []
    for p in lan_csvs:
        rows += load_csv(p)
    rows.sort()
    found = []
    for tj in tracker_jsons:
        with open(tj, "r", encoding="utf-8") as f:
            doc = json.load(f)
        align = (doc.get("config") or {}).get("rk_time_alignment") or {}
        for h in align.get("history") or []:
            if not h.get("accepted") or h.get("pc_bounce_t") is None or h.get("pc_minus_rk") is None:
                continue
            est = offset_at(rows, float(h["pc_bounce_t"]))
            if est is None:
                continue
            found.append({"session": Path(tj).stem, "update": h.get("update"), "pc_t": float(h["pc_bounce_t"]),
                          "ball": float(h["pc_minus_rk"]), "lan": est["pc_minus_rk"],
                          "bias_ms": (float(h["pc_minus_rk"]) - est["pc_minus_rk"]) * 1e3,
                          "ball_err_m": h.get("err"), "lan_kept_spread_ms": est["kept_spread_ms"],
                          "lan_rtt_p50_ms": est["rtt_p50_ms"]})
    if not found:
        print("没有能配上的网球对时（局域网 CSV 是否覆盖了 tracker 这几场的时段？）")
        return 1
    print(f"{'session':28s} {'#':>3s} {'PC t':>14s} {'网球−局域网 ms':>14s} {'网球 err m':>10s} {'LAN 散布 ms':>11s}")
    for r in found:
        print(f"{r['session']:28s} {r['update']!s:>3s} {r['pc_t']:14.3f} {r['bias_ms']:+14.3f} "
              f"{(r['ball_err_m'] or float('nan')):10.4f} {r['lan_kept_spread_ms']:11.3f}")
    b = np.array([r["bias_ms"] for r in found])
    summary = {"n": int(len(b)), "bias_ms_median": float(np.median(b)), "bias_ms_mean": float(b.mean()),
               "bias_ms_std": float(b.std(ddof=1)) if len(b) > 1 else None,
               "bias_ms_sem": float(b.std(ddof=1) / math.sqrt(len(b))) if len(b) > 1 else None}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if save:
        doc = {
            "contract": "pc_minus_rk_s = 那一刻的 lan_offset + bias_s（bias = 网球 − 局域网，应为常数）；lan_offset 按 lan_clock_sync.offset_at（前后 5 s、往返最短 20% 取中位）",
            "bias_s": summary["bias_ms_median"] / 1e3,
            "summary": summary,
            "throws": found,
            "lan_csvs": [str(p) for p in lan_csvs],
            "tracker_jsons": [str(p) for p in tracker_jsons],
        }
        with open(save, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"已存 {save}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="role", required=True)
    b = sub.add_parser("bias", help="离线：网球对时（真值）− 局域网对时 = 偏差")
    b.add_argument("--lan-csv", nargs="+", required=True)
    b.add_argument("--tracker-json", nargs="+", required=True, help="同时段 tracker 场次的 tracker_*.json")
    b.add_argument("--save", default=None, help="把偏差存成 JSON")
    r = sub.add_parser("responder", help="Jetson 端：UDP 应答")
    r.add_argument("--port", type=int, default=DEFAULT_PORT)
    i = sub.add_parser("initiator", help="ASUS 端：按固定频率发对时包、逐条记 CSV")
    i.add_argument("--peer", required=True, help="Jetson IP（18F：192.168.50.110）")
    i.add_argument("--port", type=int, default=DEFAULT_PORT)
    i.add_argument("--rate", type=float, default=20.0, help="每秒发几次")
    i.add_argument("--duration", type=float, default=4 * 3600.0, help="最长跑多久（秒）")
    i.add_argument("--out", default=None, help="CSV 路径（默认 tracker_output/lan_sync_<时间>.csv）")
    args = ap.parse_args()
    if args.role == "responder":
        return run_responder(args.port)
    if args.role == "bias":
        return run_bias(args.lan_csv, args.tracker_json, args.save)
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "tracker_output"
        / f"lan_sync_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return run_initiator(args.peer, args.port, args.rate, out, args.duration)


if __name__ == "__main__":
    sys.exit(main())
