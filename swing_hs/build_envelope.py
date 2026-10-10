# -*- coding: utf-8 -*-
"""从实打场次生成黑标包络的关节采样：swing_hs/envelope_<car>.json。

每拍（/tennis/status 的 `rl_swing done … ht=`）取 [ht − 0.8, ht + 0.5] s 的 /joint_states（控制器 6 槽），
按触球时 J1 分正手（J1@ht > −90°）/ 反手，隔 decimate 条取一条，存成度。开录时
run_swing_hs 按 --side 与 J1 范围过滤这些采样、做 FK 得黑标包络、投到各相机定 ROI。

用法（本机 tracker_output 里有这几场的 _arm.json）：
  python swing_hs/build_envelope.py --car v05 tracker_20261010_095807 tracker_20261010_101015 tracker_20261010_113329
换了臂标定 / 挥拍策略、包络明显变了再重生成；文件随仓库提交。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WINDOW_S = (-0.8, 0.5)
FOREHAND_MIN_J1_DEG = -90.0


def swing_samples(arm: dict, window: tuple[float, float], decimate: int) -> dict[str, list[np.ndarray]]:
    states = arm["states"]
    t = np.asarray([s["t"] for s in states], dtype=np.float64)
    q = np.asarray([s["position"] for s in states], dtype=np.float64)
    out: dict[str, list[np.ndarray]] = {"fore": [], "back": []}
    for event in arm["events"]:
        if event["topic"] != "/tennis/status" or "rl_swing done" not in event["text"]:
            continue
        match = re.search(r"\bht=([0-9.]+)", event["text"])
        if not match:
            continue
        ht = float(match.group(1))
        i = int(np.clip(np.searchsorted(t, ht), 0, len(t) - 1))
        side = "fore" if math.degrees(q[i, 0]) > FOREHAND_MIN_J1_DEG else "back"
        window_q = q[(t >= ht + window[0]) & (t <= ht + window[1])]
        out[side].append(window_q[::decimate])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="tracker_output 下的场次名（要有 <run>_arm.json）")
    ap.add_argument("--car", required=True, choices=["v04", "v05"])
    ap.add_argument("--output-root", default=str(REPO / "tracker_output"))
    ap.add_argument("--decimate", type=int, default=5, help="200 Hz 关节流隔几条取一条（默认 5 = 40 Hz）")
    ap.add_argument("--out", default=None, help="默认 swing_hs/envelope_<car>.json")
    args = ap.parse_args()

    sides: dict[str, list[np.ndarray]] = {"fore": [], "back": []}
    swings = {"fore": 0, "back": 0}
    for run in args.runs:
        arm = json.loads((Path(args.output_root) / run / f"{run}_arm.json").read_text(encoding="utf-8"))
        if str(arm.get("car")) != args.car:
            raise SystemExit(f"{run} 是 {arm.get('car')} 的场次，不是 {args.car}")
        for side, chunks in swing_samples(arm, WINDOW_S, args.decimate).items():
            sides[side].extend(chunks)
            swings[side] += len(chunks)
    doc = {
        "car": args.car,
        "built_from": args.runs,
        "window_s": list(WINDOW_S),
        "decimate": args.decimate,
        "swings": swings,
        "note": "控制器 6 槽关节角（度）；五轴臂槽 4 恒 0。fore = J1@ht > −90°。",
    }
    for side, chunks in sides.items():
        rows = np.degrees(np.vstack(chunks)) if chunks else np.zeros((0, 6))
        doc[side] = [[round(float(v), 2) for v in row] for row in rows]
        if len(rows):
            print(f"{side}: {swings[side]} 拍、{len(rows)} 条采样；J1 {rows[:, 0].min():+.1f}..{rows[:, 0].max():+.1f}°")
    out = Path(args.out) if args.out else Path(__file__).with_name(f"envelope_{args.car}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
