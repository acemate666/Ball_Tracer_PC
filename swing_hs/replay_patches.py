# -*- coding: utf-8 -*-
"""swing_hs 小块回放：录制时存下的每帧小块（cam_<sn>/patches_<sn>.bin）→ 现在的在线黑标候选（marker_yolo）+ 主进程同一个
三角化（MarkerFusion：锚点门、偏差学习、smooth）→ 解出率、几台解、学到的偏差、每块耗时。
换了识别算法 / 模型，先在采集机上拿一场旧录制跑一遍（同一台 CPU 上的真实耗时），再开真录。

    .venv_ros2\\Scripts\\python.exe -m swing_hs.replay_patches tracker_output\\swing_hs_<…> [--limit-slots N]

锚点、车 yaw、帧时间取 marker.jsonl 当时在线算的那一行；像素格式取 cam_<sn>/hs_<sn>.json（录像元数据）。
耗时 = 解码 + 候选（单线程，同识别进程），不含三角化。在线每台相机 DETECT_WORKERS 个识别进程分摊 100 fps，
每块要 < DETECT_WORKERS × 10 ms 才跟得上（还要给认 tag 留余量）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from swing_hs import geometry  # noqa: E402
from swing_hs.camera_proc import DETECT_WORKERS, PATCH_HEADER, bayer_code, marker_candidates_in_patch  # noqa: E402
from swing_hs.marker_solver import MarkerFusion  # noqa: E402
from swing_hs.marker_yolo import MarkerYolo, model_info  # noqa: E402


def read_patches(path: Path):
    """patches_<sn>.bin → (slot, 帧时间, (块左上全幅 x, y), 原始 Bayer 块)。"""
    data = path.read_bytes()
    off = 0
    while off + PATCH_HEADER.size <= len(data):
        slot, _fnum, frame_t, x0, y0, w, h = PATCH_HEADER.unpack_from(data, off)
        off += PATCH_HEADER.size
        yield slot, frame_t, (x0, y0), np.frombuffer(data, dtype=np.uint8, count=w * h, offset=off).reshape(h, w)
        off += w * h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--limit-slots", type=int, default=0, help="只回放前 N 格（0 = 全部）")
    a = ap.parse_args()
    ses = a.session
    meta = json.loads((ses / "meta.json").read_text(encoding="utf-8"))
    serials = list(meta["config"]["serials"])
    calib = Path(meta["calibration"])
    if not calib.is_file():     # 别的机器录的：同名标定在本仓库
        calib = REPO / "src" / "config" / Path(str(meta["calibration"]).replace("\\", "/")).name
    rbm = geometry.load_test_src("racket_ht_black_marker")
    cameras = rbm._load_cameras(calib, serials)
    codes = {}
    for s in serials:
        hs = json.loads((ses / f"cam_{s}" / f"hs_{s}.json").read_text(encoding="utf-8"))
        codes[s] = bayer_code(int(hs["recording"]["pixel_type"]))
    rows = {}
    with open(ses / "marker.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("anchor_mm") is not None:
                rows[r["slot"]] = r
    detector = MarkerYolo(threads=1)
    cands: dict[int, dict] = defaultdict(dict)
    ms: list[float] = []
    for s in serials:
        for slot, _t, origin, block in read_patches(ses / f"cam_{s}" / f"patches_{s}.bin"):
            if a.limit_slots and slot >= a.limit_slots:
                break
            r = rows.get(slot)
            if r is None:
                continue
            anchor_uv = geometry.project_m(cameras[s], np.asarray(r["anchor_mm"]) / 1000.0)[0]
            t0 = time.perf_counter()
            cands[slot][s] = marker_candidates_in_patch(detector, block, origin, (float(anchor_uv[0]), float(anchor_uv[1])),
                                                        codes[s])
            ms.append(1e3 * (time.perf_counter() - t0))
    fusion = MarkerFusion(cameras, serials)
    solved, smooth = 0, 0
    by_cams: dict[int, int] = defaultdict(int)
    online = sum(1 for slot in cands if rows[slot].get("fit"))
    for slot in sorted(cands):
        r = rows[slot]
        if r.get("car") is None:
            continue
        sol, _ = fusion.solve_slot(np.asarray(r["anchor_mm"], dtype=np.float64), float(r["car"][2]),
                                   {s: v for s, v in cands[slot].items() if v}, float(r["t"]))
        if sol is not None:
            solved += 1
            by_cams[len(sol.serials)] += 1
            smooth += sol.smooth is True
    t = np.asarray(ms)
    info = model_info()
    print(f"模型 {info['model']}（{info.get('version', '?')}，sha256 {info['sha256'][:12]}）；场次 {ses.name}")
    print(f"格 {len(cands)}：解出 {solved}（{100 * solved / max(1, len(cands)):.1f}%）{dict(sorted(by_cams.items()))}，"
          f"smooth {smooth}；录制时在线解出 {online}（{100 * online / max(1, len(cands)):.1f}%）")
    print(f"学到的「黑标 − FK」偏差 {None if fusion.offset.value is None else np.round(fusion.offset.value, 1).tolist()} mm"
          f"（录制时 {rows[max(rows)].get('offset_car_mm')}）")
    print(f"每块（解码 + 候选，单线程）{len(t)} 块：中位 {np.median(t):.2f} ms、p99 {np.percentile(t, 99):.2f}、最大 {t.max():.1f}；"
          f"在线每台 {DETECT_WORKERS} 个识别进程，100 fps 下每块预算 {DETECT_WORKERS * 10} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
