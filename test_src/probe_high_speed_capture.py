"""不起 tracker、只开相机：按高速配置跑 N 秒，验证同拍 / 丢帧 / 写盘，并存样图。

    python test_src/probe_high_speed_capture.py --seconds 20
    python test_src/probe_high_speed_capture.py --camera-config src/config/camera_18_hs405.json \
        --seconds 60 --busy-threads 4      # 加几条纯 Python 忙线程，看抢 GIL 时触发环迟到多少

输出到 --out（默认 tracker_output/hs_probe_<时间>）：
  probe_hs_<serial>.{bayer,idx,json}   高速相机录像（读法见 test_src/hs_recording.py）
  sample_<serial>.jpg                  同一拍的各相机画面（高速相机是贴回全幅的画布）
  sample_<serial>_roi.jpg              高速相机同拍那一帧的 ROI 原图
  probe_report.json                    各项统计
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.ball_grabber import SyncCapture, frame_to_numpy  # noqa: E402
from hs_recording import HighSpeedRecording  # noqa: E402


def _busy(stop: threading.Event) -> None:
    x = 0
    while not stop.is_set():
        for i in range(10000):
            x += i * i


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-config", default=str(_REPO / "src" / "config" / "camera_18_hs405.json"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--exposure-us", type=float, default=None)
    ap.add_argument("--busy-threads", type=int, default=0)
    ap.add_argument("--switch-interval", type=float, default=None,
                    help="sys.setswitchinterval（秒）；默认 Python 是 0.005")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else _REPO / "tracker_output" / f"hs_probe_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"exposure_us": float(args.exposure_us)} if args.exposure_us is not None else {}

    stop_busy = threading.Event()
    busy = [threading.Thread(target=_busy, args=(stop_busy,), daemon=True) for _ in range(args.busy_threads)]
    for t in busy:
        t.start()

    sets: list[dict[str, float]] = []
    sample = None
    with SyncCapture.from_config(args.camera_config, **overrides) as cap:
        hs = cap.high_speed_serial
        if hs is None:
            print("相机配置里没有 high_speed_camera")
            return 2
        time.sleep(1.0)
        recorder = cap.start_high_speed_recording(out / f"probe_hs_{hs}")
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.seconds:
            frames = cap.get_frames(timeout_s=0.2)
            if frames is None:
                continue
            sets.append({sn: f.exposure_start_pc for sn, f in frames.items()} | {"_hs_frame_num": frames[hs].frame_num})
            if sample is None and len(sets) >= 40:
                sample = frames
        duration = time.perf_counter() - t_start
        info = cap.high_speed_info()
    stop_busy.set()
    stats = cap.stats()

    serials = [sn for sn in sets[0] if not sn.startswith("_")] if sets else []
    slow = [sn for sn in serials if sn != hs]
    if len(sets) < 2:
        print(f"只拼出 {len(sets)} 组同步帧，取帧链被卡住了；看下面的 trigger / per_camera 统计")
    report: dict = {
        "camera_config": args.camera_config,
        "seconds": round(duration, 3),
        "busy_threads": args.busy_threads,
        "switch_interval_s": sys.getswitchinterval(),
        "high_speed": info,
        "capture": stats,
        "sets": len(sets),
        "sets_fps": round(len(sets) / duration, 3) if duration > 0 else None,
    }
    if len(sets) >= 2:
        t_sets = np.array([np.mean([s[sn] for sn in slow]) for s in sets])
        gaps = np.diff(t_sets)
        report["set_gap_ms"] = {
            "median": round(float(np.median(gaps)) * 1000, 3),
            "p99": round(float(np.percentile(gaps, 99)) * 1000, 3),
            "over_1_5x": int(np.sum(gaps > 1.5 / cap.fps)),
        }
        hs_dt = np.array([s[hs] - np.mean([s[sn] for sn in slow]) for s in sets if s["_hs_frame_num"] >= 0])
        report["hs_vs_slow_exposure_ms"] = {
            "mean": round(float(np.mean(hs_dt)) * 1000, 4) if len(hs_dt) else None,
            "max_abs": round(float(np.max(np.abs(hs_dt))) * 1000, 4) if len(hs_dt) else None,
            "black_placeholders": int(sum(1 for s in sets if s["_hs_frame_num"] < 0)),
        }
        slow_spread = np.array([max(s[sn] for sn in slow) - min(s[sn] for sn in slow) for s in sets])
        report["slow_exposure_spread_ms_max"] = round(float(np.max(slow_spread)) * 1000, 4)
    rec = HighSpeedRecording.open(recorder.json_path)
    report["recording"] = rec.summary()
    per_cam = stats["per_camera"]
    report["per_camera_fps"] = {sn: round(v["received"] / duration, 2) for sn, v in per_cam.items()}

    if sample is not None:
        for sn, f in sample.items():
            img = frame_to_numpy(f)
            cv2.imwrite(str(out / f"sample_{sn}.jpg"), cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        i = rec.nearest(sample[hs].exposure_start_pc)
        cv2.imwrite(str(out / f"sample_{hs}_roi.jpg"), rec.bgr(i))

    (out / "probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in report if k not in ("high_speed", "capture")},
                     ensure_ascii=False, indent=2))
    print("trigger:", json.dumps(stats.get("trigger"), ensure_ascii=False))
    print("high_speed:", json.dumps(stats.get("high_speed"), ensure_ascii=False))
    print("per_camera:", json.dumps(stats.get("per_camera"), ensure_ascii=False))
    print("assembly:", json.dumps({k: stats.get(k) for k in (
        "exposure_spread_discards", "arrival_spread_discards", "assembly_timeouts")}, ensure_ascii=False))
    print(f"输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
