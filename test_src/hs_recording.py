"""读高速相机录像（src/ball_grabber.py 的 HighSpeedRecorder 产物）。

一场里是三个文件，<base> = tracker_output/<run>/<run>_hs_<serial>：
  <base>.json   ROI / 画布 / 倍频 / 实测帧率上限 / 收尾统计 / 同拍帧号
  <base>.idx    逐帧 40 字节记录（frame_num, dev_timestamp, exposure_start_pc, arrival_perf, lost_packet）
  <base>.bayer  逐帧原始 8-bit Bayer，首尾相接，第 i 帧在 i*W*H

用法：
    rec = HighSpeedRecording.open(base)
    rec.index["exposure_start_pc"]           # 逐帧曝光时刻（PC perf 轴，与 session json 同轴）
    img = rec.bgr(i)                          # 第 i 帧 BGR，ROI 尺寸
    u_full, v_full = u + rec.offset_x, v + rec.offset_y   # ROI 像素 → 全幅（标定）像素

命令行（看画面 / 查时间轴）：
    python test_src/hs_recording.py <base> --summary
    python test_src/hs_recording.py <base> --mp4 out.mp4 [--start-pc T0] [--end-pc T1] [--play-fps 30]
    python test_src/hs_recording.py <base> --png-dir out/ --start-pc T0 --end-pc T1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

INDEX_DTYPE = np.dtype([
    ("frame_num", "<i8"),
    ("dev_timestamp", "<u8"),
    ("exposure_start_pc", "<f8"),
    ("arrival_perf", "<f8"),
    ("lost_packet", "<i8"),
])

# GigE PFNC 像素格式码 → OpenCV 解码。海康的 Bayer 命名与 OpenCV 相反（同 ball_grabber.frame_to_numpy）
_BAYER_TO_BGR = {
    0x01080008: cv2.COLOR_BayerGB2BGR,   # BayerGR8（18F MV-CS032-60GC 实测）
    0x01080009: cv2.COLOR_BayerBG2BGR,   # BayerRG8
    0x0108000A: cv2.COLOR_BayerGR2BGR,   # BayerGB8
    0x0108000B: cv2.COLOR_BayerRG2BGR,   # BayerBG8
}
_MONO8 = 0x01080001


def _base_of(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix in (".json", ".idx", ".bayer"):
        p = p.with_suffix("")
    return p


@dataclass
class HighSpeedRecording:
    base: Path
    meta: dict
    index: np.ndarray
    frames: np.ndarray          # memmap (n, H, W) uint8
    pixel_type: int

    @classmethod
    def open(cls, path: str | Path) -> "HighSpeedRecording":
        base = _base_of(path)
        meta = json.loads(base.with_name(base.name + ".json").read_text(encoding="utf-8"))
        index = np.fromfile(base.with_name(base.name + ".idx"), dtype=INDEX_DTYPE)
        rec = meta.get("recording") or {}
        if rec.get("frame_shape"):
            h, w = rec["frame_shape"]
        else:  # 录像没正常收尾：按配置的 ROI 尺寸读
            h, w = meta["roi"]["height"], meta["roi"]["width"]
        raw_path = base.with_name(base.name + ".bayer")
        n = min(len(index), raw_path.stat().st_size // (h * w))
        frames = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(n, h, w)) if n else np.zeros((0, h, w), np.uint8)
        pixel_type = int(rec.get("pixel_type") or 0x01080008)
        return cls(base=base, meta=meta, index=index[:n], frames=frames, pixel_type=pixel_type)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def offset_x(self) -> int:
        return int(self.meta["roi"]["offset_x"])

    @property
    def offset_y(self) -> int:
        return int(self.meta["roi"]["offset_y"])

    @property
    def exposure_pc(self) -> np.ndarray:
        return self.index["exposure_start_pc"]

    def bgr(self, i: int) -> np.ndarray:
        raw = np.asarray(self.frames[i])
        if self.pixel_type == _MONO8:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(raw, _BAYER_TO_BGR[self.pixel_type])

    def nearest(self, t_pc: float) -> int:
        return int(np.argmin(np.abs(self.exposure_pc - t_pc)))

    def window(self, t0: float | None, t1: float | None) -> np.ndarray:
        t = self.exposure_pc
        keep = np.ones(len(t), dtype=bool)
        if t0 is not None:
            keep &= t >= t0
        if t1 is not None:
            keep &= t <= t1
        return np.flatnonzero(keep)

    def summary(self) -> dict:
        t = self.exposure_pc
        gaps = np.diff(t) if len(t) > 1 else np.zeros(0)
        slot = 1.0 / float(self.meta.get("slot_fps") or 120.0)
        step = int(self.meta.get("video_every_slots") or 1)     # swing_hs 录像：每 N 个触发格存一帧，帧号正常差 N
        fn = self.index["frame_num"]
        return {
            "frames": len(self),
            "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            "fps": float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 else 0.0,
            "slot_ms": round(slot * 1000.0, 3),
            "gap_ms_median": round(float(np.median(gaps)) * 1000.0, 3) if len(gaps) else None,
            "gap_ms_p99": round(float(np.percentile(gaps, 99)) * 1000.0, 3) if len(gaps) else None,
            # 时间轴上的空档 = 跳格（触发没发或被拒）；frame_num 缺口 = 相机侧/线上丢帧
            "gaps_over_1_5_slots": int(np.sum(gaps > 1.5 * slot)),
            "frame_num_gaps": int(np.sum(np.diff(fn) > step)) if len(fn) > 1 else 0,
            "lost_packet_frames": int(np.sum(self.index["lost_packet"] > 0)),
            "sync_frames": len(self.meta.get("sync_frame_nums") or []),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="<run>_hs_<serial>（带不带 .json/.idx/.bayer 后缀都行）")
    ap.add_argument("--summary", action="store_true", help="打印帧数/帧率/跳格/丢包")
    ap.add_argument("--start-pc", type=float, default=None, help="窗口起点（PC perf 秒，同 session json）")
    ap.add_argument("--end-pc", type=float, default=None, help="窗口终点")
    ap.add_argument("--mp4", default=None, help="导出 mp4（每帧左上角印 frame_num 与 t）")
    ap.add_argument("--play-fps", type=float, default=30.0, help="mp4 播放帧率（120 帧/秒录的，30 播 = 4 倍慢放）")
    ap.add_argument("--png-dir", default=None, help="窗口内逐帧导 png")
    args = ap.parse_args()

    rec = HighSpeedRecording.open(args.base)
    if args.summary or not (args.mp4 or args.png_dir):
        print(json.dumps(rec.summary(), ensure_ascii=False, indent=2))
    idx = rec.window(args.start_pc, args.end_pc)
    if args.mp4:
        h, w = rec.frames.shape[1:]
        writer = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.play_fps, (w, h))
        for i in idx:
            img = rec.bgr(int(i))
            label = f"#{int(rec.index['frame_num'][i])}  t={rec.exposure_pc[i]:.4f}"
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(img)
        writer.release()
        print(f"mp4: {args.mp4} ({len(idx)} 帧)")
    if args.png_dir:
        out = Path(args.png_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i in idx:
            cv2.imwrite(str(out / f"hs_{int(rec.index['frame_num'][i]):07d}.png"), rec.bgr(int(i)))
        print(f"png: {out} ({len(idx)} 帧)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
