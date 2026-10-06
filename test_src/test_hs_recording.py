from __future__ import annotations

import numpy as np

from hs_recording import HighSpeedRecording
from src.ball_grabber import Frame, HighSpeedRecorder, PixelType_Gvsp_BayerGR8, frame_to_numpy


def _frames(n: int, *, w: int = 12, h: int = 8, t0: float = 50.0, period: float = 1 / 120):
    rng = np.random.default_rng(11)
    return [
        Frame(
            data=rng.integers(0, 256, size=(h, w), dtype=np.uint8).tobytes(),
            width=w, height=h, frame_num=1000 + i,
            pixel_type=PixelType_Gvsp_BayerGR8,
            exposure_start_pc=t0 + i * period,
            dev_timestamp=10_000 + i,
        )
        for i in range(n)
    ]


def test_reader_round_trips_what_the_recorder_wrote(tmp_path):
    meta = {"serial": "DB0260405", "slot_fps": 120.0,
            "roi": {"offset_x": 720, "offset_y": 268, "width": 12, "height": 8}}
    recorder = HighSpeedRecorder(tmp_path / "s_hs_DB0260405", meta=meta)
    recorder.start()
    frames = _frames(9)
    frames.pop(4)                                   # 中间丢一帧：时间轴上留一个两格的空档
    for f in frames:
        recorder.submit(f)
    recorder.close(extra={"sync_frame_nums": [1002, 1005]})

    rec = HighSpeedRecording.open(tmp_path / "s_hs_DB0260405.json")
    assert len(rec) == 8 and rec.frames.shape == (8, 8, 12)
    assert (rec.offset_x, rec.offset_y) == (720, 268)
    np.testing.assert_array_equal(rec.index["frame_num"], [f.frame_num for f in frames])
    np.testing.assert_array_equal(rec.bgr(5), frame_to_numpy(frames[5]))   # 解码与采集侧一致
    assert rec.nearest(frames[6].exposure_start_pc + 0.001) == 6
    summary = rec.summary()
    assert summary["gaps_over_1_5_slots"] == 1 and summary["frame_num_gaps"] == 1
    assert summary["sync_frames"] == 2
    assert list(rec.window(frames[2].exposure_start_pc, frames[5].exposure_start_pc)) == [2, 3, 4, 5]
