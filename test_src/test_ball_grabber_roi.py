from __future__ import annotations

import numpy as np

from src.ball_grabber import (
    Frame,
    PixelType_Gvsp_BayerRG8,
    frame_bayer_roi_to_numpy,
    frame_to_numpy,
)


def test_bayer_roi_matches_full_decode_away_from_crop_edges():
    width = 24
    height = 24
    raw = np.random.default_rng(7).integers(
        0, 256, size=(height, width), dtype=np.uint8
    )
    frame = Frame(
        data=raw.tobytes(),
        width=width,
        height=height,
        frame_num=1,
        pixel_type=PixelType_Gvsp_BayerRG8,
    )

    x, y, size = 3, 5, 12
    roi = frame_bayer_roi_to_numpy(
        frame,
        x=x,
        y=y,
        width=size,
        height=size,
        resize_to=size,
    )
    expected = frame_to_numpy(frame)[y:y + size, x:x + size]

    np.testing.assert_array_equal(roi[2:-2, 2:-2], expected[2:-2, 2:-2])
