"""Decode recorded racket BLE packets on the PC receipt-time axis."""

import json
import struct

import numpy as np
from mcap.reader import make_reader


def load_racket_imu(session):
    data = json.loads((session / (session.name + ".json")).read_text(encoding="utf-8"))
    origin = data["config"]["first_frame_exposure_pc"]
    notifications = []
    with next(session.rglob("*.mcap")).open("rb") as stream:
        for _, _, message in make_reader(stream).iter_messages(topics=["/racket/imu"], log_time_order=False):
            size = struct.unpack_from("<I", message.data, 4)[0]
            notifications.append(json.loads(message.data[8:8 + size - 1]))
    if [r["notification_index"] for r in notifications] != list(range(len(notifications))):
        raise ValueError("Non-contiguous notification indices")
    frames, batch_ends, receipts, registers = [], [], [], {}
    for row in notifications:
        raw = bytes.fromhex(row["raw_hex"])
        offset, before = 0, len(frames)
        while offset < len(raw):
            if raw[offset:offset + 2] == bytes([85, 224]) and len(raw) - offset >= 22:
                frames.append(raw[offset:offset + 22])
                offset += 22
            elif raw[offset:offset + 2] == bytes([85, 113]) and len(raw) - offset >= 20:
                register, value = struct.unpack_from("<HH", raw, offset + 2)
                registers[hex(register)] = value
                offset += 20
            else:
                raise ValueError("Unsupported or partial BLE frame: " + raw[offset:].hex())
        if len(frames) > before:
            batch_ends.append(len(frames) - 1)
            receipts.append(row["recv_pc"])
    raw = np.array([struct.unpack_from("<6h", frame, 10) for frame in frames], dtype=np.int32)
    counter = np.array([struct.unpack_from("<H", frame, 8)[0] for frame in frames])
    steps = np.diff(counter) % 1000
    if not np.all((steps > 0) & (steps % 5 == 0)):
        raise ValueError("Device counter is not on the expected 5 ms sample grid")
    slots = np.r_[0, np.cumsum(steps // 5)]
    # Counters identify missing slots only; recorded perf_counter receipts set time.
    period, intercept = np.polyfit(slots[batch_ends], np.asarray(receipts) - origin, 1)
    t_pc = origin + intercept + slots * period
    return {"data": data, "raw": raw, "acc_g": raw[:, :3] / 2048,
            "gyro_dps": raw[:, 3:] * 2000 / 32768, "t_pc": t_pc, "period_pc_s": period,
            "sample_step": steps // 5, "registers": registers, "notification_count": len(notifications),
            "receipt_residual_ms": (np.asarray(receipts) - t_pc[batch_ends]) * 1000}
