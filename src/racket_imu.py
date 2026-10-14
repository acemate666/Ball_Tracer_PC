"""Preserve complete WitMotion BLE notifications with PC receipt timestamps."""

from __future__ import annotations

import asyncio
import contextlib
import struct
import time


NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
WRITE_UUID = "0000ffe9-0000-1000-8000-00805f9a34fb"
TOPIC = "/racket/imu"


async def _check_configuration(client, notifications) -> None:
    buffer = bytearray()
    for register, expected, label in (
        (0x03, 0x0B, "return rate 200 Hz"),
        (0x1F, 0x00, "bandwidth 256 Hz"),
        (0x0E, 0xE0, "fields time + acceleration + angular velocity (55 E0)"),
    ):
        try:
            await asyncio.wait_for(client.write_gatt_char(
                WRITE_UUID, bytes([0xFF, 0xAA, 0x27, register, 0]), response=False,
            ), timeout=2)
            deadline = time.perf_counter() + 2
            value = None
            while value is None:
                buffer.extend(await asyncio.wait_for(
                    notifications.get(), timeout=max(0, deadline - time.perf_counter()),
                ))
                while len(buffer) >= 2:
                    flag = buffer[1]
                    if buffer[0] != 0x55 or (flag != 0x71 and flag & 0x10):
                        raise ValueError(f"unrecognized packet header {buffer[:2].hex()}")
                    # Output-mask lengths let replies be read even when fields differ from E0.
                    size = 20 if flag == 0x71 else 2 + sum(
                        length for bit, length in ((0x80, 8), (0x40, 6), (0x20, 6),
                                                  (0x08, 6), (0x04, 8), (0x02, 12), (0x01, 6))
                        if flag & bit
                    )
                    if len(buffer) < size:
                        break
                    if flag == 0x71 and struct.unpack_from("<H", buffer, 2)[0] == register:
                        value = struct.unpack_from("<H", buffer, 4)[0]
                    del buffer[:size]
                    if value is not None:
                        break
            if value != expected:
                print(f"[racket_imu] WARNING: expected {label}; register 0x{register:02X}="
                      f"0x{value:04X}, expected 0x{expected:04X}. Recording raw data unchanged.", flush=True)
            else:
                print(f"[racket_imu] Configuration OK: {label}", flush=True)
        except Exception as exc:
            print(f"[racket_imu] WARNING: could not verify {label}: {type(exc).__name__}: {exc}. "
                  "Recording raw data unchanged.", flush=True)


async def stream_racket_imu(address: str, publish) -> None:
    from bleak import BleakClient, BleakScanner

    device = await BleakScanner.find_device_by_address(address, timeout=10)
    if device is None:
        raise RuntimeError(f"Racket IMU {address} not found; disconnect WitMotion and power on the IMU")

    received = asyncio.Queue()
    configuration_notifications = asyncio.Queue()
    check_task = None

    def on_notification(_sender, data):
        raw = bytes(data)
        received.put_nowait((time.perf_counter(), raw))
        if check_task is None or not check_task.done():
            configuration_notifications.put_nowait(raw)

    async with BleakClient(device, timeout=15) as client:
        await client.start_notify(NOTIFY_UUID, on_notification)
        print(f"[racket_imu] Connected: {device.name} {device.address} -> {TOPIC}", flush=True)
        check_task = asyncio.create_task(_check_configuration(client, configuration_notifications))
        count = 0
        try:
            while True:
                try:
                    recv_pc, data = await asyncio.wait_for(received.get(), timeout=2)
                except TimeoutError as exc:
                    raise RuntimeError("Racket IMU received no BLE data for 2 seconds") from exc
                publish({
                    "recv_pc": recv_pc,
                    "address": device.address,
                    "notification_index": count,
                    "raw_hex": data.hex(),
                })
                count += 1
                if count == 1:
                    print(f"[racket_imu] Receiving: {len(data)} bytes/notification", flush=True)
        finally:
            check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await check_task
            print(f"[racket_imu] Received {count} notifications", flush=True)
