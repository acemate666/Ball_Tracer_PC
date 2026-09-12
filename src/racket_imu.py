"""Preserve complete WitMotion BLE notifications with PC receipt timestamps."""

from __future__ import annotations

import asyncio
import time


NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
TOPIC = "/racket/imu"


async def stream_racket_imu(address: str, publish) -> None:
    from bleak import BleakClient, BleakScanner

    device = await BleakScanner.find_device_by_address(address, timeout=10)
    if device is None:
        raise RuntimeError(f"Racket IMU {address} not found; disconnect WitMotion and power on the IMU")

    received = asyncio.Queue()

    def on_notification(_sender, data):
        received.put_nowait((time.perf_counter(), bytes(data)))

    async with BleakClient(device, timeout=15) as client:
        await client.start_notify(NOTIFY_UUID, on_notification)
        print(f"[racket_imu] Connected: {device.name} {device.address} -> {TOPIC}", flush=True)
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
            print(f"[racket_imu] Received {count} notifications", flush=True)
