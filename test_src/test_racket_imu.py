import asyncio
import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from racket_imu import NOTIFY_UUID, stream_racket_imu


class RacketImuTest(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_batched_bytes_receipt_time_and_order(self):
        device = SimpleNamespace(name="WT901BLE68", address="F4:69:A2:72:B4:23")
        packet = bytes.fromhex("55e01a090a0c22091e034900ef028607000000000000") * 8
        received = []
        published = asyncio.Event()
        client = AsyncMock()
        client.__aenter__.return_value = client

        async def notify(uuid, callback):
            self.assertEqual(uuid, NOTIFY_UUID)
            callback(None, bytearray(packet))
            callback(None, bytearray(b"\x55\x61"))

        def publish(sample):
            received.append(sample)
            if len(received) == 2:
                published.set()

        client.start_notify.side_effect = notify
        with patch("bleak.BleakScanner.find_device_by_address", AsyncMock(return_value=device)), \
             patch("bleak.BleakClient", return_value=client), \
             patch("racket_imu.time.perf_counter", side_effect=[100.0, 100.04]):
            task = asyncio.create_task(stream_racket_imu(device.address, publish))
            try:
                await asyncio.wait_for(published.wait(), timeout=1)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self.assertEqual([s["notification_index"] for s in received], [0, 1])
        self.assertEqual([s["recv_pc"] for s in received], [100.0, 100.04])
        self.assertEqual(bytes.fromhex(received[0]["raw_hex"]), packet)
        self.assertEqual(bytes.fromhex(received[1]["raw_hex"]), b"\x55\x61")
        self.assertEqual(received[0]["address"], device.address)
        client.__aexit__.assert_awaited_once()

    async def test_missing_device_is_an_error(self):
        with patch("bleak.BleakScanner.find_device_by_address", AsyncMock(return_value=None)):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                await stream_racket_imu("F4:69:A2:72:B4:23", self.fail)
