import asyncio
import contextlib
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from racket_imu import NOTIFY_UUID, WRITE_UUID, _check_configuration, stream_racket_imu


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
        receipt_times = iter([100.0, 100.04])
        with patch("bleak.BleakScanner.find_device_by_address", AsyncMock(return_value=device)), \
             patch("bleak.BleakClient", return_value=client), \
             patch("racket_imu.time.perf_counter", side_effect=lambda: next(receipt_times, 100.04)):
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

    async def test_configuration_checks_never_filter_or_stop_raw_recording(self):
        for values, failure in (
            ({0x03: 11, 0x1F: 0, 0x0E: 0xE0}, None),
            ({0x03: 8, 0x1F: 4, 0x0E: 0xC0}, None),
            ({0x03: 11, 0x1F: 0, 0x0E: 0xE0}, OSError("read failed")),
            ({0x03: 11, 0x1F: 0, 0x0E: 0xE0}, TimeoutError()),
        ):
            with self.subTest(values=values, failure=failure):
                device = SimpleNamespace(name="WT901BLE68", address="F4:69:A2:72:B4:23")
                client = AsyncMock()
                client.__aenter__.return_value = client
                sent, samples = [], []
                checked = asyncio.Event()
                recorded_after_check = asyncio.Event()
                callback = None
                # A C0 frame may contain reply-like bytes in its acceleration payload.
                packet = bytes.fromhex("55c01a090a0c22091e03557103000b00")
                tail = b"final raw notification"

                def emit(raw):
                    sent.append(raw)
                    callback(None, bytearray(raw))

                async def notify(_uuid, handler):
                    nonlocal callback
                    callback = handler
                    emit(packet)

                async def write(uuid, command, *, response):
                    self.assertEqual(uuid, WRITE_UUID)
                    self.assertFalse(response)
                    self.assertEqual(command[:3], bytes.fromhex("ffaa27"))
                    self.assertEqual(command[4], 0)
                    register = command[3]
                    if register == 0x03 and failure is not None:
                        raise failure
                    reply = b"\x55\x71" + struct.pack("<H8H", register, values[register], *([0] * 7))
                    combined = packet + reply + packet
                    # Register replies can be batched with data and split across notifications.
                    emit(combined[:19])
                    emit(combined[19:])

                async def check(client, notifications):
                    await _check_configuration(client, notifications)
                    checked.set()
                    emit(tail)

                def publish(sample):
                    samples.append(sample)
                    if bytes.fromhex(sample["raw_hex"]) == tail:
                        recorded_after_check.set()

                client.start_notify.side_effect = notify
                client.write_gatt_char.side_effect = write
                with patch("bleak.BleakScanner.find_device_by_address", AsyncMock(return_value=device)), \
                     patch("bleak.BleakClient", return_value=client), \
                     patch("racket_imu._check_configuration", side_effect=check), \
                     patch("builtins.print") as output:
                    task = asyncio.create_task(stream_racket_imu(device.address, publish))
                    try:
                        await asyncio.wait_for(checked.wait(), timeout=1)
                        await asyncio.wait_for(recorded_after_check.wait(), timeout=1)
                        self.assertFalse(task.done())
                    finally:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                self.assertEqual([bytes.fromhex(s["raw_hex"]) for s in samples], sent)
                self.assertEqual([s["notification_index"] for s in samples], list(range(len(sent))))
                self.assertEqual(client.write_gatt_char.await_count, 3)
                warnings = [str(call.args[0]) for call in output.call_args_list if "WARNING" in str(call.args[0])]
                expected_warnings = 1 if failure is not None else (3 if values[0x03] != 11 else 0)
                self.assertEqual(len(warnings), expected_warnings)

    async def test_missing_register_reply_only_warns(self):
        notifications = asyncio.Queue()
        client = AsyncMock()

        async def write(_uuid, command, *, response):
            register = command[3]
            if register != 0x03:
                value = {0x1F: 0, 0x0E: 0xE0}[register]
                notifications.put_nowait(b"\x55\x71" + struct.pack("<H8H", register, value, *([0] * 7)))

        client.write_gatt_char.side_effect = write
        with patch("builtins.print") as output:
            await asyncio.wait_for(_check_configuration(client, notifications), timeout=4)
        warnings = [str(call.args[0]) for call in output.call_args_list if "WARNING" in str(call.args[0])]
        self.assertEqual(len(warnings), 1)
        self.assertIn("return rate", warnings[0])
        self.assertEqual(client.write_gatt_char.await_count, 3)
