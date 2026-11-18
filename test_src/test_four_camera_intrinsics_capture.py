from __future__ import annotations

import contextlib
import http.client
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from calibration import four_camera_intrinsics_capture as capture
from calibration.phone_preview import PhonePreviewServer


class FourCameraCaptureControlTests(unittest.TestCase):
    def test_capture_waits_for_web_start_and_saves_on_stop(self):
        self._capture(web_control=True, stop_after=1)

    def test_stop_before_start_saves_empty_session(self):
        self._capture(web_control=True, stop_after=0)

    def test_web_capture_finishes_at_target(self):
        self._capture(web_control=True, stop_after=None)

    def test_default_capture_starts_automatically(self):
        self._capture(web_control=False, stop_after=None)

    def test_keyboard_interrupt_while_waiting_saves_empty_session(self):
        self._capture(web_control=True, stop_after=0, interrupt=True)

    def test_keyboard_interrupt_during_capture_preserves_counts(self):
        self._capture(web_control=True, stop_after=1, interrupt=True)

    def _capture(self, *, web_control, stop_after, interrupt=False):
        serials = ["CAM_A", "CAM_B", "CAM_C", "CAM_D"]
        preview = PhonePreviewServer(serials, 3, port=0, web_control=web_control)
        clock = 100.0
        calls = 0
        image = np.zeros((12, 16, 3), dtype=np.uint8)

        def request(path, method="GET"):
            connection = http.client.HTTPConnection("127.0.0.1", preview.port, timeout=2)
            connection.request(method, path)
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(200, response.status)
            connection.close()
            return body

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            camera_config = root / "cameras.json"
            camera_config.write_text(json.dumps({
                "master_serial": serials[0], "slave_serials": serials[1:],
            }))
            session = root / "session"

            def get_frames(**_kwargs):
                nonlocal clock, calls
                calls += 1
                self.assertLess(calls, 10, "capture loop did not stop")
                clock += 100.0 if web_control and calls <= 2 else 1.0
                saved = len(list((session / serials[0]).glob("*.png")))
                if web_control and calls <= 2:
                    status = json.loads(request("/status.json"))
                    self.assertEqual("waiting", status["state"])
                    self.assertEqual(0.0, status["elapsed_s"])
                    self.assertEqual(0, saved)
                    if calls == 2:
                        self.assertTrue(request("/frame/CAM_A.jpg").startswith(b"\xff\xd8"))
                        if stop_after == 0:
                            if interrupt:
                                raise KeyboardInterrupt
                            request("/capture/stop", "POST")
                            return None
                elif web_control and calls == 3:
                    request("/capture/start", "POST")
                if stop_after and saved >= stop_after:
                    if interrupt:
                        raise KeyboardInterrupt
                    request("/capture/stop", "POST")
                    return None
                return {sn: image for sn in serials}

            cap = MagicMock()
            cap.sync_serials = serials
            cap.get_frames.side_effect = get_frames
            sdk = types.SimpleNamespace(
                SyncCapture=MagicMock(), frame_to_numpy=lambda frame: frame,
            )
            sdk.SyncCapture.from_config.return_value.__enter__.return_value = cap
            argv = ["capture", "--count", "3", "--duration", "3", "--session", "session",
                    "--camera-config", str(camera_config)]
            if web_control:
                argv.append("--web-control")
            with (
                patch("sys.argv", argv),
                patch.dict("sys.modules", {"src": sdk}),
                patch.object(capture, "FOUR_CAMERA_CALIB_ROOT", root),
                patch.object(capture, "PhonePreviewServer", return_value=preview),
                patch.object(capture, "time", types.SimpleNamespace(
                    perf_counter=lambda: clock, sleep=lambda _seconds: None,
                )),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                capture.main()

            expected = 3 if stop_after is None else stop_after
            manifest = json.loads((session / "session.json").read_text())
            self.assertEqual(expected, manifest["results"]["captured_frames"])
            self.assertEqual({sn: expected for sn in serials},
                             manifest["results"]["per_camera_counts"])
            for sn in serials:
                self.assertEqual(expected, len(list((session / sn).glob("*.png"))))
            self.assertEqual("stopped", preview.capture_state)
            if web_control:
                self.assertLess(preview.status()["elapsed_s"], 4.0)
            sdk.SyncCapture.from_config.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
