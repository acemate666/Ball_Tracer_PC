from __future__ import annotations

import http.client
import json
import time
import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from calibration.phone_preview import PhonePreviewServer, _PreviewHandler


class PhonePreviewServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preview = PhonePreviewServer(["CAM_A", "CAM_B"], target_frames=25, port=0)
        self.preview.start(host="127.0.0.1")

    def tearDown(self) -> None:
        self.preview.close()

    def request(self, path: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.preview.port, timeout=2.0)
        connection.request(method, path)
        response = connection.getresponse()
        status = response.status
        headers = {key.lower(): value for key, value in response.getheaders()}
        body = response.read()
        connection.close()
        return status, headers, body

    def test_page_status_and_resized_jpeg(self) -> None:
        self.assertFalse(self.preview.should_publish(time.perf_counter()))
        status, headers, page = self.request("/")
        self.assertEqual(200, status)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn(b"CAM_A", page)
        self.assertIn(b"CAM_B", page)

        self.preview.update_status(captured=7, elapsed_s=3.5)
        status, _, body = self.request("/status.json")
        self.assertEqual(200, status)
        self.assertEqual(
            {"captured": 7, "target": 25, "elapsed_s": 3.5, "state": "capturing"},
            json.loads(body),
        )

        now = time.perf_counter()
        self.assertTrue(self.preview.should_publish(now))
        image = np.full((1536, 2048, 3), (20, 120, 220), dtype=np.uint8)
        self.preview.publish({"CAM_A": image, "CAM_B": image}, now)
        status, headers, jpeg = self.request("/frame/CAM_A.jpg")
        self.assertEqual(200, status)
        self.assertEqual("image/jpeg", headers["content-type"])
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((540, 720), decoded.shape[:2])

    def test_read_only_mode_rejects_capture_controls(self) -> None:
        for action in ("start", "stop"):
            status, _, _ = self.request(f"/capture/{action}", "POST")
            self.assertEqual(409, status)
        self.assertEqual("capturing", self.preview.capture_state)

    def test_browser_disconnect_during_response_is_ignored(self) -> None:
        handler = object.__new__(_PreviewHandler)
        handler.wfile = MagicMock()
        with (
            patch.object(handler, "send_response"),
            patch.object(handler, "send_header"),
            patch.object(handler, "end_headers") as end_headers,
            patch("calibration.phone_preview.BaseHTTPRequestHandler.handle") as handle,
        ):
            for error in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                with self.subTest(error=error, stage="read"):
                    handle.side_effect = error()
                    handler.handle()
                handle.side_effect = lambda: handler._send(HTTPStatus.OK, "image/jpeg", b"frame")
                with self.subTest(error=error, stage="headers"):
                    end_headers.side_effect = error()
                    handler.handle()
                with self.subTest(error=error, stage="body"):
                    end_headers.side_effect = None
                    handler.wfile.write.side_effect = error()
                    handler.handle()
            handler.wfile.write.side_effect = ValueError("unexpected failure")
            with self.assertRaisesRegex(ValueError, "unexpected failure"):
                handler.handle()
        status, _, _ = self.request("/status.json")
        self.assertEqual(200, status)

    def test_web_controls_start_once_and_stop(self) -> None:
        self.preview.close()
        self.preview = PhonePreviewServer(["CAM_A"], 25, port=0, web_control=True)
        self.preview.start(host="127.0.0.1")
        self.assertEqual("waiting", self.preview.capture_state)
        status, _, _ = self.request("/capture/start")
        self.assertEqual(404, status)
        self.assertEqual("waiting", self.preview.capture_state)
        status, _, body = self.request("/capture/start", "POST")
        self.assertEqual(200, status)
        self.assertEqual("capturing", json.loads(body)["state"])
        status, _, _ = self.request("/capture/start", "POST")
        self.assertEqual(409, status)
        status, _, body = self.request("/capture/stop", "POST")
        self.assertEqual(200, status)
        self.assertEqual("stopped", json.loads(body)["state"])
        status, _, _ = self.request("/capture/start", "POST")
        self.assertEqual(409, status)


if __name__ == "__main__":
    unittest.main()
