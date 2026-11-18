# -*- coding: utf-8 -*-
"""LAN preview with optional start/stop controls for calibration capture."""

from __future__ import annotations

import html
import ipaddress
import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

import cv2
import numpy as np


PREVIEW_PORT = 8765
_PREVIEW_INTERVAL_S = 0.2
_PREVIEW_WIDTH = 720
_JPEG_QUALITY = 72


def _page(serials: list[str], web_control: bool) -> bytes:
    panels = "".join(
        "<section class=\"camera\">"
        f"<div class=\"label\">{html.escape(serial)}</div>"
        f"<img data-frame=\"/frame/{quote(serial, safe='')}.jpg\" alt=\"{html.escape(serial)}\">"
        "</section>"
        for serial in serials
    )
    controls = """
<div class="controls">
  <button id="start" type="button">开始采集</button>
  <button id="stop" type="button">结束采集</button>
  <span id="capture-state">等待开始</span>
</div>""" if web_control else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Calibration Capture</title>
<style>
*{{box-sizing:border-box;letter-spacing:0}}
body{{margin:0;background:#111;color:#f2f2f2;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{height:48px;padding:0 12px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #333;background:#181818}}
h1{{margin:0;font-size:15px;font-weight:600}}
.status{{display:flex;align-items:center;gap:8px;font-variant-numeric:tabular-nums;font-size:13px;color:#d7d7d7}}
.signal{{width:4px;height:14px;background:#3ecf8e}}
.controls{{height:56px;display:flex;align-items:center;gap:10px;padding:8px 12px}}
button{{min-height:40px;padding:0 14px;border:0;border-radius:4px;font:inherit;cursor:pointer}}
#start{{background:#3ecf8e;color:#111}}#stop{{background:#d9534f;color:#fff}}
button:disabled{{opacity:.4;cursor:default}}
#capture-state{{font-size:13px}}
main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;padding:6px;max-width:1440px;margin:0 auto}}
.camera{{position:relative;min-width:0;aspect-ratio:4/3;background:#050505;border:1px solid #303030;border-radius:4px;overflow:hidden}}
.camera img{{display:block;width:100%;height:100%;object-fit:contain}}
.label{{position:absolute;z-index:1;top:6px;left:6px;max-width:calc(100% - 12px);padding:3px 5px;background:rgba(0,0,0,.72);font:600 11px ui-monospace,"Cascadia Mono",monospace;overflow-wrap:anywhere}}
@media (orientation:landscape) and (max-height:600px){{header{{height:40px}}main{{height:calc(100vh - 40px);grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr))}}.camera{{aspect-ratio:auto}}}}
@media (orientation:landscape) and (max-height:600px){{.web-control main{{height:calc(100vh - 96px)}}}}
</style>
</head>
<body class="{'web-control' if web_control else ''}">
<header><h1>Calibration Capture</h1><div class="status"><span class="signal"></span><span id="count">0 / 0</span><span id="elapsed">0.0 s</span></div></header>
{controls}
<main>{panels}</main>
<script>
const count=document.getElementById('count');
const elapsed=document.getElementById('elapsed');
const images=document.querySelectorAll('img[data-frame]');
const startButton=document.getElementById('start');
const stopButton=document.getElementById('stop');
const captureState=document.getElementById('capture-state');
let finished=false;
let frameVersion=0;
for(const image of images){{
  image.onload=image.onerror=()=>{{image.dataset.loading='';}};
}}
function refreshFrames(){{
  if(finished) return;
  const version=++frameVersion;
  for(const image of images){{
    if(image.dataset.loading) continue;
    image.dataset.loading='1';
    image.src=`${{image.dataset.frame}}?v=${{version}}`;
  }}
}}
function showStatus(state){{
  if(finished) return;
  count.textContent=`${{state.captured}} / ${{state.target}}`;
  elapsed.textContent=`${{state.elapsed_s.toFixed(1)}} s`;
  finished=state.state==='stopped';
  if(startButton){{
    startButton.disabled=state.state!=='waiting';
    stopButton.disabled=finished;
    captureState.textContent={{waiting:'等待开始',capturing:'采集中',stopped:'已结束'}}[state.state];
  }}
}}
async function refreshStatus(){{
  if(finished) return;
  try{{
    const response=await fetch('/status.json',{{cache:'no-store'}});
    showStatus(await response.json());
  }}catch(_error){{
    elapsed.textContent='offline';
  }}
}}
async function control(action){{
  startButton.disabled=true;
  stopButton.disabled=true;
  try{{
    const response=await fetch(`/capture/${{action}}`,{{method:'POST'}});
    if(!response.ok) throw new Error('control failed');
    showStatus(await response.json());
  }}catch(_error){{
    captureState.textContent='请求失败，请刷新重试';
  }}
}}
if(startButton){{
  startButton.onclick=()=>control('start');
  stopButton.onclick=()=>control('stop');
}}
refreshFrames();
refreshStatus();
setInterval(refreshFrames,200);
setInterval(refreshStatus,500);
</script>
</body>
</html>""".encode("utf-8")


def _lan_ipv4_addresses() -> list[str]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        preferred = probe.getsockname()[0]
        if preferred:
            return [preferred]
    except OSError:
        pass
    finally:
        probe.close()

    addresses: list[str] = []
    try:
        addresses.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass

    result = []
    for address in addresses:
        parsed = ipaddress.ip_address(address)
        if parsed.is_loopback or parsed.is_link_local or not parsed.is_private:
            continue
        if address not in result:
            result.append(address)
    return result


class _PreviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, preview: "PhonePreviewServer"):
        self.preview = preview
        super().__init__(server_address, _PreviewHandler)


class _PreviewHandler(BaseHTTPRequestHandler):
    server: _PreviewHTTPServer

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers can disconnect while reading a request or sending a response.
            return

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        self.server.preview.note_client()
        if path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", self.server.preview.page)
            return
        if path == "/status.json":
            body = json.dumps(self.server.preview.status()).encode("utf-8")
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", body)
            return
        if path.startswith("/frame/") and path.endswith(".jpg"):
            serial = path.removeprefix("/frame/").removesuffix(".jpg")
            jpeg = self.server.preview.frame(serial)
            if jpeg is None:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8", b"no frame")
            else:
                self._send(HTTPStatus.OK, "image/jpeg", jpeg)
            return
        self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/capture/start", "/capture/stop"}:
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")
            return
        preview = self.server.preview
        accepted = preview.control(path.rsplit("/", 1)[1])
        body = json.dumps(preview.status()).encode("utf-8")
        self._send(HTTPStatus.OK if accepted else HTTPStatus.CONFLICT,
                   "application/json; charset=utf-8", body)

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


class PhonePreviewServer:
    def __init__(self, serials: list[str], target_frames: int, port: int = PREVIEW_PORT,
                 *, web_control: bool = False):
        self.serials = list(serials)
        self.page = _page(self.serials, web_control)
        self._web_control = web_control
        self._state = "waiting" if web_control else "capturing"
        self._target_frames = int(target_frames)
        self._port = int(port)
        self._lock = threading.Lock()
        self._frames: dict[str, bytes] = {}
        self._captured = 0
        self._elapsed_s = 0.0
        self._last_publish_perf = 0.0
        self._client_active_until = 0.0
        self._server: _PreviewHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._port
        return int(self._server.server_address[1])

    @property
    def urls(self) -> list[str]:
        addresses = _lan_ipv4_addresses()
        if not addresses:
            addresses = ["127.0.0.1"]
        return [f"http://{address}:{self.port}" for address in addresses]

    def start(self, host: str = "0.0.0.0") -> None:
        if self._server is not None:
            raise RuntimeError("phone preview server is already started")
        try:
            server = _PreviewHTTPServer((host, self._port), self)
        except OSError as exc:
            raise RuntimeError(f"cannot start phone preview on {host}:{self._port}: {exc}") from exc
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="phone-preview", daemon=True)
        self._thread.start()

    def close(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def note_client(self) -> None:
        with self._lock:
            self._client_active_until = time.perf_counter() + 2.0

    def should_publish(self, now: float) -> bool:
        with self._lock:
            return (
                now < self._client_active_until
                and now - self._last_publish_perf >= _PREVIEW_INTERVAL_S
            )

    def publish(self, images: dict[str, np.ndarray], now: float) -> None:
        encoded: dict[str, bytes] = {}
        for serial in self.serials:
            image = images.get(serial)
            if image is None:
                continue
            height, width = image.shape[:2]
            if width > _PREVIEW_WIDTH:
                preview_height = max(1, round(height * _PREVIEW_WIDTH / width))
                image = cv2.resize(image, (_PREVIEW_WIDTH, preview_height), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"failed to encode preview frame for camera {serial}")
            encoded[serial] = jpeg.tobytes()

        with self._lock:
            self._frames.update(encoded)
            self._last_publish_perf = now

    def frame(self, serial: str) -> bytes | None:
        with self._lock:
            return self._frames.get(serial)

    @property
    def capture_state(self) -> str:
        with self._lock:
            return self._state

    def control(self, action: str) -> bool:
        with self._lock:
            if not self._web_control:
                return False
            if action == "start" and self._state == "waiting":
                self._state = "capturing"
                return True
            if action == "stop" and self._state in {"waiting", "capturing"}:
                self._state = "stopped"
                return True
            return False

    def update_status(self, captured: int, elapsed_s: float, *, finished: bool = False) -> None:
        with self._lock:
            self._captured = int(captured)
            self._elapsed_s = float(elapsed_s)
            if finished:
                self._state = "stopped"

    def status(self) -> dict:
        with self._lock:
            return {
                "captured": self._captured,
                "target": self._target_frames,
                "elapsed_s": self._elapsed_s,
                "state": self._state,
            }
