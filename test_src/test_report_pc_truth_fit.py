# -*- coding: utf-8 -*-
"""回归测试：报告中的 PC 坐标真值只能来自目标时刻前的轨迹拟合。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
NODE = shutil.which("node")

# tracker_20260717_211020 第 11 拍的真实失败片段：触球附近无 3D，
# 最后一帧入弧与第一帧出弧之间跨过了垂直速度冲量。
CONTACT_T = 179.514209
INCOMING = [
    {"t": 178.96500279055908, "x": 0.6239561916027039, "y": 9.116895664586607, "z": 0.1935399446761026},
    {"t": 179.06820324476575, "x": 0.6420783717542466, "y": 8.799327461375468, "z": 0.7010046381736231},
    {"t": 179.10280739777954, "x": 0.6486899107088009, "y": 8.69063428104331, "z": 0.8377788737412453},
    {"t": 179.13730374805164, "x": 0.6521398511192236, "y": 8.5847216964659, "z": 0.9786220536091201},
    {"t": 179.17200700577814, "x": 0.664182915307448, "y": 8.445411278816284, "z": 1.0957109624293084},
    {"t": 179.20646655152086, "x": 0.6668102286931112, "y": 8.412132739139754, "z": 1.1888742435798725},
    {"t": 179.2405919075245, "x": 0.6679687094365742, "y": 8.261444227369287, "z": 1.2921673071550313},
    {"t": 179.27599974925397, "x": 0.6745699823864145, "y": 8.162187099806044, "z": 1.3725020873147225},
    {"t": 179.30963140854146, "x": 0.6828339686610223, "y": 8.08805967122834, "z": 1.435445108886485},
    {"t": 179.34411374875344, "x": 0.6910786774092686, "y": 8.007356308561233, "z": 1.4856933470199425},
    {"t": 179.44792300503468, "x": 0.6976158367351957, "y": 7.618150003421958, "z": 1.600047791383868},
    {"t": 179.48247415578226, "x": 0.6968117294276229, "y": 7.512739989654369, "z": 1.614178093856178},
]
OUTGOING = {
    "t": 179.55130625178572,
    "x": 0.7334796537486125,
    "y": 7.342568562075761,
    "z": 1.5541189447209376,
}


def _pc_truth_core_js() -> str:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(
        r"// \[\[pc-truth-core-begin\]\].*?\n(.*)// \[\[pc-truth-core-end\]\]",
        text,
        re.S,
    )
    assert match, "generate_curve3_html.py 缺 [[pc-truth-core-begin/end]] 标记"
    return match.group(1)


def _run_truth(tmp_path: Path, rows: list[dict]) -> dict | None:
    harness = (
        f"const pcRows={json.dumps(rows)};\n"
        "const carAt=t=>({x:0,y:0,yaw:0});\n"
        "const relToCar=(b,c)=>b;\n"
        f"{_pc_truth_core_js()}\n"
        f"console.log(JSON.stringify(pcTruthAt({CONTACT_T})));\n"
    )
    script = tmp_path / "pc_truth_harness.js"
    script.write_text(harness, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_ignores_outgoing_frame_across_contact_gap(tmp_path):
    fitted = _run_truth(tmp_path, INCOMING + [OUTGOING])
    assert fitted is not None
    assert fitted["z"] == pytest.approx(1.6159227, abs=0.001)

    before = INCOMING[-1]
    fraction = (CONTACT_T - before["t"]) / (OUTGOING["t"] - before["t"])
    chord_z = before["z"] + fraction * (OUTGOING["z"] - before["z"])
    assert chord_z == pytest.approx(1.58649, abs=0.001)
    assert fitted["z"] - chord_z > 0.025


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_pc_truth_does_not_fall_back_with_too_few_incoming_points(tmp_path):
    assert _run_truth(tmp_path, INCOMING[-4:] + [OUTGOING]) is None
