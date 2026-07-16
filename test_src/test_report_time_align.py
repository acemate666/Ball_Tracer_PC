# -*- coding: utf-8 -*-
"""回归测试:报告 JS 端 PC↔RK 自动对齐(estimateOffset)。

0716 早场 bug(081638):RK 各节点晚入网 70s+,extract_rk_tracking_bag 的
t0=全 bag 最小 payload 把 RK 相对轴锚偏,真实 rkOffset=+61.97s 溢出旧 ±60s
硬编码扫描窗,自动对齐落假谷(0.6m)、全报告错轴。

测试从 generate_curve3_html.py 模板抽取 [[align-core-begin/end]] 标记间的
JS 原文(不是 Python 复刻,测的就是发布的那段代码),用 node 跑合成抛球数据:
- offset=+75s:超出旧 ±60s 界,锁死"扫描界由数据重叠推导"这一修复;
- offset=-9s:extractor 正常锚点下的典型小偏移;
- 混叠防护:各抛间距/斜率刻意不同 + RK 侧躺地球段,复现 0712 假谷土壤。
node 不在 PATH 时跳过(报告本身在浏览器跑,node 仅测试用)。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent / "generate_curve3_html.py"
NODE = shutil.which("node")


def _align_core_js() -> str:
    text = SRC.read_text(encoding="utf-8")
    m = re.search(r"// \[\[align-core-begin\]\].*?\n(.*)// \[\[align-core-end\]\]", text, re.S)
    assert m, "generate_curve3_html.py 缺 [[align-core-begin/end]] 标记"
    return m.group(1)


def _synth(offset: float):
    """6 抛,PC 轴 30Hz;间距/初始高度刻意不等,杜绝整数抛混叠假谷。"""
    starts = [10.0, 24.5, 40.5, 55.0, 71.0, 85.5]
    obs = []
    rk = []
    for k, t0 in enumerate(starts):
        y0 = 8.0 - 0.6 * k          # 每抛斜率不同(6.7→4.2 m/s,均过 1.5 m/s 运动门)
        for i in range(37):          # 1.2s 飞行
            t = t0 + i / 30.0
            y = y0 * (1.0 - (i / 30.0) / 1.2)
            obs.append({"rel_s": round(t, 6), "x": 0.0, "y": round(y, 6), "z": 1.0})
            rk.append((round(t - offset, 6), round(y + 0.02 * math.sin(i * 1.7), 6)))
    for i in range(200):             # RK 侧躺地球段:y 恒定,应被运动门滤掉
        rk.append((round(starts[-1] + 3.0 - offset + i / 30.0, 6), 4.0))
    rk.sort()
    return obs, {"world": {"t": [t for t, _ in rk], "y": {"y": [y for _, y in rk]}}}


def _run_estimate(obs, rk_data, tmp_path: Path) -> dict:
    harness = (
        "const isNum = v => typeof v==='number' && isFinite(v);\n"
        f"const obs = {json.dumps(obs)};\n"
        "const car = [];\n"
        "const relTime = v => v;\n"
        f"const RK = {json.dumps(rk_data)};\n"
        f"{_align_core_js()}\n"
        "console.log(JSON.stringify(estimateOffset()));\n"
    )
    script = tmp_path / "align_harness.js"
    script.write_text(harness, encoding="utf-8")
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("offset", [75.0, -9.0])  # 75 > 旧 ±60 硬编码界:本 bug 的回归锁
def test_estimate_offset_recovers(tmp_path, offset):
    obs, rk_data = _synth(offset)
    best = _run_estimate(obs, rk_data, tmp_path)
    assert best["err"] is not None and best["err"] < 0.05, best
    assert abs(best["off"] - offset) <= 0.01, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_offset_degenerate_returns_null_err(tmp_path):
    """运动样本不足(<30)时必须返回 err=null,让报告端质量门标红,而非硬给假谷。"""
    obs, rk_data = _synth(5.0)
    rk_data["world"]["t"] = rk_data["world"]["t"][:20]
    rk_data["world"]["y"]["y"] = rk_data["world"]["y"]["y"][:20]
    best = _run_estimate(obs, rk_data, tmp_path)
    assert best == {"off": 0, "err": None, "n": 0}
