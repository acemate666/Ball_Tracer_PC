# -*- coding: utf-8 -*-
"""Regression tests for the report-side PC/RK z-axis alignment."""

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
    match = re.search(
        r"// \[\[align-core-begin\]\].*?\n(.*)// \[\[align-core-end\]\]",
        text,
        re.S,
    )
    assert match, "generate_curve3_html.py is missing the align-core markers"
    return match.group(1)


def _synth(
    offset: float,
    *,
    z_bias: float = 0.08,
    sparse_pc: bool = False,
    noise_amplitude: float = 0.01,
):
    starts = [10.0, 24.5, 40.5, 55.0, 71.0, 85.5]
    obs = []
    rk = []
    for index, start in enumerate(starts):
        duration = 1.08 + 0.04 * (index % 3)
        amplitude = 0.9 + 0.12 * index
        sample_count = round(duration * 30) + 1
        for sample in range(sample_count):
            elapsed = sample / 30.0
            t = start + elapsed
            phase = min(elapsed / duration, 1.0)
            z = 0.25 + amplitude * 4.0 * phase * (1.0 - phase)
            if not sparse_pc or sample % 9 not in (3, 4):
                obs.append(
                    {"rel_s": round(t, 6), "x": 0.0, "y": 0.0, "z": round(z, 6)}
                )
            rk.append(
                (
                    round(t - offset, 6),
                    round(
                        z + z_bias + noise_amplitude * math.sin(sample * 1.7),
                        6,
                    ),
                )
            )

    for sample in range(200):
        rk.append(
            (
                round(starts[-1] + 3.0 - offset + sample / 30.0, 6),
                0.25 + z_bias,
            )
        )

    rk.sort()
    return obs, {
        "world": {
            "t": [t for t, _ in rk],
            "y": {"z": [z for _, z in rk]},
        }
    }


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
    result = subprocess.run(
        [NODE, str(script)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("offset", [75.0, -9.0])
def test_estimate_offset_recovers(tmp_path, offset):
    obs, rk_data = _synth(offset)
    best = _run_estimate(obs, rk_data, tmp_path)
    assert best["err"] is not None and best["err"] < 0.05, best
    assert abs(best["off"] - offset) <= 0.01, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_offset_ignores_z_bias_and_pc_gaps(tmp_path):
    obs, rk_data = _synth(-8.823, z_bias=0.35, sparse_pc=True)
    best = _run_estimate(obs, rk_data, tmp_path)
    assert best["err"] is not None and best["err"] < 0.05, best
    assert abs(best["off"] + 8.823) <= 0.012, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_offset_sub_millisecond_refinement(tmp_path):
    obs, rk_data = _synth(-8.8222, noise_amplitude=0.0)
    best = _run_estimate(obs, rk_data, tmp_path)
    assert abs(best["off"] + 8.8222) <= 0.0005, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_offset_weights_flights_not_frames(tmp_path):
    true_offset = -8.8222
    obs, rk_data = _synth(true_offset, noise_amplitude=0.0)
    outlier_offset = true_offset + 0.030
    for sample in range(481):
        elapsed = sample / 120.0
        t = 110.0 + elapsed
        phase = elapsed / 4.0
        z = 0.25 + 1.4 * 4.0 * phase * (1.0 - phase)
        obs.append({"rel_s": round(t, 6), "x": 0.0, "y": 0.0, "z": round(z, 6)})
        rk_data["world"]["t"].append(round(t - outlier_offset, 6))
        rk_data["world"]["y"]["z"].append(round(z + 0.2, 6))

    best = _run_estimate(obs, rk_data, tmp_path)
    assert abs(best["off"] - true_offset) <= 0.003, best


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_estimate_offset_degenerate_returns_null_err(tmp_path):
    obs, rk_data = _synth(5.0)
    rk_data["world"]["t"] = rk_data["world"]["t"][:20]
    rk_data["world"]["y"]["z"] = rk_data["world"]["y"]["z"][:20]
    best = _run_estimate(obs, rk_data, tmp_path)
    assert best == {"off": 0, "err": None, "n": 0}
