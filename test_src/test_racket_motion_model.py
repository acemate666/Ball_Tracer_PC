import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from test_src.racket_motion_model import calibrate_mount, estimate_window, integrate_attitude


class RacketMotionModelTest(unittest.TestCase):
    def test_rotation_prefix_matches_sequential_composition(self):
        rng = np.random.default_rng(8)
        t = time.perf_counter() + np.cumsum(rng.uniform(.004, .006, 800))
        w = rng.normal(0, 90, (len(t), 3))
        actual = integrate_attitude(t, w)
        expected = Rotation.identity()
        for i in range(1, len(t)):
            expected = expected * Rotation.from_rotvec(np.deg2rad((w[i - 1] + w[i]) * .5) * (t[i] - t[i - 1]))
        self.assertLess(float((expected.inv() * actual[-1]).magnitude()), 1e-11)

    def test_recovers_known_mount_time_and_gyro_bias(self):
        t = time.perf_counter() + np.arange(0, 12, .005)
        u = t - t[0]
        a, b, c = .4 * np.sin(.7 * u), .3 * np.sin(1.1 * u) + .1, .5 * np.cos(.9 * u) - .5
        ad, bd, cd = .28 * np.cos(.7 * u), .33 * np.cos(1.1 * u), -.45 * np.sin(.9 * u)
        omega = np.c_[cd - ad * np.sin(b), bd * np.cos(c) + ad * np.sin(c) * np.cos(b),
                      -bd * np.sin(c) + ad * np.cos(c) * np.cos(b)]
        base = Rotation.from_rotvec([.2, -.1, .4])
        rotation = base * Rotation.from_euler("ZYX", np.c_[a, b, c])
        lever = np.array([-.40, .025, -.035])
        bias = np.array([.3, -.2, .1])
        shift = -.024
        ids = np.arange(80, len(t) - 80, 40)
        observations = [{"t_pc": t[i] + shift, "lever_world_m": rotation[i].apply(lever).tolist()} for i in ids]
        fit = calibrate_mount(t, np.rad2deg(omega) + bias, observations, np.linalg.norm(lever))
        np.testing.assert_allclose(fit["lever_sensor_m"], lever, atol=.001)
        np.testing.assert_allclose(fit["gyro_bias_dps"], bias, atol=.015)
        self.assertAlmostEqual(fit["timing_shift_pc_s"], shift, delta=.001)

    def test_translation_and_rotation_velocity_match_physical_trajectory(self):
        t = time.perf_counter() + np.arange(0, 2.005, .005)
        u = t - t[0]
        base = Rotation.from_rotvec([.2, -.15, .3])
        rotation = base * Rotation.from_rotvec(np.c_[np.zeros(len(u)), np.zeros(len(u)), .4 * u + .2 * u ** 2])
        omega = np.c_[np.zeros(len(u)), np.zeros(len(u)), .4 + .4 * u]
        marker = np.c_[.2 * np.sin(1.1 * u), .1 * u, np.ones(len(u))]
        marker_velocity = np.c_[.22 * np.cos(1.1 * u), np.full(len(u), .1), np.zeros(len(u))]
        acceleration = np.c_[-.242 * np.sin(1.1 * u), np.zeros(len(u)), np.zeros(len(u))]
        lever = np.array([-.40, .03, -.02])
        bias = np.array([.2, -.1, .3])
        imu = {"t_pc": t, "gyro_dps": np.rad2deg(omega) + bias,
               "acc_g": rotation.inv().apply(acceleration - [0, 0, -9.80665]) / 9.80665}
        calibration = {"lever_sensor_m": lever, "gyro_bias_dps": bias,
                       "timing_shift_pc_s": 0., "initial_rotation_vector": base.as_rotvec()}
        ids = [0, 40, 80, 320, 360, 400]
        anchors = [{"t_pc": t[i], "imu_world_m": marker[i], "lever_world_m": rotation[i].apply(lever)} for i in ids]
        query = np.arange(120, 281, 10)
        prediction = estimate_window(imu, calibration, anchors, t[query])
        expected_position = marker[query] + rotation[query].apply(lever)
        expected_velocity = marker_velocity[query] + rotation[query].apply(np.cross(omega[query], lever))
        np.testing.assert_allclose(prediction["head_world_m"], expected_position, atol=1e-4)
        np.testing.assert_allclose(prediction["head_velocity_world_mps"], expected_velocity, atol=1e-3)
        epsilon = 1e-5
        plus = estimate_window(imu, calibration, anchors, t[query] + epsilon)["head_world_m"]
        minus = estimate_window(imu, calibration, anchors, t[query] - epsilon)["head_world_m"]
        derivative = (plus - minus) / ((t[query] + epsilon) - (t[query] - epsilon))[:, None]
        np.testing.assert_allclose(derivative, prediction["head_velocity_world_mps"], atol=1e-5)
        scale, offset = np.array([.97, .93, 1.02]), np.array([-.01, .02, -.03])
        measured_imu = {**imu, "acc_g": (imu["acc_g"] - offset) / scale}
        corrected = estimate_window(measured_imu, {**calibration, "acceleration_scale": scale,
                                    "acceleration_offset_g": offset}, anchors, t[query])
        np.testing.assert_allclose(corrected["head_world_m"], expected_position, atol=1e-4)
        np.testing.assert_allclose(corrected["head_velocity_world_mps"], expected_velocity, atol=1e-3)
