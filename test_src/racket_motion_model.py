"""Offline visual calibration and IMU propagation of a fixed racket material point."""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


def integrate_attitude(t_pc, gyro_dps):
    gyro = np.deg2rad(gyro_dps)
    increments = Rotation.from_rotvec((gyro[1:] + gyro[:-1]) * .5 * np.diff(t_pc)[:, None]).as_matrix()
    matrices = np.concatenate([np.eye(3)[None], increments])
    # Associative prefix products preserve rotation order and avoid a Python loop per sample.
    stride = 1
    while stride < len(t_pc):
        matrices[stride:] = matrices[:-stride] @ matrices[stride:]
        stride *= 2
    return Rotation.from_matrix(matrices)


def calibrate_mount(t_pc, gyro_dps, observations, length_m):
    """Fit the sensor lever, receipt alignment, initial attitude, and gyro bias."""
    zero_bias_spline = Slerp(t_pc, integrate_attitude(t_pc, gyro_dps))
    spline = zero_bias_spline
    times = np.asarray([r["t_pc"] for r in observations])
    target = np.asarray([r["lever_world_m"] for r in observations])
    cached_bias = np.zeros(3)

    def lever(angles):
        azimuth, elevation = angles
        return length_m * np.array([-np.cos(azimuth) * np.cos(elevation),
                                     np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])

    def residual(parameters):
        nonlocal spline, cached_bias
        bias = parameters[6:9]
        if not np.array_equal(bias, cached_bias):
            spline = Slerp(t_pc, integrate_attitude(t_pc, gyro_dps - bias))
            cached_bias = bias.copy()
        r = lever(parameters[:2])
        Q = Rotation.from_rotvec(parameters[3:6])
        prediction = Q.apply(spline(times - parameters[2]).apply(r))
        return (prediction - target).ravel()

    fits = []
    for shift in [-.04, .04]:
        predicted = zero_bias_spline(times - shift).apply([-length_m, 0, 0])
        initial_Q = Rotation.align_vectors(target, predicted)[0].as_rotvec()
        initial = np.r_[[0., 0., shift], initial_Q, [0., 0., 0.]]
        lower = np.r_[[-np.pi, -np.pi / 2, -.15], np.full(3, -np.inf), np.full(3, -5.)]
        upper = np.r_[[np.pi, np.pi / 2, .15], np.full(3, np.inf), np.full(3, 5.)]
        fits.append(least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=.01,
                                  x_scale=[1, 1, .04, 1, 1, 1, 1, 1, 1], diff_step=1e-4,
                                  max_nfev=600, ftol=1e-9, xtol=1e-9, gtol=1e-9))
    fit = min(fits, key=lambda r: r.cost)
    errors = np.linalg.norm(residual(fit.x).reshape(-1, 3), axis=1)
    return {"lever_sensor_m": lever(fit.x[:2]).tolist(), "lever_length_m": float(length_m),
            "timing_shift_pc_s": float(fit.x[2]), "initial_rotation_vector": fit.x[3:6].tolist(),
            "gyro_bias_dps": fit.x[6:9].tolist(),
            "geometry_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "geometry_error_p50_p95_m": np.percentile(errors, [50, 95]).tolist(),
            "success": bool(fit.success), "cost": float(fit.cost),
            "solutions": [{"cost": float(r.cost), "lever_sensor_m": lever(r.x[:2]).tolist(),
                            "timing_shift_pc_s": float(r.x[2]), "gyro_bias_dps": r.x[6:9].tolist()} for r in fits]}


def estimate_window(imu, calibration, anchors, query_pc):
    """Estimate a hidden interval from bracketing visible material-point anchors.

    imu contains t_pc (N,), gyro_dps (N, 3), and acc_g (N, 3).
    Each anchor contains t_pc, imu_world_m, and lever_world_m; positions are metres.
    query_pc is a 1-D array on the recorded PC perf_counter axis. Calibration's
    attitude, gyro bias, and time shift must belong to this recording. Only the
    sensor-frame lever transfers unchanged between recordings of the same mount.
    Optional acceleration_scale and acceleration_offset_g apply a sensor-frame
    calibration: corrected_g = measured_g * scale + offset_g.
    """
    t = np.asarray(imu["t_pc"])
    shift = calibration["timing_shift_pc_s"]
    lever = np.asarray(calibration["lever_sensor_m"])
    at = np.asarray([r["t_pc"] for r in anchors])
    query_pc = np.asarray(query_pc)
    if len(anchors) < 6 or np.any(query_pc < at.min()) or np.any(query_pc > at.max()):
        raise ValueError("Queries require at least six bracketing visual anchors")
    start = max(0, int(np.searchsorted(t, at.min() - shift)) - 2)
    end = min(len(t), int(np.searchsorted(t, at.max() - shift)) + 3)
    times = t[start:end]
    bias = np.asarray(calibration["gyro_bias_dps"])
    rotations = integrate_attitude(times, imu["gyro_dps"][start:end] - bias)
    orientation = RotationSpline(times, rotations)
    center = float(at.mean())
    observed = np.asarray([r["lever_world_m"] for r in anchors])
    visible_marker = np.asarray([r["imu_world_m"] for r in anchors])
    inertial_levers = orientation(at - shift).apply(lever)
    force_g = imu["acc_g"][start:end]
    if "acceleration_scale" in calibration:
        force_g = force_g * np.asarray(calibration["acceleration_scale"]) + np.asarray(calibration["acceleration_offset_g"])
    inertial_force = rotations.apply(force_g * 9.80665)
    start_attitude = integrate_attitude(t[:start + 1], imu["gyro_dps"][:start + 1] - bias)[-1]
    initial = Rotation.from_rotvec(calibration["initial_rotation_vector"]) * start_attitude
    design = np.c_[np.ones(len(at)), at - center, .5 * (at - center) ** 2]
    augmented_design = np.vstack([design, [0, 0, .005]])

    def trajectory(parameters):
        Q = Rotation.from_rotvec(parameters[:3]) * initial
        rate = parameters[3:]
        corrections = Rotation.from_rotvec((times + shift - center)[:, None] * rate)
        acceleration = corrections.apply(Q.apply(inertial_force)) + [0, 0, -9.80665]
        velocity = cumulative_trapezoid(acceleration, times, axis=0, initial=0)
        position = cumulative_trapezoid(velocity, times, axis=0, initial=0)
        spline = CubicHermiteSpline(times, position, velocity)
        prior = spline(at - shift)
        # Eliminate linear translation parameters while attitude and position fit together.
        coefficients = np.linalg.lstsq(augmented_design, np.vstack([visible_marker - prior, np.zeros(3)]), rcond=None)[0]
        predicted_lever = Rotation.from_rotvec((at - center)[:, None] * rate).apply(Q.apply(inertial_levers))
        return Q, rate, spline, coefficients, predicted_lever, prior + design @ coefficients

    def residual(parameters):
        _, _, _, coefficients, predicted_lever, predicted_marker = trajectory(parameters)
        return np.r_[(predicted_lever - observed).ravel(), (predicted_marker - visible_marker).ravel(),
                     .03 * parameters[:3], .005 * parameters[3:], .005 * coefficients[2]]

    fit = least_squares(residual, np.zeros(6), loss="soft_l1", f_scale=.01,
                        x_scale=[1, 1, 1, 2, 2, 2], max_nfev=400)
    Q, rate, inertial_position, coefficients, predicted_lever, predicted_marker = trajectory(fit.x)
    dt = query_pc - center
    local_pose = orientation(query_pc - shift)
    correction = Rotation.from_rotvec(dt[:, None] * rate)
    world_lever = correction.apply(Q.apply(local_pose.apply(lever)))
    marker_position = inertial_position(query_pc - shift) + np.c_[np.ones(len(dt)), dt, .5 * dt ** 2] @ coefficients
    marker_velocity = inertial_position(query_pc - shift, 1) + coefficients[1] + dt[:, None] * coefficients[2]
    lever_velocity = correction.apply(Q.apply(local_pose.apply(np.cross(orientation(query_pc - shift, 1), lever))))
    lever_velocity += np.cross(rate, world_lever)
    return {"head_world_m": marker_position + world_lever, "head_velocity_world_mps": marker_velocity + lever_velocity,
            "marker_world_m": marker_position, "local_rate_correction_dps": np.rad2deg(rate),
            "acceleration_correction_mps2": coefficients[2],
            "pose_anchor_rmse_m": float(np.sqrt(np.mean(np.sum((predicted_lever - observed) ** 2, axis=1)))),
            "marker_anchor_rmse_m": float(np.sqrt(np.mean(np.sum((predicted_marker - visible_marker) ** 2, axis=1))))}


def calibrate_acceleration(imu, calibration, windows):
    """Fit shared axis scales/offsets from reference windows, separate from evaluation motions.

    Each window contains anchors, query_pc and reference head_world_m. References
    are training data here and must not also be reported as held-out accuracy.
    """
    def residual(parameters):
        candidate = {**calibration, "acceleration_scale": parameters[:3], "acceleration_offset_g": parameters[3:]}
        errors = [(estimate_window(imu, candidate, w["anchors"], w["query_pc"])["head_world_m"] - w["head_world_m"]).ravel()
                  / np.sqrt(len(w["query_pc"])) for w in windows]
        return np.r_[np.concatenate(errors), .15 * (parameters[:3] - 1), .10 * parameters[3:]]

    fit = least_squares(residual, np.r_[np.ones(3), np.zeros(3)],
                        bounds=(np.r_[np.full(3, .85), np.full(3, -.2)], np.r_[np.full(3, 1.15), np.full(3, .2)]),
                        x_scale=np.full(6, .05), diff_step=1e-4, max_nfev=100)
    if not fit.success:
        raise RuntimeError("Acceleration calibration did not converge: " + fit.message)
    return {"acceleration_scale": fit.x[:3].tolist(), "acceleration_offset_g": fit.x[3:].tolist()}
