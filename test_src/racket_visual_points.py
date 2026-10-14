"""Fixed material-point extraction and multi-camera geometry for recorded video."""

import json
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np


def red_marker(panel, seed):
    """Locate compact red paint near the expected grip-mounted IMU."""
    sx, sy = np.rint(seed).astype(int)
    x0, y0, x1, y1 = sx - 20, sy - 20, sx + 21, sy + 21
    if x0 < 0 or y0 < 0 or x1 > panel.shape[1] or y1 > panel.shape[0]:
        return []
    patch = panel[y0:y1, x0:x1].astype(float)
    blue, green, red = patch.transpose(2, 0, 1)
    excess = red - np.maximum(green, blue)
    mask = (red > 35) & (excess > 10) & (excess > .18 * red)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    result = []
    yy, xx = np.indices(mask.shape)
    for label in range(1, count):
        _, _, width, height, area = stats[label]
        if not 2 <= area <= 65 or max(width, height) > 16:
            continue
        weights = np.where(labels == label, excess, 0)
        point = np.array([np.sum(weights * xx), np.sum(weights * yy)]) / weights.sum() + [x0, y0]
        distance = np.linalg.norm(point - seed)
        if distance <= 18:
            result.append({"xy": point.tolist(), "score": float(weights.sum() / area - .4 * distance)})
    return sorted(result, key=lambda r: r["score"], reverse=True)[:3]


def feature(gray, seed, bright, polygon=None):
    sx, sy = np.rint(seed).astype(int)
    x0, y0, x1, y1 = sx - 16, sy - 16, sx + 17, sy + 17
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        return []
    patch = gray[y0:y1, x0:x1].astype(float)
    small = cv2.GaussianBlur(patch, (0, 0), .6)
    large = cv2.GaussianBlur(patch, (0, 0), 2.5)
    score = small - large if bright else large - small
    yy, xx = np.indices(patch.shape)
    radius = 13 if bright else 7
    allowed = (xx - 16) ** 2 + (yy - 16) ** 2 < radius ** 2
    allowed &= (small > 55) & (large < 185) if bright else (small < 110)
    if bright:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((patch > 60).astype(np.uint8))
        compact = np.zeros(count, dtype=bool)
        for label in range(1, count):
            _, _, width, height, area = stats[label]
            compact[label] = 2 <= area <= 25 and max(width, height) <= 8
        allowed &= compact[labels]
    if polygon is not None:
        interior = np.zeros(patch.shape, np.uint8)
        cv2.fillConvexPoly(interior, cv2.convexHull(np.rint(polygon - [x0, y0]).astype(np.int32)), 1)
        allowed &= interior != 0
    score = np.where(allowed, score - .4 * np.hypot(xx - 16, yy - 16), -1e6)
    result = []
    for _ in range(3 if bright else 1):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        strength = float(score[y, x])
        if strength < (20 if bright else 15):
            break
        region = (xx - x) ** 2 + (yy - y) ** 2 < 3 ** 2
        weights = np.where(region & allowed, np.maximum((small - large if bright else large - small) - 10, 0), 0)
        point = np.array([np.sum(weights * xx), np.sum(weights * yy)]) / weights.sum() + [x0, y0]
        if bright or np.linalg.norm(point - seed) <= 4:
            result.append({"xy": point.tolist(), "score": strength})
        score[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = -1e6
    return result


class VisualPoints:
    def __init__(self, calibration_path):
        self.cameras = {}
        for sn, c in json.loads(Path(calibration_path).read_text())["cameras"].items():
            K, D, R, t = [np.asarray(c[k], float) for k in ["K", "D", "R_world", "t_world"]]
            self.cameras[sn] = (K, D, K @ np.c_[R, t], cv2.Rodrigues(R)[0], t)

    def triangulate(self, measured):
        rows = []
        for sn, xy in measured.items():
            K, D, P, _, _ = self.cameras[sn]
            point = np.asarray(xy, float) * 2 + .5
            uv = cv2.undistortPoints(point.reshape(1, 1, 2), K, D, P=K).ravel()
            rows.extend([uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]])
        _, _, vt = cv2.SVDecomp(np.asarray(rows))
        return vt[-1, :3] / vt[-1, 3] / 1000

    def project(self, xyz, sn):
        K, D, _, rv, t = self.cameras[sn]
        uv, _ = cv2.projectPoints(np.asarray(xyz).reshape(1, 3) * 1000, rv, t, K, D)
        return (uv.ravel() - .5) / 2

    def locate(self, candidates, head=None):
        best = None
        for sa, sb in combinations(candidates, 2):
            for a, b in product(candidates[sa], candidates[sb]):
                xyz = self.triangulate({sa: a["xy"], sb: b["xy"]})
                if not (-6 < xyz[0] < 6 and -1 < xyz[1] < 18 and .1 < xyz[2] < 3.8):
                    continue
                if head is not None and not .20 < np.linalg.norm(head - xyz) < .65:
                    continue
                inliers, errors = {}, []
                for sn, values in candidates.items():
                    projection = self.project(xyz, sn)
                    distances = [float(np.linalg.norm(projection - v["xy"])) for v in values]
                    i = int(np.argmin(distances))
                    if distances[i] < 1.5:
                        inliers[sn] = values[i]["xy"]
                        errors.append(distances[i])
                if len(inliers) < 2:
                    continue
                rank = (len(inliers), -float(np.mean(errors)))
                if best is None or rank > best[0]:
                    best = (rank, inliers)
        if best is None:
            return None
        measured = best[1]
        xyz = self.triangulate(measured)
        errors = {sn: float(np.linalg.norm(self.project(xyz, sn) - xy)) for sn, xy in measured.items()}
        if max(errors.values()) >= 1.5:
            return None
        return {"xyz_m": xyz.tolist(), "pixels": measured, "errors_panel_px": errors}
