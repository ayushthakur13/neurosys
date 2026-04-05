from __future__ import annotations

from collections import defaultdict

import numpy as np


def build_trajectories(z: np.ndarray, block_ids: list[str]) -> dict[str, list[np.ndarray]]:
    trajectories: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, b in enumerate(block_ids):
        trajectories[b].append(z[i])
    return trajectories


def latent_velocity(z: np.ndarray) -> np.ndarray:
    if len(z) < 2:
        return np.zeros((0, z.shape[1]), dtype=z.dtype)
    return z[1:] - z[:-1]


def trajectory_risk_score(current_velocity: np.ndarray, failure_velocities: np.ndarray) -> float:
    if len(current_velocity) == 0 or len(failure_velocities) == 0:
        return 0.0
    cv = np.mean(current_velocity, axis=0)
    fv = np.mean(failure_velocities, axis=0)
    num = float(np.dot(cv, fv))
    den = float(np.linalg.norm(cv) * np.linalg.norm(fv) + 1e-8)
    cosine = num / den
    return max(0.0, cosine)
