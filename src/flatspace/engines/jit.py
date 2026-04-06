from __future__ import annotations

import numba  # type: ignore[unresolved-import]
import numpy as np


@numba.njit(cache=True)
def _tick_kernel(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, n: int, dt: float) -> np.ndarray:
    """JIT-compiled physics kernel. Returns boolean array marking removed particles."""
    # Euler integrate positions
    for i in range(n):
        pos[i, 0] += dt * vel[i, 0]
        pos[i, 1] += dt * vel[i, 1]

    removed = np.zeros(n, dtype=numba.boolean)

    # Pairwise interactions (interleaved collision + gravity, same as naive)
    for i in range(n):
        if removed[i]:
            continue
        for j in range(i + 1, n):
            if removed[j]:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = (dx * dx + dy * dy) ** 0.5

            wi = mass[i] ** 0.5
            wj = mass[j] ** 0.5
            threshold = (wi + wj) / 2.0

            if dist < threshold:
                # Inelastic merge j into i
                total_mass = mass[i] + mass[j]
                pos[i, 0] = (pos[i, 0] * mass[i] + pos[j, 0] * mass[j]) / total_mass
                pos[i, 1] = (pos[i, 1] * mass[i] + pos[j, 1] * mass[j]) / total_mass
                vel[i, 0] = (vel[i, 0] * mass[i] + vel[j, 0] * mass[j]) / total_mass
                vel[i, 1] = (vel[i, 1] * mass[i] + vel[j, 1] * mass[j]) / total_mass
                mass[i] = total_mass
                removed[j] = True
            else:
                inv_dist_sq = 1.0 / (dist * dist)
                fm = mass[i] * mass[j] * inv_dist_sq
                fx = dx * fm
                fy = dy * fm

                vel[i, 0] += dt * fx / mass[i]
                vel[i, 1] += dt * fy / mass[i]
                vel[j, 0] -= dt * fx / mass[j]
                vel[j, 1] -= dt * fy / mass[j]

    return removed


class JitEngine:
    """Numba JIT-compiled physics engine. First tick incurs ~1-3s compilation overhead (cached on disk)."""

    def __init__(self, capacity: int = 1024) -> None:
        self._pos = np.zeros((capacity, 2), dtype=np.float64)
        self._vel = np.zeros((capacity, 2), dtype=np.float64)
        self._mass = np.zeros(capacity, dtype=np.float64)
        self._n = 0

    def add_particle(self, pos: np.ndarray, vel: np.ndarray, mass: float) -> None:
        if self._n >= len(self._mass):
            self._grow()
        self._pos[self._n] = pos
        self._vel[self._n] = vel
        self._mass[self._n] = mass
        self._n += 1

    def _grow(self) -> None:
        new_cap = len(self._mass) * 2
        new_pos = np.zeros((new_cap, 2), dtype=np.float64)
        new_vel = np.zeros((new_cap, 2), dtype=np.float64)
        new_mass = np.zeros(new_cap, dtype=np.float64)
        new_pos[: self._n] = self._pos[: self._n]
        new_vel[: self._n] = self._vel[: self._n]
        new_mass[: self._n] = self._mass[: self._n]
        self._pos = new_pos
        self._vel = new_vel
        self._mass = new_mass

    @property
    def positions(self) -> np.ndarray:
        return self._pos[: self._n]

    @property
    def velocities(self) -> np.ndarray:
        return self._vel[: self._n]

    @property
    def masses(self) -> np.ndarray:
        return self._mass[: self._n]

    @property
    def count(self) -> int:
        return self._n

    def tick(self, dt: float) -> list[int]:
        n = self._n
        if n < 2:
            if n == 1:
                self._pos[0] += dt * self._vel[0]
            return []

        removed_mask = _tick_kernel(self._pos, self._vel, self._mass, n, dt)

        removed_indices = np.where(removed_mask)[0]
        if len(removed_indices) > 0:
            keep = np.where(~removed_mask)[0]
            self._pos[: len(keep)] = self._pos[keep]
            self._vel[: len(keep)] = self._vel[keep]
            self._mass[: len(keep)] = self._mass[keep]
            self._n = len(keep)

        return sorted(removed_indices.tolist(), reverse=True)
