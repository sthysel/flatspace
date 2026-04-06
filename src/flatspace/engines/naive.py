from __future__ import annotations

import numpy as np
from numpy.linalg import norm


class NaiveEngine:
    """Python-loop physics engine. Same algorithm as the original Universe.tick()."""

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
        pos = self._pos
        vel = self._vel
        mass = self._mass

        # Euler integrate positions
        pos[:n] += dt * vel[:n]

        # Pairwise interactions
        removed: set[int] = set()
        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                diff = pos[j] - pos[i]
                dist = norm(diff)

                wi = mass[i] ** 0.5
                wj = mass[j] ** 0.5
                threshold = (wi + wj) / 2.0

                if dist < threshold:
                    # Inelastic merge j into i
                    total_mass = mass[i] + mass[j]
                    pos[i] = (pos[i] * mass[i] + pos[j] * mass[j]) / total_mass
                    vel[i] = (vel[i] * mass[i] + vel[j] * mass[j]) / total_mass
                    mass[i] = total_mass
                    removed.add(j)
                else:
                    force = diff * (mass[i] * mass[j]) / (dist**2)
                    vel[i] += dt * force / mass[i]
                    vel[j] -= dt * force / mass[j]

        # Compact arrays
        if removed:
            keep = np.array([i for i in range(n) if i not in removed])
            self._pos[: len(keep)] = self._pos[keep]
            self._vel[: len(keep)] = self._vel[keep]
            self._mass[: len(keep)] = self._mass[keep]
            self._n = len(keep)

        return sorted(removed, reverse=True)
