from __future__ import annotations

import numpy as np


class VectorizedEngine:
    """NumPy-broadcasting physics engine. Vectorizes pairwise gravity; collisions resolved sequentially."""

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

        pos = self._pos[:n]
        vel = self._vel[:n]
        mass = self._mass[:n]

        # Euler integrate positions
        pos += dt * vel

        # Pairwise distance matrix: diff[i,j] = pos[j] - pos[i]
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (n, n, 2)
        dist = np.linalg.norm(diff, axis=2)  # (n, n)
        np.fill_diagonal(dist, np.inf)

        # Collision detection
        widths = mass**0.5
        thresholds = (widths[:, np.newaxis] + widths[np.newaxis, :]) / 2.0
        collision_mask = dist < thresholds

        # Greedy collision resolution (sequential — merges are order-dependent)
        collision_pairs = np.argwhere(np.triu(collision_mask, k=1))
        removed: set[int] = set()
        for i, j in collision_pairs:
            if i in removed or j in removed:
                continue
            total_mass = mass[i] + mass[j]
            pos[i] = (pos[i] * mass[i] + pos[j] * mass[j]) / total_mass
            vel[i] = (vel[i] * mass[i] + vel[j] * mass[j]) / total_mass
            mass[i] = total_mass
            removed.add(j)

        # Vectorized gravity on surviving particles
        if removed:
            alive = np.array([i for i in range(n) if i not in removed])
        else:
            alive = np.arange(n)
        n_alive = len(alive)

        if n_alive >= 2:
            a_pos = pos[alive]
            a_mass = mass[alive]

            a_diff = a_pos[np.newaxis, :, :] - a_pos[:, np.newaxis, :]  # a_diff[i,j] = pos[j] - pos[i]
            a_dist = np.linalg.norm(a_diff, axis=2)
            np.fill_diagonal(a_dist, np.inf)

            # force[i,j] = diff[i,j] * (m_i * m_j) / dist[i,j]^2
            force_mag = (a_mass[:, np.newaxis] * a_mass[np.newaxis, :]) / (a_dist**2)
            forces = a_diff * force_mag[:, :, np.newaxis]  # (n_alive, n_alive, 2)

            # Net force on each particle: sum over j (upper triangle already has sign via diff direction)
            # force[i,j] = direction i->j, force[j,i] = direction j->i, so net = sum over all j
            net_force = np.sum(forces, axis=1)  # (n_alive, 2)

            vel[alive] += dt * net_force / a_mass[:, np.newaxis]

        # Compact arrays
        if removed:
            keep = np.array([i for i in range(n) if i not in removed])
            self._pos[: len(keep)] = self._pos[keep]
            self._vel[: len(keep)] = self._vel[keep]
            self._mass[: len(keep)] = self._mass[keep]
            self._n = len(keep)

        return sorted(removed, reverse=True)
