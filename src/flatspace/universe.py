from __future__ import annotations

import numpy as np

from .engines import Engine
from .particle import Particle


class Universe:
    def __init__(self, canvas, bg_color, engine: Engine, fg_color=None, particles=0):
        self.MERGE_COUNTER = 0
        self.canvas = canvas
        self.bg_color = np.array(bg_color)
        self.engine = engine

        self.particles: list[Particle] = []
        if isinstance(particles, int):
            for _ in range(particles):
                self.add_particle(
                    Particle(
                        pos=(2 * np.random.rand(2) - 1) * self.canvas.size * 0.8,
                        vel=(2 * np.random.rand(2) - 1) * self.canvas.size * 0,
                        mass=1.0,
                        color=fg_color or (255 - self.bg_color),
                        canvas=self.canvas,
                    )
                )
        else:
            for p in particles:
                if fg_color is not None:
                    p.color = fg_color
                p.canvas = self.canvas
                self.add_particle(p)

    def add_particle(self, p: Particle) -> None:
        self.particles.append(p)
        self.engine.add_particle(np.array(p.pos, dtype=np.float64), np.array(p.vel, dtype=np.float64), float(p.mass))

    def draw(self):
        self.canvas.fill(self.bg_color)
        for p in self.particles:
            p.draw()
        self.canvas.update()

    def tick(self, dt):
        removed_indices = self.engine.tick(dt)

        # Remove merged particle objects (indices are descending, so pop is safe)
        for idx in removed_indices:
            self.particles.pop(idx)
            self.MERGE_COUNTER += 1

        # Sync physics state from engine arrays to particle objects for drawing
        for i, p in enumerate(self.particles):
            p.pos = self.engine.positions[i].copy()
            p.mass = float(self.engine.masses[i])

    def loop(self, terminate_on_last_particle=False, max_ticks=None):
        i = 0
        while True:
            self.draw()
            self.tick(1 / self.canvas.fps)
            if terminate_on_last_particle and (len(self.particles) == 1):
                break
            if i == max_ticks:
                break
            i += 1
