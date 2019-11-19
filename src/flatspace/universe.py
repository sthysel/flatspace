from itertools import combinations
from random import random

import numpy as np
from numpy.linalg import norm

from .particle import Particle


class Universe():
    def __init__(self, canvas, bg_color, fg_color=None, particles=0):
        self.MERGE_COUNTER = 0
        self.canvas = canvas
        self.bg_color = np.array(bg_color)

        # particle init
        self.particles = []
        if isinstance(particles, int):
            for _ in range(particles):
                self.add_particle(Particle(
                    pos=(2 * np.random.rand(2) - 1) * self.canvas.size * 0.8,
                    vel=(2 * np.random.rand(2) - 1) * self.canvas.size * 0,
                    mass=1.0,
                    color=fg_color or (255 - self.bg_color),
                    canvas=self.canvas))
        else:
            for p in particles:
                if fg_color is not None:
                    p.color = fg_color
                p.canvas = self.canvas
                self.add_particle(p)

        # draw
        self.draw()

    def add_particle(self, p):
        self.particles.append(p)

    def remove_particle(self, p):
        for i in range(len(self.particles)):
            if p == self.particles[i]:
                self.particles.pop(i)
                break

    def draw(self):
        self.canvas.fill(self.bg_color)
        for p in self.particles:
            p.draw()
        self.canvas.update()

    def tick(self, dt):
        for p1 in self.particles:
            p1.tick(dt)

        for p1, p2 in combinations(self.particles, r=2):
            diff = p2.pos - p1.pos
            dist = norm(diff)

            if dist < (p1.width + p2.width) / 2:  # collision
                self.MERGE_COUNTER -= 1
                if p1.collide_with(p2):  # p1 merged with p2
                    self.MERGE_COUNTER += 2
                    self.remove_particle(p2)
            else:  # gravitational influence
                force = diff * (p1.mass * p2.mass) / (dist ** 2) 
                p1.apply_force(+force, dt)
                p2.apply_force(-force, dt)

    def loop(self,
             terminate_on_last_particle=False,
             max_ticks=None):
        max_particles = len(self.particles)
        i = 0
        while True:
            self.draw()
            self.tick(1 / self.canvas.fps)
            if terminate_on_last_particle and (len(self.particles) == 1):
                break
            if i == max_ticks:
                break
            i += 1
