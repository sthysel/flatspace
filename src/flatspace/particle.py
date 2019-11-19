from math import isclose
from random import random

import numpy as np
from numpy import allclose, dot
from numpy.linalg import norm


def _normalize(x):
    """ maps x for [0, inf) to [0, 1) """
    return x / (x + 1)


def _ratio(a, b):
    """ returns a / b if a < b else b / a """
    return (a * b) / max(a, b) ** 2



def _compare(a, b):
    type_a = type(a)
    if type_a is not type(b):
        return False

    if type_a == float:
        return isclose(a, b)
    elif type_a == np.ndarray:
        return allclose(a, b)
    else:
        return (a == b)


class Particle():
    def __init__(self, pos, vel, mass, color, shape: {"[", "square", "(", "circle", ".", "dot"}="[", canvas=None):
        # physical properties
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.mass = mass
        #self.charge = charge

        # drawing related properties
        self.canvas = canvas
        self.color = np.array(color)
        self.shape = shape

    def __eq__(self, other):
        ret = True
        for attr in ["pos", "vel", "color", "mass", "shape"]:
            ret &= _compare(getattr(self, attr),
                            getattr(other, attr))
        return ret

    @property
    def width(self):
        return 1 if (self.shape == ".") else self.mass ** 0.5

    def draw(self):
        if self.canvas is not None:
            if self.shape in {"[", "square"}:
                self.canvas.draw_rect(self.pos, self.width, self.color)
            elif self.shape in {"(", "circle"}:
                self.canvas.draw_circle(self.pos, self.width // 2, self.color)
            elif self.shape in {".", "dot"}:
                self.canvas.draw_pixel(self.pos, self.color)
        else:
            raise ValueError("No canvas")

    def tick(self, dt):
        self.pos += dt * self.vel

    def apply_force(self, force, dt):
        self.vel += dt * force / self.mass

    def can_merge_with(self, p):
        return True
        # the bigger the difference in mass...
        mass_factor = 1 - _ratio(self.mass, p.mass)

        # the more parallel their velocities are...
        direction_factor = np.cross(self.vel, p.vel) ** 2 / (norm(self.vel) * norm(p.vel))

        # the more similar their speeds are...
        speed_factor = _ratio(norm(self.vel), norm(p.vel))

        # ... the bigger the chance of merging
        #return random() < (speed_factor * direction_factor * mass_factor) ** (1 / 3)
        return 0.5 < (speed_factor * direction_factor * mass_factor) ** (1 / 3)

    def collide_with(self, p):
        if self.can_merge_with(p):
            self.inelastic_collision(p)
            return True
        else:
            self.elastic_collision(p)
            return False

    def inelastic_collision(self, p):
        total_momentum = p.vel * p.mass + self.vel * self.mass
        total_mass = p.mass + self.mass

        # weighted average for position after merge
        self.pos = (self.pos * self.mass + p.pos * p.mass) / total_mass

        # see https://www.wikiwand.com/en/Inelastic_collision
        self.mass = total_mass
        self.vel = total_momentum / total_mass

    def elastic_collision(self, p):
        dpos = p.pos - self.pos
        dvel = p.vel - self.vel
        dist = norm(dpos)

        # see https://en.wikipedia.org/wiki/Elastic_collision
        x = 2 / (self.mass + p.mass) * dot(dvel, dpos) / (dist ** 2) * dpos
        self.vel += p.mass * x
        p.vel -= self.mass * x
