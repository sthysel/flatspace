from random import random

import click
import numpy as np

from .canvas import Canvas
from .particle import Particle
from .universe import Universe

bg_color = (0, 16, 40)
resolution = (720, 720)
fps = 60


def cli():
    with Canvas(
            resolution,
            fps,
            px_per_unit=1,
            preview=True,
            render=True,
    ) as canvas:
        u = Universe(canvas, bg_color)
        for _ in range(400):
            u.add_particle(
                Particle(pos=(2 * np.random.rand(2) - 1) * u.canvas.size * 0.8,
                         vel=(2 * np.random.rand(2) - 1) * u.canvas.size *
                         0.05,
                         mass=20.0 * random(),
                         color=255 - np.array(bg_color),
                         shape="(",
                         canvas=u.canvas),)
        u.loop(max_ticks=fps * 60)
