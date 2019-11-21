from random import random

import click
import numpy as np

from .canvas import Canvas
from .particle import Particle
from .universe import Universe

bg_color = (0, 16, 40)
resolution = (720, 720)


@click.command(context_settings=dict(max_content_width=120))
@click.option(
    '--fps',
    help='FPS',
    default=60,
    show_default=True,
)
@click.option(
    '-s',
    '--suns',
    help='Number of suns',
    default=100,
    show_default=True,
)
@click.option(
    '--preview/--batch',
    help='Display preview window',
    default=True,
    show_default=True,
)
@click.version_option()
def cli(fps, suns, preview):
    with Canvas(
        resolution,
        fps,
        px_per_unit=1,
        preview=preview,
        render=True,
    ) as canvas:
        universe = Universe(canvas, bg_color)
        for _ in range(suns):
            universe.add_particle(
                Particle(
                    pos=(2 * np.random.rand(2) - 1) * universe.canvas.size * 0.8,
                    vel=(2 * np.random.rand(2) - 1) * universe.canvas.size * 0.05,
                    mass=20.0 * random(),
                    color=255 - np.array(bg_color),
                    shape="(",
                    canvas=universe.canvas,
                ),
            )
        universe.loop(max_ticks=fps * 60)
