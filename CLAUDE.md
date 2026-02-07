# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flatspace is a 2D gravitational n-body simulator. Particles ("suns") interact via gravity, collide, and merge. Visualization uses OpenCV for real-time preview and MP4 video export. Based on Christoph Smithmyer's "Just Dust" project.

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run simulator
uv run flatspace                        # 100 suns, 60 fps, preview window
uv run flatspace --fps 30 -s 50        # Custom fps and sun count
uv run flatspace --batch                # Render to file only (no preview window)

# Test
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files

# Type check
uv run ty check
```

## Architecture

All source lives in `src/flatspace/`. Four modules with a clear layered flow:

```
cli.py → universe.py → particle.py
                ↘        ↗
               canvas.py
```

- **`cli.py`** — Click-based entry point (`flatspace` command). Creates particles with random positions/velocities and runs the simulation loop for 60 seconds.
- **`universe.py`** — Simulation controller. Runs the tick loop: draws state, computes gravitational forces between all pairs (`itertools.combinations`), detects collisions by overlap, merges or bounces particles. Force: `F = diff * (m1 * m2) / dist²`.
- **`particle.py`** — Physics entity. Holds position, velocity, mass, color, shape. Euler integration for motion. Supports inelastic (merge, momentum-conserving) and elastic collisions. Shape types: `"("` circle, `"["` square, `"."` dot.
- **`canvas.py`** — OpenCV rendering. Context manager (`with Canvas(...) as c:`). Converts simulation coordinates to pixels (origin at center, y-axis flipped). Outputs timestamped MP4 files and optional live preview window. ESC or 'q' exits.

## Key Details

- Coordinate system origin is canvas center; y-axis is inverted for display
- Particle width scales as `mass**0.5` (dots are always 1px)
- Collision merge logic exists but the `should_merge` check is hardcoded to `True`
- No test files exist yet despite pytest being configured
- Video output: MP4V codec, filename pattern `flatspace-YYYYMMDDThhmmss.mp4`
- Dependencies: `click`, `numpy`, `opencv-python`
