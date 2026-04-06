# Flatspace (Version 0.0.1)

A space simulator, based on Christoph Smithmyer's [Just
Dust](https://gitlab.com/chrismit3s/just-dust/tree/master/src)

![Orbit](docs/flatspace-20191120T011617.gif)


# Usage

```zsh
Usage: flatspace [OPTIONS]

Options:
  --fps INTEGER                   FPS  [default: 60]
  -s, --suns INTEGER              Number of suns  [default: 100]
  --preview / --batch             Display preview window  [default: preview]
  --engine [naive|numpy|numba]    Physics engine backend  [default: numpy]
  --version                       Show the version and exit.
  --help                          Show this message and exit.
```

# Install

```zsh
$ uv sync
```

For Numba JIT support:

```zsh
$ uv sync --extra numba
```

# Run

```zsh
$ uv run flatspace
$ uv run flatspace --fps 30 -s 50
$ uv run flatspace --batch
$ uv run flatspace --engine numba -s 200
```

# Implementation

Flatspace simulates gravitational interactions between particles in 2D space.
Particles attract each other via gravity, collide, and merge. The simulation
uses Euler integration for motion and renders output via OpenCV as a live
preview window and/or MP4 video file.

## Architecture

```
cli.py --> universe.py --> engines/{naive,vectorized,jit}.py
                 \            /
                  canvas.py
                      |
                  particle.py
```

- **cli.py** -- Click-based entry point. Creates particles with random
  positions, velocities, and masses, then runs the simulation loop for 60
  seconds.
- **universe.py** -- Simulation controller. Orchestrates the tick/draw loop,
  delegates physics computation to the selected engine, and syncs engine state
  back to particle objects for rendering.
- **particle.py** -- Rendering entity. Holds position, velocity, mass, color,
  and shape. Used by the canvas for drawing; physics is handled by the engine.
- **canvas.py** -- OpenCV rendering backend. Context manager that converts
  simulation coordinates to pixels (origin at center, y-axis flipped), outputs
  timestamped MP4 files, and optionally displays a live preview window.

## Physics

Each simulation tick performs:

1. **Position integration** -- Euler method: `pos += dt * vel`
2. **Pairwise interactions** -- For every unique pair of particles:
   - **Collision**: If the distance between two particles is less than the
     average of their widths (`width = sqrt(mass)`), they merge. The merge
     conserves momentum: the surviving particle inherits the combined mass and
     the mass-weighted average position and velocity.
   - **Gravity**: Otherwise, a gravitational force `F = diff * (m1 * m2) / dist^2`
     is applied to both particles, where `diff` is the displacement vector.

## Physics Engines

Three interchangeable backends implement the pairwise physics computation. All
store particle state as contiguous NumPy arrays (struct-of-arrays layout) for
cache efficiency. Select at runtime with `--engine`.

### naive

The baseline engine. Iterates over all particle pairs using nested Python loops,
computing gravity and resolving collisions inline. This reproduces the original
simulation algorithm exactly.

- **Algorithm**: Sequential nested loop over pairs. Collision and gravity are
  interleaved -- when a collision is detected the merge happens immediately and
  the absorbed particle is skipped for all subsequent pairs in that tick.
- **Complexity**: O(n^2) Python-level iterations with per-pair NumPy array
  operations.
- **When to use**: Debugging, correctness reference, or when avoiding extra
  dependencies and complexity is preferred.

### numpy (default)

Vectorized engine using NumPy broadcasting. The expensive O(n^2) pairwise
distance and gravity calculations are performed as bulk array operations,
eliminating Python loop overhead.

- **Algorithm**: Two-pass approach. First, a full pairwise distance matrix is
  computed via broadcasting. Collision pairs are extracted and resolved
  sequentially with greedy assignment (each particle merges at most once per
  tick). Second, gravitational forces for all surviving particle pairs are
  computed as a single vectorized operation and applied in bulk.
- **Trade-off**: The separation of collision and gravity passes means the
  execution order differs slightly from the naive engine (which interleaves
  them). For a chaotic n-body system with visual output this difference is
  negligible.
- **Complexity**: O(n^2) in NumPy C-level operations, with a small sequential
  loop only for collision resolution.
- **When to use**: Default choice. No additional dependencies beyond NumPy
  (already required). Provides significant speedup over naive for any particle
  count.

### numba

JIT-compiled engine using Numba's `@njit` decorator. The entire pairwise loop
(gravity + collision) is compiled to optimized machine code via LLVM, operating
directly on scalar values without Python object overhead or temporary array
allocation.

- **Algorithm**: Identical to the naive engine -- sequential nested loop with
  interleaved collision and gravity. The difference is that the loop body is
  compiled to native code, operating on raw float64 scalars instead of NumPy
  array objects.
- **First-run cost**: The JIT kernel is compiled on first invocation (~1-3
  seconds). The compiled code is cached to disk (`cache=True`) so subsequent
  runs start immediately.
- **Complexity**: O(n^2) in native machine code. Approximately 50-200x faster
  than the naive engine depending on particle count and hardware.
- **Dependency**: Requires `numba>=0.61` (optional). Install with
  `uv sync --extra numba`. If not installed, selecting `--engine numba` produces
  a clear error message.
- **When to use**: Large simulations (200+ particles) where frame rate matters,
  or batch rendering of long simulations.

# Refs

https://www.reddit.com/r/Python/comments/dxq4ea/this_is_one_of_the_most_interesting_outputs_of/
