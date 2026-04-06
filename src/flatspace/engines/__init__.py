from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Engine(Protocol):
    """Contract for physics backends."""

    @property
    def positions(self) -> np.ndarray: ...

    @property
    def velocities(self) -> np.ndarray: ...

    @property
    def masses(self) -> np.ndarray: ...

    @property
    def count(self) -> int: ...

    def add_particle(self, pos: np.ndarray, vel: np.ndarray, mass: float) -> None: ...

    def tick(self, dt: float) -> list[int]: ...


def create_engine(name: str, capacity: int = 1024) -> Engine:
    if name == "naive":
        from .naive import NaiveEngine

        return NaiveEngine(capacity)
    elif name == "numpy":
        from .vectorized import VectorizedEngine

        return VectorizedEngine(capacity)
    elif name == "numba":
        try:
            from .jit import JitEngine
        except ImportError:
            raise ImportError("Numba is required for the 'numba' engine. Install with: uv sync --extra numba") from None
        return JitEngine(capacity)
    else:
        raise ValueError(f"Unknown engine: {name!r}. Choose from: naive, numpy, numba")
