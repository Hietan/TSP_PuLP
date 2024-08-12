from __future__ import annotations

import time

import numpy as np


class InvalidSizeError(ValueError):
    def __init__(self) -> None:
        super().__init__("Size must be an int or a tuple of two ints")


class Model:
    def __init__(
        self, size: int | tuple[int, int], random_seed: int | None = None
    ) -> None:
        self._set_size(size)
        if random_seed is None:
            random_seed = int(time.time())
        self._random = np.random.default_rng(random_seed)

    def _set_size(self, value: int | tuple[int, int]) -> None:
        dimensions = 2
        if isinstance(value, int):
            self.size = (value, value)
        elif isinstance(value, tuple) and len(value) == dimensions:
            self.size = value
        else:
            raise InvalidSizeError

    def set_nodes_randomly(self, n: int) -> None:
        col0 = self._random.integers(0, self.size[0] + 1, size=n)
        col1 = self._random.integers(0, self.size[1] + 1, size=n)
        self.nodes = np.column_stack((col0, col1))

    def get_nodes_count(self) -> int:
        return self.nodes.shape[0]

    def get_distance_matrix(self) -> np.ndarray:
        v = self.get_nodes_count()
        d = np.zeros((v, v))
        for i in range(v):
            for j in range(v):
                d[i, j] = np.linalg.norm(self.nodes[i] - self.nodes[j])
        return d

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        self._size = value

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @nodes.setter
    def nodes(self, value: np.ndarray) -> None:
        self._nodes = value
