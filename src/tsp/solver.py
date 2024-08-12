from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pulp

if TYPE_CHECKING:
    from tsp.model import Model


class UnsolvableError(ValueError):
    def __init__(self) -> None:
        super().__init__("No optimal solution found")


class Solver(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def solve(self, model: Model) -> np.ndarray:
        pass


class LpSolver(Solver):
    def __init__(self, name: str = "TSP", solver_name: str | None = None) -> None:
        super().__init__(name)
        self.solver = solver_name

    def solve(self, model: Model) -> np.ndarray:
        problem_info = self.get_lp_problem(model)
        status = problem_info["problem"].solve(self.solver)

        if status != pulp.LpStatusOptimal:
            raise UnsolvableError

        return self.get_solution(problem_info["variable"]["y"])

    def get_lp_problem(self, model: Model) -> pulp.LpProblem:
        v_size = model.get_nodes_count()
        v_set = range(v_size)
        d = model.get_distance_matrix()

        problem = pulp.LpProblem("TSP", sense=pulp.LpMinimize)
        x = [
            [pulp.LpVariable(f"x_{u}_{v}", cat=pulp.LpBinary) for v in v_set]
            for u in v_set
        ]
        y = [
            pulp.LpVariable(
                f"y_{v}", cat=pulp.LpInteger, lowBound=0, upBound=v_size - 1
            )
            for v in v_set
        ]

        # Objective function
        problem += (
            pulp.lpSum(d[u][v] * x[u][v] for v in v_set for u in v_set if u != v),
            "Total Distance",
        )

        # Constraints
        for v in v_set:
            problem += pulp.lpSum(x[u][v] for u in v_set if u != v) == 1, f"Enter_{v}"

        for u in v_set:
            problem += pulp.lpSum(x[u][v] for v in v_set if u != v) == 1, f"Leave_{u}"

        for u in range(1, v_size):
            for v in range(1, v_size):
                if u != v:
                    problem += (
                        y[u] - y[v] + v_size * x[u][v] <= v_size - 1,
                        f"Subtour_{u}_{v}",
                    )

        for v in v_set:
            problem += x[v][v] == 0, f"Diagonal_{v}"

        problem += y[0] == 0, "Start"

        return {"problem": problem, "variable": {"x": x, "y": y}}

    def get_solution(self, y: list[pulp.LpVariable]) -> np.ndarray:
        y = [int(pulp.value(var)) for var in y]

        solution = np.zeros(len(y), dtype=int)
        for i, value in enumerate(y):
            solution[value] = i

        return solution

    @property
    def solver(self) -> pulp.LpSolver | None:
        return self._solver

    @solver.setter
    def solver(self, solver_name: str | None = None) -> None:
        match solver_name:
            case "CPLEX" if pulp.CPLEX_CMD().available():
                self._solver = pulp.CPLEX_CMD(msg=False)
            case "Gurobi" if pulp.GUROBI_CMD().available():
                self._solver = pulp.GUROBI_CMD(msg=False)
            case _ if pulp.PULP_CBC_CMD().available():
                self._solver = pulp.PULP_CBC_CMD(msg=False)
            case _:
                self._solver = None
