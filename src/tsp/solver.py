import pulp

from abc import ABC, abstractmethod
import numpy as np

from tsp.model import Model

class UnsolvableError(ValueError):
    def __init__(self) -> None:
        super().__init__('No optimal solution found')

class Solver(ABC):
    def __init__(self, name : str = None) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def solve(self, model : Model) -> np.ndarray:
        pass

class LpSolver(Solver):
    def __init__(self, name : str = 'TSP', solver_name : str = None) -> None:
        super().__init__(name)
        self.solver = solver_name

    def solve(self, model : Model) -> np.ndarray:
        problem_info = self.get_lp_problem(model)
        status = problem_info['problem'].solve(self.solver)

        if status != pulp.LpStatusOptimal:
            raise UnsolvableError()

        return self.get_solution(problem_info['variable']['y'])

    def get_lp_problem(self, model : Model) -> pulp.LpProblem:
        V_size = model.get_nodes_count()
        V = range(V_size)
        d = model.get_distance_matrix()

        problem = pulp.LpProblem('TSP', sense=pulp.LpMinimize)
        x = [[pulp.LpVariable(f'x_{u}_{v}', cat=pulp.LpBinary) for v in V] for u in V]
        y = [pulp.LpVariable(f'y_{v}', cat=pulp.LpInteger, lowBound=0, upBound=V_size - 1) for v in V]

        # Objective function
        problem += pulp.lpSum(d[u][v] * x[u][v] for v in V for u in V if u != v), 'Total Distance'

        # Constraints
        for v in V:
            problem += pulp.lpSum(x[u][v] for u in V if u != v) == 1, f'Enter_{v}'

        for u in V:
            problem += pulp.lpSum(x[u][v] for v in V if u != v) == 1, f'Leave_{u}'

        for u in range(1, V_size):
            for v in range(1, V_size):
                if u != v:
                    problem += y[u] - y[v] + V_size * x[u][v] <= V_size - 1, f'Subtour_{u}_{v}'

        for v in V:
            problem += x[v][v] == 0, f'Diagonal_{v}'

        problem += y[0] == 0, 'Start'

        return {
            'problem': problem,
            'variable': {
                'x': x,
                'y': y
            }
        }

    def get_solution(self, y: list[pulp.LpVariable]) -> np.ndarray:
        y = list(map(lambda var: int(pulp.value(var)), y))

        solution = np.zeros(len(y), dtype=int)
        for i, value in enumerate(y):
            solution[value] = i

        return solution

    @property
    def solver(self) -> pulp.LpSolver | None:
        return self._solver

    @solver.setter
    def solver(self, solver_name : str = None) -> None:
        match solver_name:
            case 'CPLEX' if pulp.CPLEX_CMD().available():
                self._solver = pulp.CPLEX_CMD(msg=False)
            case 'Gurobi' if pulp.GUROBI_CMD().available():
                self._solver = pulp.GUROBI_CMD(msg=False)
            case _ if pulp.PULP_CBC_CMD().available():
                self._solver = pulp.PULP_CBC_CMD(msg=False)
            case _:
                self._solver = None

