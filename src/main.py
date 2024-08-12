import logging

from tsp.model import Model
from tsp.solver import LpSolver

logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    model = Model(size=(50, 50), random_seed=10)
    logging.debug(model.size)

    model.set_nodes_randomly(10)

    logging.debug(model.nodes)

    solver = LpSolver()
    answer = solver.solve(model)

    logging.debug(answer)


if __name__ == "__main__":
    main()
