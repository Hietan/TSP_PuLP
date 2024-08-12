import logging

from tsp.model import Model

logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    model = Model(size=(10, 20))
    logging.debug(model.size)

    model.set_nodes_randomly(10)

    logging.debug(model.nodes)


if __name__ == "__main__":
    main()
