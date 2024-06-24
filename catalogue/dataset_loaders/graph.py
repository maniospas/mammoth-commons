import os.path

from mammoth.datasets import Graph
from mammoth.integration import loader


@loader(
    namespace="maniospas",
    version="v003",
    python="3.11",
    packages=("pygrank",)
)
def data_graph(
    path: str = "",
) -> Graph:
    """This loads the edges of a graph."""
    import pygrank as pg

    _, graph, communities = next(pg.load_datasets_multiple_communities([path]))
    return Graph(graph, communities)
