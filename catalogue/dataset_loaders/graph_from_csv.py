from mammoth.datasets import Graph_CSH
from mammoth.integration import loader


@loader(namespace="mauritzniklas",
        version="v002",
        python="3.11",
        packages=("pandas",))
def data_graph_csv(
    path_nodes: str = "", path_edges: str = "", attributes: list[str] = ""
) -> Graph_CSH:
    """This loads the nodes and edges of a graph.

    Args:
        path_nodes: A csv file containing the node identifiers as the first column.
        path_edges: A csv file, where each raw corresponds to a pair of node identifiers of an edge.
        attributes: A comma separated list of attributes to retain from the dataset.
    """
    import pandas as pd

    nodes = pd.read_csv(path_nodes)
    edges = pd.read_csv(path_edges)
    if isinstance(attributes, str):
        attributes = [attribute.strip() for attribute in attributes.split(",")]
    return Graph_CSH(nodes, edges, attributes)
