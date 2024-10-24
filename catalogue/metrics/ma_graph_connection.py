from mammoth.datasets import Graph_CSH
from mammoth.models.empty import EmptyModel
from mammoth.exports import Markdown

from multisoc.infer import aux_functions, inference
from typing import List
from mammoth.integration import metric


@metric(
    namespace="mauritzniklas",
    version="v001",
    python="3.11",
    packages=("multisoc",)
)
def connection_properties(
    dataset: Graph_CSH,
    model: EmptyModel,  # TODO: seems we cannot give a default model!
    sensitive: list[str],
) -> Markdown:
    """
    Performs analysis of connection properties in a graph.
    If no sensitive attributes are provided, all node column attributes are considered
    sensitive.
    """

    if len(sensitive) == 0:
        sensitive = dataset.cols
    assert all(
        [attr in dataset.cols for attr in sensitive]
    ), "All sensitive attributes must be in the dataset."

    n, counts = aux_functions.get_n_and_counts(
        dataset.nodes_df, dataset.edges_df, sensitive
    )
    df = inference.create_table(n, counts)
    md = df.to_markdown()
    return Markdown(md)
