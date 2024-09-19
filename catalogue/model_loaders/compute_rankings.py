from mammoth.integration import loader
from mammoth.models.node_ranking import NodeRanking


@loader(namespace="maniospas", version="v003", python="3.11")
def model_normal_ranking(
    path: str,
) -> NodeRanking:
    """This is a Ranking loader"""

    return NodeRanking(path)
