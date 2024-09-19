from mammoth.models import NodeRanking
from mammoth.integration import loader


@loader(namespace="maniospas",
        version="v003",
        python="3.11",
        packages=("pygrank",))
def model_fair_node_ranking(
    diffusion: float = 0.85,
    redistribution: str = "original"
) -> NodeRanking:
    """Constructs a node ranking algorithm based on PageRank.
    The algorithm employs a diffusion parameter in the range [0, 1),
    and can either have a none, uniform or original rank redistribution
    strategy to achieve fairness. This strategy transfers node score mass
    from over-represented groups of nodes to those with lesser average mass.

    Args:
        diffusion: The diffusion parameters of the corresponding PageRank algorithm.
        redistribution: The redistribution strategy. Can be none, uniform or original.
    """
    import pygrank as pg

    assert redistribution in ["none", "original", "uniform"], "Invalid node score redistribution strategy."
    diffusion = float(diffusion)
    assert diffusion >= 0, "The diffusion should be non-negative"
    assert diffusion < 1, "The diffusion should be <1"  # careful not to allow 1

    params = {"alpha": diffusion, "tol": 1.0e-9, "max_iters": 3000}
    ranker = (
        pg.PageRank(**params)
        if redistribution == "none"
        else pg.LFPR(redistributor=redistribution, **params)
    )
    return NodeRanking(ranker >> pg.Normalize("max"))
