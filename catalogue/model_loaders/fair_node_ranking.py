from mammoth.models import NodeRanking
from mammoth.integration import loader


@loader(
    namespace="maniospas",
    version="v003",
    python="3.11",
    packages=("pygrank",)
)
def model_fair_node_ranking(
        diffusion: float = 0.85,
        redistribution: str = "original"
) -> NodeRanking:
    """This constructs a node ranking algorithm based on PageRank.
     The algorithm employs a diffusion parameter in the range [0, 1),
     and can either have a "none", "uniform" or "original" rank redistribution
     strategy to achieve fairness
    """
    import pygrank as pg

    assert redistribution in ["none", "original", "uniform"]
    assert diffusion >= 0
    assert diffusion < 1  # careful not to allow 1

    params = {"alpha": diffusion, "tol": 1.E-9, "max_iters": 3000}
    ranker = pg.PageRank(**params) if redistribution == "none" else pg.LFPR(redistributor=redistribution, **params)
    return NodeRanking(ranker >> pg.Normalize("max"))
