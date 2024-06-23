from mammoth.integration import loader
from mammoth.models.ranking import Ranking


@loader(
    namespace="mammotheu",
    version="v003",
    python="3.11"
)
def normal_ranking_model(
    path: str,
) -> Ranking:
    """This is a Ranking loader"""

    return Ranking(path)
