from mammoth.integration import loader
from Rankings import normal_ranking


@loader(namespace="mammotheu", version="v003", python="3.11")

def normal_ranking_model(
    path: str,
    ranking_variable: str,
) -> normal_ranking:
    '''This is a Ranking loader
    '''
    return normal_ranking(path,ranking_variable)

