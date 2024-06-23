from mammoth.exports import Markdown
from mammoth.integration import metric
from loader_data_csv_rankings import data_csv_rankings
from Rankings import RANKINGS
import numpy as np


class Fairness_metrics_in_rankings:
    def __init__(self, path: str, EDr):
        self.model_url = path
        self.EDr = EDr

    def b(k):
        '''Function defining the position bias: the highest ranked candidates receive more attention from users than candidates at lower ranks, and here is adoptedwith algorithmic discount with smooth reduction and favorable theoretical properties (https://proceedings.mlr.press/v30/Wang13.html).'''
        return 1 / np.log2(k + 1)

    def Exposure_distance(self,
                          path,
                          model,
                          sensitive):
        '''Exposure distance to see where are the two groups located in the ranking'''
        assert len(sensitive) == 2

        dataset = data_csv_rankings(path)
        Dataframe_ranking = model(dataset, 'Value')

        rankings_per_attribute = {}
        for attribute_value in sensitive:
            rankings_per_attribute[attribute] = list(
                Dataframe_ranking[Dataframe_ranking[attribute] == attribute_value].Ranking)

        self.EDr = np.round((sum([self.b(1 / (r + 1)) for r in Rankings_per_attribute[Protected_attirbute]]) - sum(
            [self.b(1 / (r + 1)) for r in Rankings_per_attribute[Non_protected_attribute]])) / 2000, 2)

        return self.EDr


@metric(namespace="mammotheu", version="v003", python="3.11")
def ExposureDistance(
        dataset: data_csv_rankings,
        model: RANKINGS.normal_ranking,
        sensitive: str = 'Gender'
) -> Markdown:
    '''Compute the exposure distance  '''

    EDr = Fairness_metrics_in_rankings.Exposure_distance(dataset, model, sensitive)

    return Markdown(text=str(EDr))


