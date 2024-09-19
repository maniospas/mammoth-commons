import pandas as pd
from mammoth.models.model import Model
from loader_data_csv_rankings import data_csv_rankings


class Ranking(Model):
    def __init__(self, path: str):
        self.model_url = path

    def normal_ranking(self, path, ranking_variable="citations"):
        """Ranking without considering any prottected attribute and just one variable"""

        dataframe_ranking = normal_ranking(dataset, variable)

        dataframe = data_csv_rankings(path)
        rankend_dataframe = dataframe.sort_values(ranking_variable)
        rankend_dataframe["Ranking"] = [
            i + 1 for i in range(rankend_dataframe.shape[0])
        ]

        return rankend_dataframe
