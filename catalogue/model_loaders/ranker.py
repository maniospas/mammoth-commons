from mammoth.datasets import CSV
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric
from mammoth.integration import loader
from mammoth.models import onnx
from loader_data_csv_rankings import data_csv_rankings
from Rankings import normal_ranking


@loader(namespace="mammotheu", version="v003", python="3.11")
def normal_ranking_model(
    path: str,
    ranking_variable: str,
) -> normal_ranking:
    """This is a Ranking loader"""
    return normal_ranking(path, ranking_variable)
