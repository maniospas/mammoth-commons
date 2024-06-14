from mammoth.datasets import CSV
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric
from mammoth.models import onnx
from loader_data_csv_rankings import data_csv_rankings
from Rankings import RANKINGS
from Fairness_metrics_in_rankings import Fairness_metrics_in_rankings

@metric(namespace="mammotheu", version="v003", python="3.11")


def ExposureDistance(
    dataset: data_csv_rankings,
    model : RANKINGS.normal_ranking, 
    sensitive: str = 'Gender', 
    Protected_attirbute: str = 'Women',
    Non_protected_attribute: str = 'Men'
) -> Markdown:
    '''Compute the exposure distance  '''
    
    EDr = Fairness_metrics_in_rankings.Exposure_distance(dataset,model,sensitive,Protected_attirbute,Non_protected_attribute)

    return Markdown(text=str(EDr))
    
    
