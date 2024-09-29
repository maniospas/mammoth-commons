from mammoth.exports import Markdown
from mammoth.integration import metric
from mammoth.models.node_ranking import NodeRanking
from catalogue.dataset_loaders.data_csv_rankings import data_csv_rankings
from mammoth.datasets.csv import Dataset
import numpy as np

def b(k):
    '''Function defining the position bias: the highest ranked candidates receive more attention from users than candidates at lower ranks, and here is adoptedwith algorithmic discount with smooth reduction and favorable theoretical properties (https://proceedings.mlr.press/v30/Wang13.html).'''
    return 1 / np.log2(k + 1)

def Exposure_distance(
                        dataset,
                        model,
                        ranking_variable,
                        sensitive_attribute,
                        protected_attirbute):
    '''Exposure distance to see where are the two groups located in the ranking'''
    Dataframe_ranking = model.rank(dataset, ranking_variable)

    # Remove rows with missing values in the sensitive attribute
    # e.g.: If sensitive_attribute is "Gender", remove rows where Gender is missing or NaN or None
    Dataframe_ranking = Dataframe_ranking[
        ~Dataframe_ranking[sensitive_attribute].isnull()
    ]

    rankings_per_attribute = {}
    sensitive = list(set(Dataframe_ranking[sensitive_attribute]))
    try:
        assert len(sensitive) == 2

        for attribute_value in sensitive:
            rankings_per_attribute[attribute_value] = list(
                Dataframe_ranking[Dataframe_ranking[sensitive_attribute] == attribute_value][
                    ranking_variable
                ]
            )

        non_protected_attribute = [i for i in sensitive if i != protected_attirbute][0]

        ranking_position_protected_attribute = [
            b(1 / (r + 1)) for r in rankings_per_attribute[protected_attirbute]
        ]
        ranking_position_non_protected_attribute = [
            b(1 / (r + 1)) for r in rankings_per_attribute[non_protected_attribute]
        ]

        Min_size = min(
            len(ranking_position_protected_attribute),
            len(ranking_position_non_protected_attribute),
        )
        EDr = np.round(
            (
                sum(ranking_position_protected_attribute[:Min_size])
                - sum(ranking_position_non_protected_attribute[:Min_size])
            ),
            2,
        )
    except:
        EDr = np.nan
    return EDr

    
@metric(namespace="mammotheu", version="v003", python="3.11")
def ExposureDistance(
        dataset: Dataset,
        model: NodeRanking,
        sensitive: str = 'Gender',
        protected: str = 'female',
        sampling: str = 'Nationality_IncomeGroup',
        ranking_variable: str = 'Degree',
        intro: str = ''
) -> Markdown:
    '''Compute the exposure distance  '''

    EDr = Exposure_distance(
        dataset=dataset, 
        model=model, 
        protected_attirbute=protected,
        sensitive_attribute=sensitive,
        ranking_variable=ranking_variable,
    )

    the_text = f"{intro} is {str(EDr)}"

    return Markdown(text=str(the_text))