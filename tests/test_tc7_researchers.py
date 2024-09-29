from mammoth import testing
from mammoth.models.empty import EmptyModel
from catalogue.dataset_loaders.data_csv_rankings import data_csv_rankings

from catalogue.model_loaders.compute_researcher_ranking import model_normal_ranking
from catalogue.model_loaders.compute_researcher_ranking import model_mitigation_ranking

from catalogue.metrics.ranking_fairness import ExposureDistance

def test_researchers_ranking_unfair():
    with testing.Env(data_csv_rankings, model_normal_ranking, model_mitigation_ranking, ExposureDistance) as env:
        dataset = env.data_csv_rankings(
            path="./data/researchers/Top_researchers.csv", 
            delimiter="|"
        )

        model_normal = env.model_normal_ranking()

        analysis_outcome_normal = env.ExposureDistance(dataset, model_normal, intro="ED for the normal ranking")
        analysis_outcome_normal.show(port=8000)


def test_researchers_ranking_fair():
    with testing.Env(data_csv_rankings, model_normal_ranking, model_mitigation_ranking, ExposureDistance) as env:
        dataset = env.data_csv_rankings(
            path="./data/researchers/Top_researchers.csv", 
            delimiter="|"
        )

        model_mitigation = env.model_mitigation_ranking()

        analysis_outcome_mitigation = env.ExposureDistance(dataset, model_mitigation, intro="ED for the mitigation ranking")
        analysis_outcome_mitigation.show(port=8001)



test_researchers_ranking_unfair()
test_researchers_ranking_fair()
