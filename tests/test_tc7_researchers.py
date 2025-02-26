from mammoth import testing
from catalogue.dataset_loaders.data_csv_rankings import data_csv_rankings
from catalogue.dataset_loaders.data_researchers import data_researchers

from catalogue.model_loaders.compute_researcher_ranking import model_normal_ranking
from catalogue.model_loaders.compute_researcher_ranking import model_mitigation_ranking

from catalogue.metrics.ranking_fairness import exposure_distance_comparison

def test_researchers_ranking_comparison():
    with testing.Env(
        data_researchers,
        model_mitigation_ranking,
        exposure_distance_comparison,
    ) as env:
        dataset = env.data_researchers(
            papers_path="./data/researchers/physics_papers.csv.tar.bz2",
            papers_affiliations="./data/researchers/affiliations.csv.tar.bz2",
        )

        model_mitigation = env.model_mitigation_ranking()

        analysis_outcome_mitigation = env.exposure_distance_comparison(
            dataset,
            model_mitigation,
            n_runs=10,
            sampling_attribute="Nationality_IncomeGroup",
            ranking_variable="Degree",
        )
        analysis_outcome_mitigation.show()


if __name__ == "__main__":
    test_researchers_ranking_comparison()