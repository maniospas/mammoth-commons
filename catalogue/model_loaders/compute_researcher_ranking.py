from mammoth.integration import loader
from mammoth.models.researcher_ranking import ResearcherRanking

def normal_ranking(dataset, ranking_variable):
    ranked_dataset = dataset.sort_values(ranking_variable, ascending=False)
    ranked_dataset[f"Ranking_{ranking_variable}"] = [
        i + 1 for i in range(ranked_dataset.shape[0])
    ]
    return ranked_dataset

def Compute_mitigation_strategy(dataset, mitigation_method, ranking_variable, sensitive_attribute, protected_attribute):
    from random import choices

    """Function for several mitigation strategies"""

    Dataframe_ranking = dataset[~dataset[sensitive_attribute].isnull()]

    ERr_list = {}
    New_ranking_DDBB = {}

    Chosen_groups, Chosen_researchers = {}, {}

    sensitive = set(Dataframe_ranking[sensitive_attribute])

    Ranking_sets = {
        attribute: Dataframe_ranking[
            Dataframe_ranking[sensitive_attribute] == attribute
        ]
        for attribute in sensitive
    }

    non_protected_attribute = [i for i in sensitive if i != protected_attribute][0]
    Len_groups = Dataframe_ranking[sensitive_attribute].value_counts()

    if mitigation_method == "Statistical_parity":
        Chosen_groups = []
        Len_group_in_ranking = Len_groups
        for i in range(Dataframe_ranking.shape[0]):
            P_minority = Len_group_in_ranking[protected_attribute] / (
                Len_group_in_ranking[protected_attribute]
                + Len_group_in_ranking[non_protected_attribute]
            )
            Chosen_groups += [
                choices(
                    [protected_attribute, non_protected_attribute],
                    [P_minority, 1 - P_minority],
                )[0]
            ]
            Len_group_in_ranking[Chosen_groups[-1]] -= 1
    elif mitigation_method == "Equal_parity":
        P_minority = 0.5
    elif mitigation_method == "Updated_statistical_parity":
        print("In construction")
    elif mitigation_method == "Internal_group_fairness":
        print("In construction")

    Positions = {
        non_protected_attribute: [
            i for i, j in enumerate(Chosen_groups) if j == non_protected_attribute
        ],
        protected_attribute: [
            i for i, j in enumerate(Chosen_groups) if j == protected_attribute
        ],
    }

    Chosen_researchers = {
        i_ranking: Ranking_sets[non_protected_attribute].iloc[i_position]["id"]
        for i_position, i_ranking in enumerate(Positions[non_protected_attribute])
    }
    for i_position, i_ranking in enumerate(Positions[protected_attribute]):
        Chosen_researchers[i_ranking] = Ranking_sets[protected_attribute].iloc[
            i_position
        ]["id"]

    New_ranking = {r: i for i, r in Chosen_researchers.items()}

    Dataframe_ranking["New_" + ranking_variable] = [
        New_ranking[i] + 1 for i in Dataframe_ranking.id
    ]

    return Dataframe_ranking


def mitigation_ranking(dataset, ranking_variable, mitigation_method="Statistical_parity", sensitive_attribute="Gender", protected_attribute="female"):
    return Compute_mitigation_strategy(
        dataset,
        mitigation_method,
        ranking_variable,
        sensitive_attribute,
        protected_attribute
    )

@loader(namespace="csh", version="v001", python="3.11")
def model_normal_ranking() -> ResearcherRanking:
    """This is a Normal Ranking loader"""

    return ResearcherRanking(normal_ranking)


@loader(namespace="csh", version="v001", python="3.11")
def model_mitigation_ranking(sampling_attribute: str = None) -> ResearcherRanking:
    """This is a fair Ranking loader without Sampling"""

    return ResearcherRanking(mitigation_ranking)