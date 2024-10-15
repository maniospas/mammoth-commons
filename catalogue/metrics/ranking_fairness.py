from mammoth.exports import Markdown, HTML
from mammoth.integration import metric
from mammoth.models.node_ranking import NodeRanking
from catalogue.dataset_loaders.data_csv_rankings import data_csv_rankings
from mammoth.datasets.csv import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

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
    except Exception as e:
        print("Exception")
        EDr = np.nan
    return EDr

    
@metric(namespace="csh", version="v001", python="3.11")
def ExposureDistance(
        dataset: Dataset,
        model: NodeRanking,
        sensitive: str = 'Gender',
        protected: str = 'female',
        sampling_attribute: str = 'Nationality_IncomeGroup',
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
        sampling_attribute=sampling_attribute
    )

    the_text = f"{intro} is {str(EDr)}"

    return Markdown(text=str(the_text))

# Plotting exposure ratio:
def boxplots_mitigation_strategies(
    ER_Old, ER_Mitigation, Method, sampling_attribute=None, n_runs=1
):
    """Compare the old results with possible mitigation strategies"""

    plt.rcParams["mathtext.fontset"] = "dejavusans"
    width = 0.6
    font_size_out = 14
    nrows = 1
    ncols = 1

    Colors_boxplots = {"Statistical_parity": "darkblue", "Equal_parity": "gold"}

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(6 * ncols, 3 * nrows),
        sharex=True,
        sharey=False,
        gridspec_kw={"width_ratios": [1]},
    )
    PROPS = {
        "boxprops": {"facecolor": "none", "edgecolor": Colors_boxplots[Method]},
        "medianprops": {"color": Colors_boxplots[Method]},
        "whiskerprops": {"color": Colors_boxplots[Method]},
        "capprops": {"color": Colors_boxplots[Method]},
    }
    if sampling_attribute == None:
        ER_Mitigation_DF = pd.DataFrame(ER_Mitigation.values(), columns=["ER_run"])
        sns.boxplot(
            data=ER_Mitigation_DF,
            y="ER_run",
            color=Colors_boxplots[Method],
            saturation=0.3,
            linewidth=0.75,
            ax=axes,
            **PROPS
        )
    else:
        ER_Mitigation_DF = pd.DataFrame(
            {
                sampling_attribute: [
                    c for c in ER_Mitigation.keys() for n in range(n_runs)
                ],
                "ER_run": [n for c in ER_Mitigation.values() for n in c.values()],
            }
        )

        sns.boxplot(
            data=ER_Mitigation_DF,
            x=sampling_attribute,
            y="ER_run",
            color=Colors_boxplots[Method],
            saturation=0.3,
            linewidth=0.75,
            ax=axes,
            **PROPS
        )

    if sampling_attribute == None:
        plt.scatter(0, ER_Old, color="purple", s=60, alpha=0.7)
    else:
        plt.scatter(ER_Old.keys(), ER_Old.values(), color="purple", s=60, alpha=0.7)

    for spine in ["right", "top"]:
        axes.spines[spine].set_visible(False)

    axes.tick_params("x", size=3, colors="black", labelsize=13, rotation=90)
    axes.tick_params("y", size=2, colors="black", labelsize=12, rotation=0)

    axes.set_ylabel("Exposure distance women \n position vs men position", size=13)
    axes.set_xlabel(" ", size=0)
    # axes.set_ylim(0,16)

    # axes.set_title('ERr',  size=30)

    for l in [float(str(i).split(", ")[1]) for i in axes.get_yticklabels()][2:-1]:
        if sampling_attribute == None:
            axes.hlines(l, -0.5, 0.5, "darkgrey", lw=1, ls="--")

        else:
            axes.hlines(l, -0.5, len(ER_Old) - 0.5, "darkgrey", lw=1, ls="--")

    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    plt.close(fig)

    enc_str = get_base64_encoded_image(fig)
    return enc_str


# Function to generate a base64 string from a matplotlib plot
def get_base64_encoded_image(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return img_str

@metric(namespace="csh", version="v001", python="3.11")
def ExposureDistanceComparison(
        dataset: Dataset,
        model: NodeRanking,
        model_baseline: NodeRanking = None,
        n_runs: int = 1,
        sensitive: str = 'Gender',
        protected: str = 'female',
        sampling_attribute: str = 'Nationality_IncomeGroup',
        ranking_variable: str = 'Degree',
        intro: str = ''
) -> HTML:
    '''Compute the exposure distance and return it without any markup'''

    if not model_baseline:
        raise ValueError("provide a baseline model for comparison")

    # Only consider those rows where the sampling attribute is not missing
    dataframe_sampling = dataset[~dataset[sampling_attribute].isnull()]

    sampling_attribute = "Nationality_IncomeGroup"
    Old_ranking_variable = "Degree"
    sensitive_attribute ='Gender'
    protected_attribute = 'female'
    ER_Old = {}
    ER_Mitigation = {}
    New_ranking_DDBB = {}
        
    for category in set(dataframe_sampling[sampling_attribute]):
        dataframe_filtered = dataframe_sampling[dataframe_sampling[sampling_attribute]==category]

        # Compute the exposure distance for the normal ranking
        ER_Old[category] = Exposure_distance(
            dataframe_filtered, 
            model_baseline, 
            ranking_variable=Old_ranking_variable,
            sensitive_attribute=sensitive_attribute,
            protected_attirbute=protected_attribute
        )

        ER_Mitigation[category] = {}
        New_ranking_DDBB[category] = {}
        for r in range(n_runs):
            ER_Mitigation[category][r] = Exposure_distance(
                dataframe_filtered,
                model, 
                ranking_variable=Old_ranking_variable,
                sensitive_attribute=sensitive_attribute,
                protected_attirbute=protected_attribute
            )
    
    # We now have three dictionaries: ER_Old, ER_Mitigation, New_ranking_DDBB
    img_str = boxplots_mitigation_strategies(
        ER_Old, 
        ER_Mitigation, 
        Method="Statistical_parity",
        sampling_attribute=sampling_attribute,
        n_runs=n_runs
    ) 

    print('Mitigation experiments_done')
    # Embed the image into an HTML tag
    html_img = f'<img src="data:image/png;base64,{img_str}" />'
    return HTML(body=html_img)