from mammoth.exports import Markdown, HTML
from mammoth.integration import metric
from mammoth.models.researcher_ranking import ResearcherRanking
from catalogue.dataset_loaders.data_csv_rankings import data_csv_rankings
from mammoth.datasets.csv import CSV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import statistics


def b(k):
    """Function defining the position bias: the highest ranked candidates receive more attention from users than candidates at lower ranks, and here is adoptedwith algorithmic discount with smooth reduction and favorable theoretical properties (https://proceedings.mlr.press/v30/Wang13.html)."""
    return 1 / np.log2(k + 1)


def Exposure_distance(
    dataset, model, ranking_variable, sensitive_attribute, protected_attirbute
):
    """Exposure distance to see where are the two groups located in the ranking"""

    if callable(model):         # HACK!
        Dataframe_ranking = model(dataset, ranking_variable)
    else:
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
                    "Ranking_" + ranking_variable
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


@metric(namespace="csh", version="v002", python="3.11")
def ExposureDistance(
    dataset: CSV,
    model: ResearcherRanking,
    sensitive: str = "Gender",
    protected: str = "female",
    sampling_attribute: str = "Nationality_IncomeGroup",
    ranking_variable: str = "Degree",
    intro: str = "",
) -> Markdown:
    """Compute the exposure distance"""

    EDr = Exposure_distance(
        dataset=dataset,
        model=model,
        protected_attirbute=protected,
        sensitive_attribute=sensitive,
        ranking_variable=ranking_variable,
        sampling_attribute=sampling_attribute,
    )

    the_text = f"{intro} is {str(EDr)}"

    return Markdown(text=str(the_text))

def boxplots_rankings(dataframe, hue_variable, ranking_variable, y_variable):
    # Set figure size based on number of categories
    n_categories = len(dataframe[y_variable].unique())
    height = min(7, max(4, n_categories * 0.5))  # Adaptive height
    
    plt.rcParams['figure.autolayout'] = True 
    
    # Create figure with adjusted size
    fig, ax = plt.subplots(
        figsize=(8, height),
        constrained_layout=True
    )
    
    # Create boxplot with refined styling
    sns.boxplot(
        data=dataframe,
        x=ranking_variable,
        y=y_variable,
        hue=hue_variable,
        order=sorted(dataframe[y_variable].unique()),
        saturation=0.7,
        linewidth=0.75,
        fliersize=3,  # Smaller outlier points
        ax=ax
    )
    
    # Refine the plot style
    ax.spines[['right', 'top']].set_visible(False)
    
    # Adjust labels and ticks
    ax.tick_params(axis='both', labelsize=9)
    ax.tick_params(axis='x', rotation=0)
    
    # Move legend to a better position if there's room
    if height > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust margins
    plt.margins(y=0.02)
    
    # Save and encode
    plt.close(fig)
    return get_base64_encoded_image(fig)

def boxplots_mitigation_strategies_pretty(
    ER_Old, ER_Mitigation, Method, sampling_attribute=None, n_runs=1
):
    """Compare the old results with possible mitigation strategies"""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams['figure.autolayout'] = True 
    
    width = 0.6
    font_size_out = 14
    
    fig, axes = plt.subplots(
        figsize=(10, 7),
        constrained_layout=True
    )
    
    Colors_boxplots = {"Statistical_parity": "darkblue", "Equal_parity": "gold"}
    
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
            **PROPS,
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
            **PROPS,
        )

    # Add scatter plots
    if sampling_attribute == None:
        plt.scatter(0, ER_Old, color="purple", s=60, alpha=0.7)
    else:
        plt.scatter(range(len(ER_Old)), list(ER_Old.values()), color="purple", s=60, alpha=0.7)
    
    # Style the axes
    for spine in ["right", "top"]:
        axes.spines[spine].set_visible(False)
    
    # Adjust tick parameters
    axes.tick_params("x", size=5, colors="black", labelsize=11, rotation=45)  # Reduced rotation
    axes.tick_params("y", size=2, colors="black", labelsize=11)
    
    # Label axes
    axes.set_ylabel("Exposure distance women\nposition vs men position", size=12, labelpad=10)
    axes.set_xlabel(" ", size=0)
    
    # Add grid lines
    y_ticks = [float(str(i).split(", ")[1]) for i in axes.get_yticklabels()][2:-1]
    for l in y_ticks:
        if sampling_attribute == None:
            axes.hlines(l, -0.5, 0.5, "darkgrey", lw=1, ls="--")
        else:
            axes.hlines(l, -0.5, len(ER_Old) - 0.5, "darkgrey", lw=1, ls="--")
    
    # Adjust margins to prevent cutoff
    plt.margins(y=0.1)
    
    # Save and encode
    plt.close(fig)
    enc_str = get_base64_encoded_image(fig)
    return enc_str

# Function to generate a base64 string from a matplotlib plot
def get_base64_encoded_image(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return img_str



def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode()

def generate_group_metrics_rows(ER_Old, ER_Mitigation, n_runs):
    rows = []
    for group in ER_Old.keys():
        mitigation_values = [ER_Mitigation[group][r] for r in range(n_runs)]
        mean_fair = sum(mitigation_values) / len(mitigation_values)
        std_fair = statistics.stdev(mitigation_values) if len(mitigation_values) > 1 else 0
        
        row = f"""
        <tr>
            <td>{group}</td>
            <td>{ER_Old[group]:.2f}</td>
            <td>{mean_fair:.2f}</td>
            <td>{std_fair:.2f}</td>
        </tr>
        """
        rows.append(row)
    return "\n".join(rows)

def generate_group_stats(dataset, sampling_attribute):
    stats = []
    unique_values = [x for x in dataset[sampling_attribute].unique() if pd.notna(x)]
    for group in sorted(unique_values):
        count = len(dataset[dataset[sampling_attribute] == group])
        stats.append(f"<p>{group}: {count} researchers</p>")
    return "\n".join(stats)

# Heinous
template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }}
        .csh-container {{
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 20px;
        }}
        .parameters, .dataset-info {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
        }}
        .main-content {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .figure-caption {{
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            font-size: 0.9em;
        }}
        .caption-definition {{
            margin-bottom: 10px;
            font-style: italic;
        }}
        .caption-elements {{
            margin-top: 10px;
        }}
        .caption-element {{
            margin: 5px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .element-marker {{
            width: 20px;
            height: 20px;
            display: inline-block;
        }}
        .dot-marker {{
            background: purple;
            border-radius: 50%;
        }}
        .boxplot-marker {{
            background: #4682b4;
        }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .visualization-full {{
            grid-column: 1 / -1;
        }}
        .visualization-half {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }}
        .visualization-half img {{
            max-height: 300px;
            width: auto;
            object-fit: contain;
            margin: auto;
        }}
        .network-visualization img {{
            max-width: 500px;
            display: block;
            margin: auto;
        }}
        .exposure-distance-visualization img {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
        }}
        .network-stats {{
            font-size: 0.9em;
            margin: 10px 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .network-stat {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
        }}
        .section-title {{
            margin: 20px 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 2px solid #007bff;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="csh-container">
        <div class="parameters">
            <h3>Initial parameters</h3>
            <div class="protected-attributes">
                <h4>Protected attributes:</h4>
                <div>Sensitive attribute: {sensitive_attribute}</div>
                <div>Protected value: {protected_attribute}</div>
                <div>Sampling attribute: {sampling_attribute}</div>
            </div>
            <div class="ranking-metrics">
                <h4>Ranking variable:</h4>
                <div>{ranking_variable}</div>
            </div>
        </div>

        <div class="main-content">
            <div class="visualization-full network-visualization">
                <h3 class="section-title">1. Citation Network Structure</h3>
                <img src="data:image/png;base64,{network_img_str}" alt="Citation Network" style="width: 100%;"/>
                <div class="network-stats">
                    <div class="network-stat">Nodes: 1739</div>
                    <div class="network-stat">Edges: 9943</div>
                    <div class="network-stat">Density: 0.003</div>
                    <div class="network-stat">LCC: 0.65</div>
                    <div class="network-stat">CC: 529</div>
                </div>
            </div>
            
            <div class="visualization-full">
                <h3 class="section-title">2. Regional Distribution</h3>
                <img src="data:image/png;base64,{normal_distribution_img_str}" alt="Regional Citation Distribution" style="width: 100%;"/>
                <div class="figure-caption">
                    Distribution of {ranking_variable} across world regions, separated by gender.
                </div>
                <img src="data:image/png;base64,{distribution_img_str}" alt="Regional Distribution" style="width: 100%;"/>
                <div class="figure-caption">
                    Post-Mitigation Distribution of {ranking_variable} across world regions, separated by gender.
                </div>
            </div>

            <h3 class="section-title">3. Exposure Distance Analysis</h3>
            <div class="visualization-full exposure-distance-visualization">
                <img src="data:image/png;base64,{er_viz_str}" alt="Exposure Distance Visualization" />
                <div class="figure-caption">
                    <div class="caption-definition">
                        The Exposure Distance (ED) measures how fair is the visibility of researchers from different demographic groups in rankings. 
                        In this plot, we compare the position of each woman vs. each man inside the income group categories, and then we average those values. 
                        A value of ED closer to 0 means a fairer representation, and we show how using a mitigation strategy (statistical parity in this case) 
                        improves the metric,, making it smaller.
                    </div>
                    <div class="caption-elements">
                        <div class="caption-element">
                            <span class="element-marker dot-marker"></span>
                            Purple dots show the Exposure Distance when researchers are ranked by raw degree centrality
                        </div>
                        <div class="caption-element">
                            <span class="element-marker boxplot-marker"></span>
                            Box plots show the distribution of Exposure Distance across {n_runs} runs of the fairness-aware ranking algorithm
                        </div>
                    </div>
                </div>
            </div>

            <div class="metrics">
                <h3>Results</h3>
                
                <h4>Exposure Distance by Group</h4>
                <table class="metrics-table">
                    <tr>
                        <th>Group</th>
                        <th>Original ED</th>
                        <th>Mean Fair ED</th>
                        <th>Std Dev Fair ED</th>
                    </tr>
                    {group_metrics_rows}
                </table>

                <h4>Statistical Summary</h4>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Original</th>
                        <th>Fair (Mean)</th>
                    </tr>
                    <tr>
                        <td>Max ED Disparity</td>
                        <td>{max_disparity_old:.2f}</td>
                        <td>{max_disparity_new:.2f}</td>
                    </tr>
                    <tr>
                        <td>Std Dev Across Groups</td>
                        <td>{std_dev_old:.2f}</td>
                        <td>{std_dev_new:.2f}</td>
                    </tr>
                </table>
            </div>
        </div>

        <div class="dataset-info">
            <h3>Analysis Information</h3>
            <p><strong>Number of runs:</strong> {n_runs}</p>
            <p><strong>Method:</strong> Statistical Parity</p>
            
            <h4>Group Statistics:</h4>
            {group_stats}
        </div>
    </div>
</body>
</html>
"""

def generate_html_report(dataset, ER_Old, ER_Mitigation, boxplot_img_str, 
                        network_path, normal_distribution_img_str, distribution_img_str,
                        sensitive_attribute, protected_attribute,
                        sampling_attribute, ranking_variable, n_runs):
    # Calculate summary statistics
    max_disparity_old = max(ER_Old.values()) - min(ER_Old.values())
    mean_mitigation_by_group = {
        group: statistics.mean([ER_Mitigation[group][r] for r in range(n_runs)])
        for group in ER_Old.keys()
    }
    max_disparity_new = max(mean_mitigation_by_group.values()) - min(mean_mitigation_by_group.values())
    
    # Get base64 strings for the images
    network_img_str = image_to_base64(network_path)
    
    # Generate HTML content
    html_content = template.format(
        sensitive_attribute=sensitive_attribute,
        protected_attribute=protected_attribute,
        sampling_attribute=sampling_attribute,
        ranking_variable=ranking_variable,
        er_viz_str=boxplot_img_str,
        network_img_str=network_img_str,
        normal_distribution_img_str=normal_distribution_img_str,
        distribution_img_str=distribution_img_str,
        group_metrics_rows=generate_group_metrics_rows(ER_Old, ER_Mitigation, n_runs),
        max_disparity_old=max_disparity_old,
        max_disparity_new=max_disparity_new,
        std_dev_old=statistics.stdev(list(ER_Old.values())),
        std_dev_new=statistics.stdev(list(mean_mitigation_by_group.values())),
        n_runs=n_runs,
        group_stats=generate_group_stats(dataset, sampling_attribute)
    )
    
    return HTML(html_content)




def validate_input(dataset, model, n_runs, sensitive, protected, sampling_attribute, ranking_variable):
    if not isinstance(n_runs, int) or not (1 <= n_runs <= 100):
        raise ValueError("n_runs must be an integer between 1 and 100")

    required_columns = [sensitive, sampling_attribute, ranking_variable]
    missing_columns = [col for col in required_columns if col not in dataset.data.columns]
    
    if missing_columns:
        raise ValueError(f"The following columns are missing in the dataset: {', '.join(missing_columns)}")
    
    if protected not in dataset.data[sensitive].values:
        raise ValueError(f"The protected value '{protected}' is not present in the sensitive column '{sensitive}'")

@metric(namespace="csh", version="v002", python="3.11")
def exposure_distance_comparison(
    dataset: CSV,
    model: ResearcherRanking,
    model_baseline: ResearcherRanking = None,
    n_runs: int = 1,
    sensitive: str = "Gender",
    protected: str = "female",
    sampling_attribute: str = "Nationality_IncomeGroup",
    ranking_variable: str = "Degree",
) -> HTML:
    """Compute the exposure distance and return it without any markup"""

    validate_input(dataset, model, n_runs, sensitive, protected, sampling_attribute, ranking_variable)

    if not model_baseline:
        # initialize our own baseline model
        model_baseline = model.baseline_rank
        print("Was not provided a baseline model.  Using default")

    # Only consider those rows where the sampling attribute is not missing
    data = dataset.data
    dataframe_sampling = data[~data[sampling_attribute].isnull()]

    #sampling_attribute = "Nationality_IncomeGroup"
    Old_ranking_variable = ranking_variable
    sensitive_attribute = sensitive
    protected_attribute = protected

    ER_Old = {}
    ER_Mitigation = {}
        
    for category in sorted(set(dataframe_sampling[sampling_attribute])):
        dataframe_filtered = dataframe_sampling[dataframe_sampling[sampling_attribute]==category]
        print(f"{len(dataframe_filtered)} researchers in the category {category}")

        # Compute the exposure distance for the normal ranking
        ER_Old[category] = Exposure_distance(
            dataframe_filtered,
            model_baseline,
            ranking_variable=Old_ranking_variable,
            sensitive_attribute=sensitive_attribute,
            protected_attirbute=protected_attribute,
        )

        ER_Mitigation[category] = {}
        for r in range(n_runs):
            ER_Mitigation[category][r] = Exposure_distance(
                dataframe_filtered,
                model,
                ranking_variable=Old_ranking_variable,
                sensitive_attribute=sensitive_attribute,
                protected_attirbute=protected_attribute,
            )


    # In order to show the boxplots with rankings for the full dataset
    # i.e., not "per category"
    if callable(model_baseline):         # HACK!
        original_dataset_ranked = model_baseline(dataframe_sampling, ranking_variable)
        mitigation_dataset_ranked = model_baseline(dataframe_filtered, ranking_variable)
    else:
        original_dataset_ranked = model_baseline.rank(dataframe_sampling, ranking_variable)

    # In order to show the boxplots with rankings for the full dataset
    # i.e., not "per category"
    if callable(model):         # HACK!
        mitigation_dataset_ranked = model(dataframe_sampling, ranking_variable)
    else:
        mitigation_dataset_ranked = model.rank(dataframe_sampling, ranking_variable)
    
    # We now have three dictionaries: ER_Old, ER_Mitigation
    mitigation_strategies_image = boxplots_mitigation_strategies_pretty(
        ER_Old,
        ER_Mitigation,
        Method="Statistical_parity",
        sampling_attribute=sampling_attribute,
        n_runs=n_runs,
    )

    normal_distribution_image = boxplots_rankings(
        original_dataset_ranked,
        hue_variable=sensitive_attribute,
        y_variable=sampling_attribute,
        ranking_variable="Ranking_"+Old_ranking_variable
    )

    distribution_image = boxplots_rankings(
        mitigation_dataset_ranked,
        hue_variable=sensitive_attribute,
        y_variable=sampling_attribute,
        ranking_variable="Ranking_"+Old_ranking_variable
    )


    print("Mitigation experiments_done")

    # HACK
    network_path = "./data/researchers/network.png"

    # Generate the complete HTML report
    return generate_html_report(
        dataset=data,
        ER_Old=ER_Old,
        ER_Mitigation=ER_Mitigation,
        boxplot_img_str=mitigation_strategies_image,
        network_path=network_path,
        normal_distribution_img_str=normal_distribution_image,
        distribution_img_str=distribution_image,
        sensitive_attribute=sensitive,
        protected_attribute=protected,
        sampling_attribute=sampling_attribute,
        ranking_variable=ranking_variable,
        n_runs=n_runs
    )