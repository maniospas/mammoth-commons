from mammoth.datasets import Dataset
from mammoth.models import Predictor
from mammoth.exports import HTML
from typing import Dict, List
from mammoth.integration import metric, Options
import fairbench as fb
import numpy as np


@fb.v1.Transform
def categories(iterable):
    # print(iterable)
    is_numeric = True
    values = list()
    for value in iterable:
        try:
            values.append(float(value))
        except Exception:
            is_numeric = False
            break
    if is_numeric:
        values = np.array(values)
        mx = values.max()
        mn = values.min()
        if mx == mn:
            raise Exception(
                "Numerical sensitive attribute has the same value everywhere"
            )
        values = fb.v1.tobackend((values - mn) / (mx - mn))
        return {f"fuzzy min ({mn:.3f})": 1 - values, f"fuzzy max ({mx:.3f})": values}
    return fb.categories @ iterable


@metric(
    namespace="mammotheu",
    version="v0037",
    python="3.11",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def model_card(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
    minimum_shown_deviation: float = 0.1,
) -> HTML:
    """Creates a report that uses the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. The card includes several types of fairness/bias assessment and you can view
    it either as a) a summary table of results, b) a model card that does not have too many measures but contains
    socio-technical concerns about those shown, or c) a full report.

    The reported values summarize model behavior across all population groups.
    Multiple sensitive attributes may be present, such as gender, age, and race.
    Furthermore, each of those attributes may obtain multiple values, as happens when multiple genders or
    races are considered. Numeric attributes, like age, are normalized to
    the range [0,1] and we consider the result as truth values of membership to the group of the maximum
    value - as opposed to membership to the group with minimum value.
    A different set of stamps is computed for each prediction label.

    You may optionally analyse intersectional subgroups. In this case,
    a separate subgroup is created for each combination of sensitive attribute values. Many of those groups will have
    few members if there are too many attributes, and empty groups are ignored during the analysis.
    The generated report may be followed by details about out-of-the-box datasets.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute, but may also be computationally intensive if too many group intersections are selected.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population. For example, if the 4/5ths rule stamp is applicable, it computes positive rates and obtains the minimum ratio, either across all pairs of groups (for pairwise comparison) or otherwise between each group and the total population.
        minimum_shown_deviation: Show only results where the deviation from ideal values exceeds the given threshold. If nothing is shown, it does not mean that fairness is achieved, but this is a good way to identify the most prominent biases. If value of 0 is set (default) then all results are shown.
    """

    assert len(sensitive) != 0, "At least one sensitive attribute should be selected"
    predictions = model.predict(dataset, sensitive)
    labels = dataset.labels
    sensitive = fb.Dimensions({attr: categories @ dataset.data[attr] for attr in sensitive})

    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = fb.reports.pairwise if compare_groups == "Pairwise" else fb.reports.vsall

    if labels is not None and hasattr(labels, "columns"):
        labels = labels[labels.columns[0]]
        """labels = fb.Dimensions({
            label: labels[label].to_numpy()
            if hasattr(labels[label], "to_numpy")
            else labels[label]
            for label in labels.columns
        })"""

    report = report_type(predictions=predictions, labels=labels, sensitive=sensitive)
    minimum_shown_deviation = float(minimum_shown_deviation)
    assert 0 <= minimum_shown_deviation <= 1, "Minimum shown deviation should be in the range [0,1]"
    if minimum_shown_deviation != 0:
        report = report.filter(fb.investigate.DeviationsOver(minimum_shown_deviation))

    views = {
        "Summary": report.show(env=fb.export.HtmlTable(view=False, filename=None)),
        "Stamps": report.filter(fb.investigate.Stamps).show(
            env=fb.export.Html(view=False, filename=None), depth=1
        ),
        "Full report": report.show(env=fb.export.Html(view=False, filename=None), depth=2),
    }
    # Generate tabbed HTML content
    tab_headers = "".join(
        f'<button class="tablinks" data-tab="{key}">{key}</button>' for key in views
    )
    tab_contents = "".join(
        f'<div id="{key}" class="tabcontent">{value}</div>' for key, value in views.items()
    )

    dataset_desc = ""
    if hasattr(dataset, "description"):
        dataset_desc += "<h1>Dataset</h1>"
        if isinstance(dataset.description, str):
            dataset_desc += dataset.description + "<br>"
        elif isinstance(dataset.description, dict):
            for key, value in dataset.description.items():
                dataset_desc += f"<h3>{key}</h3>" + value.replace("\n", "<br>") + "<br>"
        else:
            raise Exception("Dataset description must be a string or a dictionary.")

    html_content = f'''
       <style>
           .tablinks {{
               background-color: #ddd;
               padding: 10px;
               cursor: pointer;
               border: none;
               border-radius: 5px;
               margin: 5px;
           }}
           .tablinks:hover {{ background-color: #bbb; }}
           .tablinks.active {{ background-color: #aaa; }}

           .tabcontent {{
               display: none;
               padding: 10px;
               border: 1px solid #ccc;
           }}
           .tabcontent.active {{ display: block; }}
       </style>
       <script>
           document.addEventListener("DOMContentLoaded", function() {{
               const tabContainer = document.querySelector("div");
               tabContainer.addEventListener("click", function(event) {{
                   if (event.target.classList.contains("tablinks")) {{
                       let tabName = event.target.getAttribute("data-tab");

                       // Remove "active" from all tabs and contents
                       document.querySelectorAll(".tablinks").forEach(tab => tab.classList.remove("active"));
                       document.querySelectorAll(".tabcontent").forEach(content => content.classList.remove("active"));

                       // Activate the selected tab and content
                       event.target.classList.add("active");
                       document.getElementById(tabName).classList.add("active");
                   }}
               }});

               // Show the first tab by default
               let firstTab = document.querySelector(".tablinks");
               if (firstTab) {{
                   firstTab.classList.add("active");
                   document.getElementById(firstTab.getAttribute("data-tab")).classList.add("active");
               }}
           }});
       </script>
       <h1>Bias report</h1>
       <p>A report was computed over several prospective biases. 
       Many values are computed to paint a broad picture
       {'; set a minimum shown deviation parameter for this analysis to simplify what is shown.' if minimum_shown_deviation==0 else f', but for simplicity only those that differ at least {minimum_shown_deviation:.3f} from their ideal values are shown; this is the minimum shown deviation parameter for the analysis.'}
       Results may not give the full picture, and not all biases may be harmful to the social context. Switch between different views.</p>
       <div>{tab_headers}</div>
       {tab_contents}
       {dataset_desc}
       '''

    return HTML(html_content)
