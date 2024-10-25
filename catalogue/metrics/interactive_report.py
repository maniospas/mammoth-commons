from mammoth.datasets import Dataset
from mammoth.models import Predictor
from mammoth.exports import HTML
from typing import Dict, List
from mammoth.integration import metric, Options
import fairbench as fb


@metric(namespace="maniospas", version="v005", python="3.11", packages=("fairbench",))
def interactive_report(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
) -> HTML:
    """Creates an interactive report using the FairBench library. The report creates traceable evaluations that
    you can shift through to find actual sources of unfairness.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population.
    """
    # obtain predictions
    predictions = model.predict(dataset, sensitive)

    # declare sensitive attributes
    labels = dataset.labels
    sensitive = fb.Fork(
        {attr + " ": fb.categories @ dataset.data[attr] for attr in sensitive}
    )

    # change behavior based on arguments
    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = fb.multireport if compare_groups == "Pairwise" else fb.unireport

    if labels is None:
        report = report_type(predictions=predictions, sensitive=sensitive)
    else:
        report = fb.Fork(
            {
                label
                + " ": report_type(
                    predictions=predictions,
                    labels=(
                        labels[label].to_numpy()
                        if hasattr(labels[label], "to_numpy")
                        else labels[label]
                    ),
                    sensitive=sensitive,
                )
                for label in labels
            }
        )
    return HTML(fb.interactive_html(report, show=False, name="Classes"))
