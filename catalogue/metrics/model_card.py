from mammoth.datasets import Dataset
from mammoth.models import Predictor
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric, Options
import fairbench as fb
import numpy as np


@fb.core.Transform
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
        values = (values - mn) / (mx - mn)
        return {f"fuzzy min ({mn:.3f})": 1 - values, f"fuzzy max ({mx:.3f})": values}
    return fb.categories @ iterable


@metric(namespace="maniospas", version="v008", python="3.11", packages=("fairbench",))
def model_card(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
) -> Markdown:
    """Creates a model card using FairBench. The card includes as many fairness stamps as
    applicable, and includes caveats and recommendations from a socio-technical database.
    All stamps summarize the behavior across population groups or subgraphs,
    where intersectional subgroups may be created for analysis of edge cases.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population.
    """

    text = ""

    # obtain predictions
    predictions = model.predict(dataset, sensitive)

    # declare sensitive attributes
    labels = dataset.labels
    sensitive = fb.Fork({attr: categories @ dataset.data[attr] for attr in sensitive})

    # change behavior based on arguments
    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = fb.multireport if compare_groups == "Pairwise" else fb.unireport
    # perform different analysis, depending on whether labels are provided
    if labels is None:
        report = report_type(predictions=predictions, sensitive=sensitive)
        stamps = fb.combine(
            fb.stamps.prule(report),
            fb.stamps.four_fifths(report),
        )
        text += fb.modelcards.tomarkdown(stamps)
    else:
        for label in labels:
            # TODO: the following analysis is only for one class label
            report = report_type(
                predictions=predictions,
                labels=(
                    labels[label].to_numpy()
                    if hasattr(labels[label], "to_numpy")
                    else labels[label]
                ),
                sensitive=sensitive,
            )
            stamps = fb.combine(
                fb.stamps.prule(report),
                fb.stamps.accuracy(report),
                fb.stamps.four_fifths(report),
                fb.stamps.dfpr(report),
                fb.stamps.dfnr(report),
                # fb.stamps.auc(report),
                # fb.stamps.abroca(report),
            )
        text += fb.modelcards.tomarkdown(stamps)

    if hasattr(dataset, "description"):
        text += "\n## Dataset\n"
        if isinstance(dataset.description, str):
            text += text + "\n"
        elif isinstance(dataset.description, dict):
            for key, value in dataset.description.items():
                text += "#### " + key + "\n" + value.replace("\n", "\n\n") + "\n"
        else:
            raise Exception(
                "Since the dataset's description field exist, it should be either string or dict from headers to descriptions"
            )
    return Markdown(text)
