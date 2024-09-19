from mammoth.datasets import CSV
from mammoth.models import ONNX
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric
import fairbench as fb


@metric(namespace="maniospas", version="v007", python="3.11", packages=("fairbench",))
def model_card(
    dataset: CSV,
    model: ONNX,
    sensitive: List[str],
    prule: bool = True,
    worst_accuracy: bool = True,
) -> Markdown:
    """Creates a model card using FairBench.
    The created model card includes caveats and recommendations from a socio-technical database and
    may include the evaluation of several fairness stamps. All stamps consider all pairs of population
    groups or subgraphs.

    Args:
        prule: The worst ratio between group positive rates.
        worst_accuracy: The worst accuracy among groups.
    """
    for attr in sensitive:
        if attr not in dataset.categorical:
            raise Exception(
                "Fairness analysis not supported on non-categorical attributes"
            )

    # obtain predictions
    if hasattr(model, "predict_fair"):
        predictions = model.predict_fair(
            dataset.to_features(), dataset.to_features(sensitive)
        )
    else:
        predictions = model.predict(dataset.to_features())

    # declare sensitive attributes
    labels = dataset.labels
    sensitive = fb.Fork(
        {attr: fb.categories @ dataset.data[attr] for attr in sensitive}
    )

    if labels is None:
        report = fb.multireport(predictions=predictions, sensitive=sensitive)
        stamps = fb.combine(
            fb.stamps.prule(report),
            fb.stamps.four_fifths(report),
        )
        text = fb.modelcards.tomarkdown(stamps)
    else:
        text = ""
        for label in labels:
            # TODO: the following analysis is only for one class label
            report = fb.multireport(
                predictions=predictions, labels=label, sensitive=sensitive
            )
            stamps = fb.combine(
                fb.stamps.prule(report),
                fb.stamps.accuracy(report),
                fb.stamps.four_fifths(report),
                fb.stamps.dfpr(report),
                fb.stamps.dfnr(report),
                fb.stamps.auc(report),
                fb.stamps.abroca(report),
            )
        text += fb.modelcards.tomarkdown(stamps)
    return Markdown(text)
