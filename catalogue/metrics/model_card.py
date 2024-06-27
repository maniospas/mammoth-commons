from mammoth.datasets import Dataset
from mammoth.models import Model
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric
import fairbench as fb


@metric(
    namespace="maniospas",
    version="v005",
    python="3.11",
    packages=("fairbench",)
)
def model_card(
    dataset: Dataset,
    model: Model,
    sensitive: List[str],
) -> Markdown:
    """Creates a model card using FairBench."""
    for attr in sensitive:
        if attr not in dataset.categorical:
            raise Exception(
                "Fairness analysis not supported on non-categorical attributes"
            )

    # obtain predictions
    if hasattr(model, "predict_fair"):
        predictions = model.predict_fair(dataset.to_features(), dataset.to_features(sensitive))
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
    else:
        # TODO: the following analysis is only for one class label
        report = fb.multireport(
            predictions=predictions, labels=labels[list(labels)[0]], sensitive=sensitive
        )
        stamps = fb.combine(
            fb.stamps.prule(report),
            fb.stamps.accuracy(report),
            fb.stamps.four_fifths(report),
        )
    return Markdown(fb.modelcards.tomarkdown(stamps))
