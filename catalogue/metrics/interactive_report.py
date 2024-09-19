from mammoth.datasets import CSV
from mammoth.models import ONNX
from mammoth.exports import HTML
from typing import Dict, List
from mammoth.integration import metric
import fairbench as fb


@metric(namespace="maniospas", version="v003", python="3.11", packages=("fairbench",))
def interactive_report(
    dataset: CSV,
    model: ONNX,
    sensitive: List[str],
) -> HTML:
    """Creates a model card using FairBench."""
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
        {attr + " ": fb.categories @ dataset.data[attr] for attr in sensitive}
    )

    if labels is None:
        report = fb.multireport(predictions=predictions, sensitive=sensitive)
    else:
        report = fb.Fork(
            {
                label
                + " class ": fb.multireport(
                    predictions=predictions, labels=label, sensitive=sensitive
                )
                for label in labels
            }
        )
    return HTML(fb.interactive_html(report, show=False))
