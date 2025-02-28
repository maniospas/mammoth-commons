import pandas as pd

from mammoth.datasets import Dataset
from mammoth.models import Predictor
from mammoth.exports import HTML
from typing import List
from mammoth.integration import metric
from aif360.metrics import BinaryLabelDatasetMetric, MDSSClassificationMetric
from aif360.detectors import bias_scan


@metric(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=(
        "aif360",
        "aif360[OptimalTransport]",
        "pandas",
        "onnxruntime",
        "ucimlrepo",
        "pygrank",
    ),
)
def bias_scan(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
) -> HTML:
    """Performs a scan for biased attributes by comparing their fraction of positive predictions
    to the dataset. It is up to the user to determine which of the biased attributes should be considered sensitive.

    <i><b>License:</b> The following description is adapted from AIF360
    (<a href="https://github.com/Trusted-AI/AIF360">https://github.com/Trusted-AI/AIF360</a>),
    which is licensed under Apache License 2.0.</i>

    <b>No sensitive attributes can be provided.</b>
    """

    assert (
        len(sensitive) == 0
    ), "Bias scan cannot have any sensitive attributes, as it helps identify those."

    text = """
    <div class="container mt-4">
        <h1 class="text-primary">Most </h1>
        <p>
            The following bias scores were obtained across all attributes. Protect the most
            biased attributes which you recognize as sensitive.
        </p>
    </div>
    """

    # Obtain predictions
    predictions = pd.Series(model.predict(dataset, sensitive))
    labels = dataset.labels

    raise Exception("Bias scan not implemented yet")

    return HTML(text)
