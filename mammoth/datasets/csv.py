import numpy as np
from fairbench.bench.loader import features
from typing import Dict
import pandas as pd


class CSV:
    def __init__(self, data, numeric, categorical, labels):
        self.data = data
        self.numeric = numeric
        self.categorical = categorical
        self.labels = labels
        self.cols = numeric + categorical

    def to_features(self):
        return features(self.data, self.numeric, self.categorical).astype(np.float64)


def load_csv(
    path: str,
    parameters: Dict[str, any] = None,
):
    if parameters is None:
        parameters = dict()
    on_bad_lines = parameters.get("on_bad_lines", "skip")
    delimiter = parameters.get("delimiter", "','")
    raw_data = pd.read_csv(path, on_bad_lines=on_bad_lines, delimiter=delimiter)
    csv = CSV(
        raw_data,
        numeric=["age", "duration", "campaign", "pdays", "previous"],
        categorical=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "poutcome",
        ],
        labels=(raw_data["y"] != "no").values,
    )
    return csv
