from mammoth.datasets import CSV
from typing import Dict, List
from mammoth.integration import loader
import pandas as pd


@loader(version="v003", python="3.11")
def data_csv_loader(
    path: str,
    delimiter: str = ",",
    on_bad_lines: str = "skip",
) -> CSV:
    """This is a CSV loader.
    """
    raw_data = pd.read_csv(path, on_bad_lines=on_bad_lines, delimiter=delimiter)
    csv_dataset = CSV(
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
    return csv_dataset

