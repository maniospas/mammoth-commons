import os.path

from mammoth.datasets import CSV
from mammoth.integration import loader
import fairbench as fb
import pandas as pd
from typing import List, Optional


@loader(namespace="csh", version="002", python="3.11")
def data_csv_rankings(
    path: str = "",
    numeric: list = [
        "Ranking",
        "Value",
    ],  # numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical: list = [
        "Gender",
        "Nationality",
    ],  # ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome",]
    delimiter: str = ",",
    on_bad_lines: str = "skip",
) -> CSV:
    """This is a Loader to load .csv files with information about researchers"""
    validate_input(path, numeric, categorical, delimiter, on_bad_lines)
    raw_data = pd.read_csv(path, on_bad_lines=on_bad_lines, delimiter=delimiter)

    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=["id"],              # Just a dummy right now.  We don't do supervised learning and don't "label" anything
    )
    return csv_dataset

def validate_input(path, numeric, categorical, delimiter, on_bad_lines):
    if not path.endswith(".csv"):
        raise Exception("The csv component does not yet support dataset cards.")
