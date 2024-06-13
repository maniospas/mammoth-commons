import os.path

from mammoth.datasets import CSV
from mammoth.integration import loader
import fairbench as fb
import pandas as pd
from typing import List, Optional


@loader(namespace="mammotheu", version="v003", python="3.11")
def data_csv_rankings(
    path: str,
    numeric: list =["Ranking", "Value"],  # numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical: list = ["Gender", "Nationality"],  # ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome",]
    labels: Optional[str] = None,
    delimiter: str = ",",
    on_bad_lines: str = "skip",
) -> CSV:
    """This is a CSV loader."""
    if not path.endswith(".csv"):
#         card = path + os.path.pathsep + "card.yaml"
#         path = path + os.path.pathsep + "data.csv"
        raise Exception("The csv component does not yet support dataset cards.")
    raw_data = pd.read_csv(
        path, on_bad_lines=on_bad_lines, delimiter=delimiter)
    #csv_dataset = CSV(raw_data,
     #   numeric=numeric,
      #  categorical=categorical,
       # labels=labels,
    #)
    return raw_data
