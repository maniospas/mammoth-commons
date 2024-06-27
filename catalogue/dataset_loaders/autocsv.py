import os.path

from mammoth.datasets import CSV
from mammoth.integration import loader
import fairbench as fb
from typing import List, Optional


@loader(
    namespace="maniospas",
    version="v004",
    python="3.11"
)
def data_csv(
    path: str = "",
    numeric: Optional[List[str]] = None,  # numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical: Optional[List[str]] = None,  # ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome",]
    labels: Optional[str] = None,
    delimiter: str = ",",
    on_bad_lines: str = "skip",
) -> CSV:
    """This is a CSV loader."""
    if not path.endswith(".csv"):
        card = path + os.path.pathsep + "card.yaml"
        path = path + os.path.pathsep + "data.csv"
        raise Exception("The csv component does not yet support dataset cards.")
    raw_data = fb.bench.loader.read_csv(
        path, on_bad_lines=on_bad_lines, delimiter=delimiter
    )
    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=labels,
    )
    return csv_dataset
