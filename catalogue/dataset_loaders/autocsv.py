import os.path

from mammoth.datasets import CSV
from mammoth.integration import loader
import fairbench as fb
from typing import List, Optional


@loader(
    namespace="maniospas",
    version="v009",
    python="3.11",
    packages=(
        "fairbench",
        "pandas",
    ),
)
def data_csv(
    path: str = "",
    numeric: Optional[
        List[str]
    ] = None,  # numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical: Optional[
        List[str]
    ] = None,  # ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    labels: Optional[str] = None,
    delimiter: str = ",",
    skip_invalid_lines: bool = True,
) -> CSV:
    """Loads a CSV file that contains numeric, categorical, and predictive data columns.

    Args:
        path: The local file path or a web URL of the file.
        numeric: A list of column names that hold numeric data.
        categorical: A list of column names that hold categorical data.
        labels: A categorical column that holds predictive labels.
        delimiter: Which character to split loaded csv rows with.
        skip_invalid_lines: Whether to skip invalid lines being read instead of creating an error.
    """
    if not path.endswith(".csv"):
        card = path + os.path.pathsep + "card.yaml"
        path = path + os.path.pathsep + "data.csv"
        raise Exception("The csv component does not yet support dataset cards.")
    raw_data = fb.bench.loader.read_csv(
        path,
        on_bad_lines="skip" if skip_invalid_lines else "error",
        delimiter=delimiter,
    )
    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=labels,
    )
    return csv_dataset
