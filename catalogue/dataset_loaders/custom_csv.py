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
def data_custom_csv(
    path: str = "",
    numeric: Optional[
        List[str]
    ] = None,  # numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical: Optional[
        List[str]
    ] = None,  # ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    label: Optional[str] = None,
    delimiter: str = ",",
    skip_invalid_lines: bool = True,
) -> CSV:
    """Loads a CSV file that contains numeric, categorical, and predictive data columns.

    Args:
        path: The local file path or a web URL of the file.
        numeric: A list of column names that hold numeric data.
        categorical: A list of column names that hold categorical data.
        label: The name of the categorical column that holds predictive label for each data sample.
        delimiter: Which character to split loaded csv rows with.
        skip_invalid_lines: Whether to skip invalid lines being read instead of creating an error.
    """
    if not path.endswith(".csv"):
        raise Exception("A file or url with .csv extension is needed.")
    raw_data = fb.bench.loader.read_csv(
        path,
        on_bad_lines="skip" if skip_invalid_lines else "error",
        delimiter=delimiter,
    )
    if raw_data.shape[1] == 1:
        raise Exception("Only one column was found. This often indicates that the wrong delimiter was specified.")
    if label not in raw_data:
        raise Exception(f"The dataset has no column name `{label}` to set as a label."
                        f"\nAvailable columns are: {', '.join(raw_data.columns)}")
    for col in categorical:
        if col not in raw_data:
            raise Exception(f"The dataset has no column name `{col}` to add to categorical attributes."
                            f"\nAvailable column are: {', '.join(raw_data.columns)}")
    for col in numeric:
        if col not in raw_data:
            raise Exception(f"The dataset has no column name `{col}` to add to numerical attributes."
                            f"\nAvailable columns are: {', '.join(raw_data.columns)}")
    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )
    return csv_dataset
