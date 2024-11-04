from mammoth.datasets import CSV
from mammoth.integration import loader, Options
import os


@loader(
    namespace="arjunroy",
    version="v002",
    python="3.11",
    packages=("pandas", "ucimlrepo"),
)
def data_uci_csv(
    dataset_name: Options("Credit", "Bank") = None,
    target=None,
) -> CSV:
    """Loads a CSV file that contains numeric, categorical, and predictive data columns.
    Automatic detection methods for the delimiter and column types are applied.
    The last categorical column is considered the dataset label. To load the file using
    different options (e.g., a subset of columns, different label column) use the
    custom csv loader instead.

    Args:
        dataset_name: The local file path or a web URL of the file.
        target: The name of the predictive column.
    """
    name = dataset_name.lower()
    if name == "credit":
        d_id = 350
    elif name == "bank":
        d_id = 222
    else:
        raise Exception("Unexpected dataset name: " + name)

    import pandas as pd
    from ucimlrepo import fetch_ucirepo

    if d_id is None:
        all_raw_data = fetch_ucirepo(name)
    else:
        all_raw_data = fetch_ucirepo(id=d_id)
    # print(raw_data.metadata.additional_info.summary)
    # print(raw_data.metadata.additional_info.variable_info)
    raw_data = all_raw_data.data.features
    numeric = [
        col
        for col in raw_data
        if pd.api.types.is_any_real_numeric_dtype(raw_data[col])
        and len(set(raw_data[col])) > 10
    ]
    numeric_set = set(numeric)
    categorical = [col for col in raw_data if col not in numeric_set]
    if len(categorical) < 1:
        raise Exception("At least two categorical columns are required.")
    label = all_raw_data.data.targets[target]

    csv_dataset = CSV(
        raw_data,
        numeric=numeric,
        categorical=categorical,
        labels=label,
    )
    csv_dataset.description = {
        "summary": all_raw_data.metadata.additional_info.summary,
        "variable_info": all_raw_data.metadata.additional_info.variable_info,
    }
    return csv_dataset
