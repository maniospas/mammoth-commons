import os

import mammoth
from autocsv import data_csv
from onnx import model_onnx
from metric import new_metric


with mammoth.testing.Env(data_csv, model_onnx, new_metric) as env:
    numeric = ["age", "duration", "campaign", "pdays", "previous"]
    categorical = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "poutcome",
    ]
    sensitive = ["marital"]
    dataset_uri = (
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv"
    )
    dataset = env.data_csv(
        dataset_uri, categorical=categorical, numeric=numeric, labels="y", delimiter=";"
    )

    model_path = "file://localhost//" + os.path.abspath("model.onnx")
    model = env.model_onnx(model_path)

    analysis_outcome = env.new_metric(dataset, model, sensitive)
    print(analysis_outcome.text)
