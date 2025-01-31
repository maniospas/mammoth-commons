import os

from mammoth import testing
from catalogue.dataset_loaders.custom_csv import data_custom_csv
from catalogue.model_loaders.onnx import model_onnx
from catalogue.metrics.interactive_sklearn_report import interactive_sklearn_report


def test_bias_exploration():
    with testing.Env(data_custom_csv, model_onnx, interactive_sklearn_report) as env:
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
        sensitive = ["marital", "age"]
        dataset_uri = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv"
        dataset = env.data_custom_csv(
            dataset_uri,
            categorical=categorical,
            numeric=numeric,
            label="y",
            delimiter=";",
        )

        model_path = "file://localhost//" + os.path.abspath("./data/model.onnx")
        model = env.model_onnx(model_path)

        html_result = env.interactive_sklearn_report(
            dataset, model, sensitive, predictor="Logistic regression"
        )
        html_result.show()


if __name__ == "__main__":
    test_bias_exploration()
