import numpy as np
from mammoth.models.predictor import Predictor
import re


class ONNXEnsemble(Predictor):
    def __init__(self, models, params):
        self.models = models
        self.params = params

    def _extract_number(self, filename):
        match = re.search(r"_(\d+)\.onnx$", filename)
        return int(match.group(1)) if match else float("inf")

    def predict(self, dataset, sensitive):
        """assert (
            sensitive is None or len(sensitive) == 0
        ), "ONNXEnsemble can only be called with no declared sensitive attributes" """
        X = dataset.to_pred(sensitive)
        # n_classes = self.params['n_classes']
        classes = self.params["classes"][:, np.newaxis]

        pred = sum(
            (estimator.predict(X, []) == classes).T * w
            for estimator, w in zip(
                self.models[: self.params["theta"]],
                self.params["alphas"][: self.params["theta"]],
            )
        )
        pred /= self.params["alphas"][: self.params["theta"]].sum()
        pred[:, 0] *= -1
        preds = classes.take(pred.sum(axis=1) > 0, axis=0)
        return np.squeeze(preds, axis=1)
