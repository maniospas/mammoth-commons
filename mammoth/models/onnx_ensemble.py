import numpy as np
from mammoth.models.model import Model
import re


class ONNXEnsemble(Model):
    def __init__(self, models, params):
        self.models = models
        self.params = params

    def _extract_number(self, filename):
        match = re.search(r"_(\d+)\.onnx$", filename)
        return int(match.group(1)) if match else float("inf")

    def predict(self, X):
        # n_classes = self.params['n_classes']
        classes = self.params["classes"][:, np.newaxis]

        pred = sum(
            (estimator.predict(X) == classes).T * w
            for estimator, w in zip(
                self.models[: self.params["theta"]],
                self.params["alphas"][: self.params["theta"]],
            )
        )
        pred /= self.params["alphas"][: self.params["theta"]].sum()
        pred[:, 0] *= -1
        preds = classes.take(pred.sum(axis=1) > 0, axis=0)
        return np.squeeze(preds, axis=1)
