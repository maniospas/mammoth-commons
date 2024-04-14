import numpy as np
import urllib
import onnxruntime as rt
from mammoth.models.model import Model


class ONNX(Model):
    def __init__(self, path: str):
        self.model_url = path

    def predict(self, x):
        with urllib.request.urlopen(self.model_url) as f:
            model_bytes = f.read()
        sess = rt.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        return sess.run([label_name], {input_name: x.astype(np.float64)})[0]
