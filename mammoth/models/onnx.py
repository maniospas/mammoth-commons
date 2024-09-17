import numpy as np
from mammoth.models.model import Model


class ONNX(Model):
    def __init__(self, model_bytes, np_type=np.float64):
        self.model_bytes = model_bytes
        self.np_type = np_type

    def predict(self, x):
        import onnxruntime as rt
        sess = rt.InferenceSession(self.model_bytes, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        return sess.run([label_name], {input_name: x.astype(self.np_type)})[0]
