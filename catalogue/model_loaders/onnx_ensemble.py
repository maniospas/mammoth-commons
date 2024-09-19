from mammoth.models import ONNX, ONNXEnsemble
from mammoth.integration import loader
import re
import numpy as np
import zipfile


@loader(namespace="arjunroy", version="v003", python="3.11", packages=("onnxruntime",))
def model_onnx_ensemble(path: str = "") -> ONNXEnsemble:
    """This is an ONNX_Ensemble loader."""

    models = []
    model_names = []
    params = None

    def myk(name):
        return int(re.findall(r"[+-]?\d+", name)[0])

    # Read the zip file
    with zipfile.ZipFile(path) as myzip:
        # Extract and load the weights file
        for file_name in myzip.namelist():
            if file_name.endswith(".npy"):
                with myzip.open(file_name) as param_file:
                    params = np.load(param_file, allow_pickle=True)
            elif file_name.endswith(".onnx"):
                model_names.append(file_name)

        model_names.sort(key=myk)

        for file_name in model_names:
            with myzip.open(file_name) as model_file:
                model_bytes = model_file.read()
                models.append(ONNX(model_bytes, np.float32))

    return ONNXEnsemble(models, params.item())
