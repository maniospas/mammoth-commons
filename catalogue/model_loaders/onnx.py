from mammoth.models import ONNX
from mammoth.integration import loader
import urllib


@loader(namespace="maniospas", version="v005", python="3.11", packages=("onnxruntime",))
def model_onnx(path: str = "") -> ONNX:
    """Loads an inference model stored in ONNx format.

    Args:
        path: A url pointing to the loaded file.
    """

    with urllib.request.urlopen(path) as f:
        model_bytes = f.read()
    return ONNX(model_bytes)
