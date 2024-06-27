from mammoth.models import ONNX
from mammoth.integration import loader
import urllib


@loader(
    namespace="maniospas",
    version="v004",
    python="3.11"
)
def model_onnx(path: str = None) -> ONNX:
    """This is an ONNX loader."""

    with urllib.request.urlopen(path) as f:
        model_bytes = f.read()
    return ONNX(model_bytes)
