from mammoth.models import ONNX
from mammoth.integration import loader


@loader(namespace="maniospas", version="v003", python="3.11")
def model_onnx(path: str = None) -> ONNX:
    """This is an ONNX loader."""
    return ONNX(path)
