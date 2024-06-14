from mammoth.models import ONNXEnsemble
from mammoth.integration import loader


@loader(namespace="maniospas", version="v003", python="3.11")
def model_onnx_ensemble(path: str) -> ONNXEnsemble:
    """This is an ONNX_Ensemble loader."""
    return ONNXEnsemble(path)