from mammoth.datasets import CSV
from mammoth.models import ONNX
from mammoth.exports import Markdown
from typing import Dict, List
from mammoth.integration import metric


@metric(namespace="mammotheu", version="v003", python="3.11")
def new_metric(
    dataset: CSV,
    model: ONNX,
    sensitive: List[str],
    branch: str = "all",
) -> Markdown:
    """Write your metric's description here.
    """
    return Markdown("This is a test")

