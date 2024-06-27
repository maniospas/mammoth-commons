from mammoth.datasets import Image
from mammoth.models.pytorch import Pytorch
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric

# install facex lib using: pip install facextool
from facex.component import run_mammoth


@metric(
    namespace="gsarridis",
    version="v003",
    python="3.11",
    packages=("torch", "torchvision", "timm", "facextool"),
)
def facex(
    dataset: Image,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = None,
    target_layer: str = None,
) -> Markdown:
    """Write your metric's description here."""

    html = run_mammoth(dataset, sensitive[0], target_class, model, target_layer)

    return html
