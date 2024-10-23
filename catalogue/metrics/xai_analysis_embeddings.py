from mammoth.datasets import ImagePairs
from mammoth.models.pytorch import Pytorch
from mammoth.exports import HTML
from typing import List
from mammoth.integration import metric

# install facex lib using: pip install facextool
from facex.component import run_embeddings_mammoth


@metric(
    namespace="gsarridis",
    version="v003",
    python="3.11",
    packages=("torch", "torchvision", "timm", "facextool"),
)
def facex_embeddings(
    dataset: ImagePairs,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = None,
    target_layer: str = None,
) -> HTML:
    """Write your metric's description here.

    Args:
        target_class: The integer identifier of the target class.
        target_layer: The layer to be explained.
    """

    target_class = int(target_class)
    html = run_embeddings_mammoth(
        dataset, sensitive[0], target_class, model.model, target_layer
    )

    return HTML(html)
