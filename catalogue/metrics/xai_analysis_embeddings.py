<<<<<<< HEAD
from mammoth.datasets import ImagePairs
=======
from mammoth.datasets import Image
>>>>>>> 0d50f192b02c4dac26015ffcb2afc77f4b78fb51
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
<<<<<<< HEAD
    dataset: ImagePairs,
=======
    dataset: Image,
>>>>>>> 0d50f192b02c4dac26015ffcb2afc77f4b78fb51
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
