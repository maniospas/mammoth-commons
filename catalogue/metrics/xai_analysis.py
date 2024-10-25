from mammoth.datasets import Image
from mammoth.models.pytorch import Pytorch
from mammoth.exports import HTML
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
def facex_regions(
    dataset: Image,
    model: Pytorch,
    sensitive: List[str],
    target_class: int = None,
    target_layer: str = None,
) -> HTML:
    """Analyses 19 facial regions and accessories to provide explanations.

    Args:
        target_class: The integer identifier of the target class.
        target_layer: The layer to be explained.
    """

    target_class = int(target_class)
    html = run_mammoth(dataset, sensitive[0], target_class, model.model, target_layer)
    html = (
        """
    <h1>About</h1>
    <p>FaceX analysed 19 facial regions and accessories to provide explanations. In the two illustrations below,
    left are face regions and right are hat and glasses. Blue are the least important regions and red the most
    important ones that are taken into account. Based on the outputs, try to the question of “where does a model
    focus on?”. We also show high-impact patches to help understand “what visual features trigger its focus?”.</p>
    """
        + html
    )
    return HTML(html)
