from mammoth.datasets import Image
from mammoth.models import EmptyModel
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric, Options
from cvbiasmitigation.suggest import analysis


@metric(
    namespace="mammotheu",
    version="v003",
    python="3.11",
    packages=("torch", "torchvision", "cvbiasmitigation"),
)
def image_bias_analysis(
    dataset: Image,
    model: EmptyModel,
    sensitive: List[str],
    task: Options("face verification", "image classification") = None,
) -> Markdown:
    """
    Performs analysis of image bias, and recommends mitigation strategies.

    Args:
        task: The type of predictive task. It should be either face verification or image classification.
    """

    assert task in [
        "face verification",
        "image classification",
    ], "The provided task should be either face verification or image classification"
    md = analysis(dataset.path, task, dataset.target, sensitive)
    return Markdown(md)
