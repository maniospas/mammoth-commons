from mammoth.datasets import Image
from mammoth.models.pytorch import Pytorch
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric
from cvbiasmitigation.suggest import analysis


@metric(
    namespace="mammotheu",
    version="v003",
    python="3.11",
    packages=("torch", "torchvision", "cvbiasmitigation")
)
def image_bias_analysis(
    dataset: Image,
    model: Pytorch,
    sensitive: List[str],
    task: str = "",
) -> Markdown:

    md = analysis(dataset.path, task, dataset.target, sensitive)
    return Markdown(md)
