from mammoth.datasets import Image
from mammoth.models.pytorch import Pytorch
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric


@metric(namespace="mammotheu", version="v003", python="3.11", packages=("torch", "torchvision"))
def facex(dataset: Image, model: Pytorch, sensitive: List[str]) -> Markdown:
    """Write your metric's description here."""

    # just testing
    torch_loader = dataset.to_torch(sensitive)
    for img, target, sensitive in torch_loader:
        predictions = model.predict(img)

    return Markdown(text="")
