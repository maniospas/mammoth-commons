from mammoth.datasets import Image
from mammoth.models.pytorch import Pytorch
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric


@metric(namespace="mammotheu", version="v003", python="3.11")
def image_bias_analysis(
    dataset: Image,
    model: Pytorch,
    sensitive: List[str],
    task: str = "",
) -> Markdown:
    """Write your metric's description here."""
    if task == "face_verification":
        raise Exception("Not implemented yet")
    elif task == "face_attribute_extraction":
        raise Exception("Not implemented yet")
    else:
        # just testing
        torch_loader = dataset.to_torch(sensitive)
        # model.load_model(model_class)
        for img, target, sensitive in torch_loader:
            predictions = model.predict(img)

    return Markdown(text="")
