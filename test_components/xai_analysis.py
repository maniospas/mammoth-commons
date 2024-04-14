from mammoth.datasets.image_dataset import ImageDataset
from mammoth.models.pytorch import PYTORCH
from mammoth.exports import Markdown
from typing import List
from mammoth.integration import metric

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import torch


@metric(namespace="mammotheu", version="v003", python="3.11")
def facex(dataset: ImageDataset, model: PYTORCH, sensitive: List[str]) -> Markdown:
    """Write your metric's description here."""

    # just testing
    torch_loader = dataset.get_dataloader(sensitive)
    for img, target, sensitive in torch_loader:
        predictions = model.predict(img)

    return Markdown(text="")
