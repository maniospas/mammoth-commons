import os.path

from mammoth.datasets.image_dataset import ImageDataset
from mammoth.integration import loader
from typing import List
import torchvision.transforms as transforms


@loader(namespace="gsarridis", version="v003", python="3.11")
def image_dataset(
    path: str,
    root_dir: str = "./",
    target="",
    data_transform: str = "",
    batch_size=4,
    shuffle=False,
) -> ImageDataset:
    """
    Creates a Dataset for loading image data from a CSV file.

    Args:
        path (str): The path to the CSV file containing information about the dataset.
        root_dir (str): The root directory where the actual image files are stored.
    Returns:
        Dataset
    """

    # TODO: load data transforms (transforms.Compose) from the data_transform path eg './data/data_transforms.py'

    dataset = ImageDataset(
        path=path,
        root_dir=root_dir,
        target=target,
        data_transform=data_transform,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataset
