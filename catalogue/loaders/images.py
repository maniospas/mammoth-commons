from mammoth.datasets import Image
from mammoth.integration import loader


@loader(namespace="gsarridis", version="v003", python="3.11")
def data_images(
    path: str,
    root_dir: str = "./",
    target: str = "",
    data_transform: str = "",
    batch_size: int = 4,
    shuffle: bool = False,
) -> Image:
    """
    Creates a Dataset for loading image data from a CSV file.

    Args:
        path (str): The path to the CSV file containing information about the dataset.
        root_dir (str): The root directory where the actual image files are stored.
    Returns:
        Dataset
    """

    # TODO: load data transforms (transforms.Compose) from the data_transform path eg './data/data_transforms.py'
    dataset = Image(
        path=path,
        root_dir=root_dir,
        target=target,
        data_transform=data_transform,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataset
