from mammoth.datasets import Image
from mammoth.integration import loader
from mammoth.externals import safeexec


@loader(namespace="gsarridis", version="v003", python="3.11")
def data_images(
    path: str = "",
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
        target (str): Indicates the predictive attribute in the dataset.
        data_transform (str): A path or implementation of a torchvision data transform.
    Returns:
        Image dataset.
    """

    data_transform = safeexec(data_transform,
                              out="transform",
                              whitelist=["torchvision"])

    dataset = Image(
        path=path,
        root_dir=root_dir,
        target=target,
        data_transform=data_transform,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataset
