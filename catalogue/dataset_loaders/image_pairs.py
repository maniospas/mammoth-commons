from mammoth.datasets import ImagePairs
from mammoth.integration import loader
from mammoth.externals import safeexec


@loader(
    namespace="gsarridis",
    version="v001",
    python="3.11",
    packages=("torch", "torchvision"),
)
def data_image_pairs(
    path: str = "",
    root_dir: str = "./",
    target: str = "",
    data_transform: str = "",
    batch_size: int = 4,
    shuffle: bool = False,
    img1_path_format: str = "{root}/{col}/{id}.png",
    img2_path_format: str = "{root}/{col}/{id}.png",
) -> ImagePairs:
    """
    Loads image pairs declared in a CSV file.
    The expected format is to have the first image's identifier in the first column,
    and the second image's identifier in the second column, Sensitive attributes
    can be selected from the rest of the columns. The images identifiers read from the columns
    are transformed to loading paths by string specifications that can contain the
    symbols: {root} to refer to the root directory, {col} to refer to the column name, and {id}
    to refer to the column entry.

    Args:
        path: The path to the CSV file containing information about the dataset.
        root_dir: The root directory where the actual image files are stored.
        target Indicates the predictive attribute in the dataset.
        data_transform: A path or implementation of a torchvision data transform.
        batch_size: The number of image pairs in each batch. Default is 4.
        shuffle: Whether to shuffle the dataset. Default is False.
        img1_path_format: The first image path format. Default is "{root}/{col}/{id}.png".
        img2_path_format: The second image path format. Default is "{root}/{col}/{id}.png".
    """

    data_transform = safeexec(
        data_transform, out="transform", whitelist=["torchvision"]
    )

    dataset = ImagePairs(
        path=path,
        root_dir=root_dir,
        target=target,
        data_transform=data_transform,
        batch_size=batch_size,
        shuffle=shuffle,
        img1_path_format=img1_path_format,
        img2_path_format=img2_path_format,
    )

    return dataset
