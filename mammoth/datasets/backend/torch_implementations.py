from PIL import Image as PILImage
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from typing import List
import os


class PytorchImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target: str,
        sensitive: List[str],
        data_transform: transforms.Compose,
    ):
        """
        PyTorch dataset for image data.

        Args:
            csv_path (str): The path to the CSV file containing information about the dataset.
            root_dir (str): The root directory where the actual image files are stored.
            target (str): The name of the column in the CSV file containing the target variable.
            sensitive (List[str]): A list of strings representing columns in the CSV file containing sensitive information.
            transforms (transforms.Compose): A composition of image transformations.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target = target
        self.sensitive = sensitive
        self.data_transform = data_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = self.root_dir + "/" + img_name
        image = PILImage.open(img_path).convert("RGB")
        target = self.data.iloc[idx][self.target]
        protected = [self.data.iloc[idx][attr] for attr in self.sensitive]
        if self.data_transform is not None:
            image = self.data_transform(image)
        return image, target, protected


class PytorchImagePairsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target: str,
        sensitive: List[str],
        data_transform: transforms.Compose,
        img1_path_format: str = "{root}/{col}/{id}.png",
        img2_path_format: str = "{root}/{col}/{id}.png",
    ):
        """
        PyTorch dataset for image data.

        Args:
            csv_path (str): The path to the CSV file containing information about the dataset.
            root_dir (str): The root directory where the actual image files are stored.
            target (str): The name of the column in the CSV file containing the target variable.
            sensitive (List[str]): A list of strings representing columns in the CSV file containing sensitive information.
            transforms (transforms.Compose): A composition of image transformations.
        """
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.target = target
        self.sensitive = sensitive
        self.data_transform = (data_transform,)
        self.img1_path = img1_path_format
        self.img2_path = img2_path_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_name = self.data.iloc[idx, 0]  # ref
        img2_name = self.data.iloc[idx, 1]  # motion
        first_column_name = self.data.columns[0]
        second_column_name = self.data.columns[1]

        id1_image_path = (
            self.img1_path.replace("{root}", self.root_dir)
            .replace("{col}", first_column_name)
            .replace("{img}", img1_name)
        )
        id2_image_path = (
            self.img2_path.replace("{root}", self.root_dir)
            .replace("{col}", second_column_name)
            .replace("{img}", img2_name)
        )

        image1 = PILImage.open(id1_image_path).convert("RGB")
        image2 = PILImage.open(id2_image_path).convert("RGB")

        target = self.data.iloc[idx][self.target]
        protected = [self.data.iloc[idx][attr] for attr in self.sensitive]
        if self.data_transform is not None:
            image1 = self.data_transform(image1)
            image2 = self.data_transform(image2)
        return image1, image2, target, protected
