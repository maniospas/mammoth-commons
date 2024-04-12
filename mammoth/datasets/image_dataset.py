from typing import List
from kfp import dsl
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import torch


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
        image = Image.open(img_path).convert("RGB")

        target = self.data.iloc[idx][self.target]

        protected = [self.data.iloc[idx][attr] for attr in self.sensitive]

        if self.data_transform is not None:
            image = self.data_transform(image)

        return image, target, protected


class ImageDataset:
    integration = dsl.Dataset

    def __init__(self, path, root_dir, target, data_transform, batch_size, shuffle):
        """
        Args:
            path (string): Path to the CSV file with annotations (should involve the columns path|attribute1|...|attributeN).
            root_dir (string): Root image dataset directory.

        """
        self.path = path
        self.root_dir = root_dir
        self.target = target
        self.data_transform = data_transform
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self, sensitive):
        torch_dataset = PytorchImageDataset(
            csv_path=self.path,
            root_dir=self.root_dir,
            target=self.target,
            sensitive=sensitive,
            data_transform=self.data_transform,
        )
        return DataLoader(
            dataset=torch_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
