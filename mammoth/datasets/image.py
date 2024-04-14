from typing import List
from mammoth.datasets import Dataset


class Image(Dataset):
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

    def to_torch(self, sensitive: List[str]):
        # dynamic dependencies here to not force a torch dependency on commons from components that don't need it
        from torch.utils.data import DataLoader
        from mammoth.datasets.backend.torch import PytorchImageDataset

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
