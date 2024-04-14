import numpy as np
from fairbench.bench.loader import features
from fairbench import categories
from mammoth.datasets.dataset import Dataset


class CSV(Dataset):
    def __init__(self, data, numeric, categorical, labels):
        self.data = data
        self.numeric = numeric
        self.categorical = categorical
        self.labels = categories @ data[labels]
        self.cols = numeric + categorical

    def to_features(self):
        return features(self.data, self.numeric, self.categorical).astype(np.float64)
