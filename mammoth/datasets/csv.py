import numpy as np
from mammoth.datasets.dataset import Dataset


def _features(df, numeric, categorical):
    import pandas as pd

    dfs = [df[col] for col in numeric] + [
        pd.get_dummies(df[col]) for col in categorical
    ]
    return pd.concat(dfs, axis=1).values


class CSV(Dataset):
    def __init__(self, data, numeric, categorical, labels):
        import pandas as pd

        self.data = data
        self.numeric = numeric
        self.categorical = categorical
        self.labels = pd.get_dummies(data[labels])
        self.cols = numeric + categorical

    def to_features(self, sensitive):
        for attr in sensitive:
            if attr not in self.categorical:
                raise Exception(
                    "Fairness analysis on CSV datasets is not supported for non-categorical sensitive attributes."
                )
        return _features(self.data, self.numeric, self.categorical).astype(np.float64)
