import torch
from mammoth.models.model import Model


class Pytorch(Model):
    def __init__(self, path: str, model: torch.nn.Module):
        self.model_url = path
        self.model = model
        self.model.load_state_dict(torch.load(self.model_url))

    def predict(self, x: torch.tensor):
        return self.model(x)
