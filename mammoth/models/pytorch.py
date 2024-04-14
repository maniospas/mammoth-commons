from kfp import dsl
import torch


class PYTORCH:
    integration = dsl.Model

    def __init__(self, path: str, model: torch.nn.Module):
        self.model_url = path
        self.model = model()
        self.model.load_state_dict(torch.load(self.model_url))

    def predict(self, x: torch.tensor):
        return self.model(x)
