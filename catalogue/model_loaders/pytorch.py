import os
from mammoth.models.pytorch import Pytorch
from mammoth.integration import loader
from mammoth.externals import safeexec
import torch


@loader(namespace="gsarridis", version="v003", python="3.11")
def model_torch(model: str = "", state: str = "") -> Pytorch:
    """This is an PYTORCH model loader."""

    model = safeexec(model, out="model", whitelist=["torchvision", "torch"])

    model.load_state_dict(torch.load(state))

    return Pytorch(model)
