import os
from mammoth.models.pytorch import PYTORCH
from mammoth.integration import loader
import torch


@loader(namespace="gsarridis", version="v003", python="3.11")
def model_torch(path: str, model_dict: str = "", model_class="") -> PYTORCH:
    """This is an PYTORCH loader."""
    # TODO get the model class from os.path.join(path,model_class) .py file
    return PYTORCH(os.path.join(path, model_dict), model_class)
