from mammoth.models import EmptyModel
from mammoth.integration import loader


@loader(namespace="maniospas", version="v005", python="3.11")
def no_model() -> EmptyModel:
    """Signifies that the analysis should focus solely on the fairness of the dataset."""

    return EmptyModel()
