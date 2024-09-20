from mammoth.models.model import Model


class EmptyModel(Model):
    def __init__(self):
        pass

    def predict(self, x):
        raise Exception("Cannot perform predictions on an empty model")
