from mammoth.models.model import Model


class Pytorch(Model):
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model(x)
