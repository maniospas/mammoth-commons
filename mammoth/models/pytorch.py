from mammoth.models.predictor import Predictor


class Pytorch(Predictor):
    def __init__(self, model):
        self.model = model

    def predict(self, dataset, sensitive):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        dataloader = dataset.to_torch(sensitive)

        model.eval()
        all_predictions = []
        all_labels = []
        all_sensitive = [[] for _ in sensitive]
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(batch[1].cpu())
                for i in range(len(sensitive)):
                    all_sensitive[i].append(batch[2][i].cpu())

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        dataset.labels = {"0": 1-all_labels, "1": all_labels}
        dataset.data = {name: torch.cat(value) for name, value in zip(sensitive, all_sensitive)}
        return all_predictions
