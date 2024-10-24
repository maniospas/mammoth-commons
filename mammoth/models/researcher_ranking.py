from mammoth.models.predictor import Predictor


class ResearcherRanking(Predictor):
    def __init__(self, ranking_function):
        self.rank = ranking_function

    def predict(self, dataset, sensitive):
        if len(sensitive) != 1:
            raise Exception("Researcher ranking data cannot have ")
        return self.rank(dataset, sensitive[0])
