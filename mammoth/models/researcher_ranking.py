from mammoth.models.model import Model


class ResearcherRanking(Model):
    def __init__(self, ranking_function):
        self.ranking_function = ranking_function

    def rank(self, dataset, ranking_variable):
        """
        A column called Ranking_{ranking_variable} is added to the dataset
        """
        return self.ranking_function(dataset, ranking_variable)