from mammoth.models.model import Model


class NodeRanking(Model):
    def __init__(self, ranker):
        import pygrank as pg

        assert isinstance(ranker, pg.NodeRanking)
        self.ranker = ranker

    def predict(self, x):
        import networkx as nx
        import pygrank as pg

        assert isinstance(x, nx.Graph) or isinstance(x, pg.Graph)
        return self.ranker(x)

    def predict_fair(self, x, sensitive):
        assert len(sensitive) == 1, "fair node ranking algorithms can only account for one sensitive attribute"
        import networkx as nx
        import pygrank as pg

        assert isinstance(x, nx.Graph) or isinstance(x, pg.Graph)
        return self.ranker(x, sensitive=sensitive[0]).np
