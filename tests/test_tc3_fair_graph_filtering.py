import mammoth
from catalogue.dataset_loaders.graph import data_graph
from catalogue.model_loaders.fair_node_ranking import model_fair_node_ranking
from catalogue.metrics.model_card import model_card


def test_fair_graph_filtering():
    with mammoth.testing.Env(data_graph, model_fair_node_ranking, model_card) as env:
        graph = env.data_graph(path="citeseer")
        model = env.model_fair_node_ranking(diffusion=0.9)

        sensitive = ["0"]
        analysis_outcome = env.model_card(graph, model, sensitive)
        print(analysis_outcome.text)

test_fair_graph_filtering()
