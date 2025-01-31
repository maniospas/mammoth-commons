echo "Building components"

pip install --upgrade -r requirements\[test\].txt
pip install -e .

kfp component build . --component-filepattern catalogue/dataset_loaders/auto_csv.py
kfp component build . --component-filepattern catalogue/dataset_loaders/custom_csv.py
kfp component build . --component-filepattern catalogue/dataset_loaders/data_csv_rankings.py
kfp component build . --component-filepattern catalogue/dataset_loaders/graph_from_csv.py
kfp component build . --component-filepattern catalogue/dataset_loaders/graph.py
kfp component build . --component-filepattern catalogue/dataset_loaders/image_pairs.py
kfp component build . --component-filepattern catalogue/dataset_loaders/images.py
kfp component build . --component-filepattern catalogue/dataset_loaders/uci_csv.py

kfp component build . --component-filepattern catalogue/model_loaders/compute_rankings.py
kfp component build . --component-filepattern catalogue/model_loaders/compute_researcher_ranking.py
kfp component build . --component-filepattern catalogue/model_loaders/fair_node_ranking.py
kfp component build . --component-filepattern catalogue/model_loaders/no_model.py
kfp component build . --component-filepattern catalogue/model_loaders/onnx_ensemble.py
kfp component build . --component-filepattern catalogue/model_loaders/onnx.py
kfp component build . --component-filepattern catalogue/model_loaders/pytorch.py

kfp component build . --component-filepattern catalogue/metrics/image_bias_analysis.py
kfp component build . --component-filepattern catalogue/metrics/interactive_report.py
kfp component build . --component-filepattern catalogue/metrics/interactive_sklearn_report.py
kfp component build . --component-filepattern catalogue/metrics/ma_graph_connection.py
kfp component build . --component-filepattern catalogue/metrics/model_card.py
kfp component build . --component-filepattern catalogue/metrics/Multi_objective_report.py
kfp component build . --component-filepattern catalogue/metrics/ranking_fairness.py
kfp component build . --component-filepattern catalogue/metrics/xai_analysis_embeddings.py
kfp component build . --component-filepattern catalogue/metrics/xai_analysis.py

mkdir yamls
mkdir yamls/data
mkdir yamls/meta

cp catalogue/dataset_loaders/component_metadata/* yamls/meta/
cp catalogue/model_loaders/component_metadata/* yamls/meta/
cp catalogue/metrics/component_metadata/* yamls/meta/
cp component_metadata/* yamls/data/


echo "Completed building components"

