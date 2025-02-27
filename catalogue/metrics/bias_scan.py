import pandas as pd

from mammoth.datasets import Dataset
from mammoth.models import Predictor
from mammoth.exports import HTML
from typing import List
from mammoth.integration import metric
from aif360.sklearn.metrics import ot_distance

@metric(
    namespace="mammotheu",
    version="v0036",
    python="3.11",
    packages=("aif360", "aif360[OptimalTransport]", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
)
def bias_scan(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
) -> HTML:
    """Performs a scan for biased attributes by comparing their fraction of positive predictions
    to the dataset. It is up to the user to determine which of the biased attributes should be considered sensitive.

    <i><b>License:</b> The following description is adapted from AIF360 (<a href="https://github.com/Trusted-AI/AIF360">https://github.com/Trusted-AI/AIF360</a>), which is licensed under Apache License 2.0.</i>
    """

    text = """
    <div class="container mt-4">
        <h1 class="text-primary">Most </h1>
        <p>
            The following bias scores were obtained across all attributes. Protect the most
            biased attributes which you recognize as sensitive.
        </p>
    </div>
    """

    if len(sensitive) != 0:
        raise Exception("Bias scan analysis cannot have any sensitive attributes; its purpose is to help you identify those.")

    # Obtain predictions
    predictions = pd.Series(model.predict(dataset, sensitive))
    labels = dataset.labels

    if hasattr(labels, "columns"):
        text += """
        <div class="container mt-4">
            <table class="table table-striped table-bordered">
                <thead class="table-dark">
                    <tr>
                        <th>Attribute</th>
                        <th>Group</th>
        """
        for label_name in labels.columns:
            text += f"<th>{label_name}</th>"
        text += """
                    </tr>
                </thead>
                <tbody>
        """

        # Collect distances for merging
        results = {}
        for label_name in labels.columns:
            label = pd.Series(labels[label_name])
            for attr in sensitive:
                df = dataset.data[attr]
                dist = ot_distance(y_true=label, y_pred=predictions, prot_attr=df)
                for k, v in dist.items():
                    if (attr, k) not in results:
                        results[(attr, k)] = {}
                    results[(attr, k)][label_name] = v

        # Render merged table
        for (attr, group), distances in results.items():
            text += f"""
            <tr>
                <td>{attr}</td>
                <td>{group}</td>
            """
            for label_name in labels.columns:
                text += f"<td>{distances.get(label_name, 'N/A'):.3f}</td>"
            text += "</tr>"

        text += """
                </tbody>
            </table>
        </div>
        """
    else:
        labels = pd.Series(labels)
        text += """
        <div class="container mt-4">
            <table class="table table-striped table-bordered">
                <thead class="table-dark">
                    <tr>
                        <th>Attribute</th>
                        <th>Group</th>
                        <th>Wasserstein Distance</th>
                    </tr>
                </thead>
                <tbody>
        """
        for attr in sensitive:
            df = dataset.data[attr]
            dist = ot_distance(y_true=labels, y_pred=predictions, prot_attr=df)
            for k, v in dist.items():
                text += f"""
                    <tr>
                        <td>{attr}</td>
                        <td>{k}</td>
                        <td>{v:.3f}</td>
                    </tr>
                """
        text += """
                </tbody>
            </table>
        </div>
        """

    text += """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    """

    return HTML(text)
