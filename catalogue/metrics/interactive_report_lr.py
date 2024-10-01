import numpy as np

from mammoth.datasets import CSV
from mammoth.models import EmptyModel
from mammoth.exports import HTML
from typing import Dict, List
from mammoth.integration import metric
import fairbench as fb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


@metric(namespace="maniospas", version="v001", python="3.11", packages=("fairbench","scikit-learn",))
def interactive_logistic_regression_report(
    dataset: CSV,
    model: EmptyModel,
    sensitive: List[str],
    intersectional: bool = True,
    pairwise_comparison: bool = True,
) -> HTML:
    """Creates an interactive report using the FairBench library, after running an internal training-test split
    on a basic sklearn model. The report creates traceable evaluations that you can shift through to find sources
    of unfairness on a common task.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute.
        pairwise_comparison: Whether to compare groups pairwise. Otherwise, each group is compared to the whole population.
    """
    for attr in sensitive:
        if attr not in dataset.categorical:
            raise Exception(
                "Fairness analysis not supported on non-categorical attributes"
            )


    X = dataset.to_features()
    y = dataset.labels
    assert y.shape[1] <= 2, "Cannot create a logicistic regression interactive report for non-binary predictions"
    y = y[y.columns[-1]]

    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(X, y, np.arange(0, y.shape[0], dtype=np.int64), test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:,1]

    # declare sensitive attributes
    sensitive = fb.Fork(
        {attr + " ": (fb.categories @ dataset.data[attr][idx_test]) for attr in sensitive}
    )

    # change behavior based on arguments
    if intersectional:
        sensitive = sensitive.intersectional()
    report_type = fb.multireport if pairwise_comparison else fb.unireport

    report = report_type(predictions=predictions, labels=y_test.to_numpy(), scores=scores, sensitive=sensitive)
    return HTML(fb.interactive_html(report, show=False, name="Binary logistic regression"))
