from mammoth.datasets import CSV
from mammoth.models.onnx_ensemble import ONNXEnsemble
from mammoth.exports import HTML
from typing import Dict, List
from mammoth.integration import metric, Options
import fairbench as fb
import numpy as np
import plotly


def create_3d_plot(x=None,y=None,z=None):
    # Generate sample 3D plot
    go=plotly.graph_objects
    if x==None or y==None or z==None:
        x = [1, 2, 3, 4, 5]
        y = [10, 11, 12, 13, 14]
        z = [5, 6, 7, 8, 9]
    hidden_data=[str(i) for i in range(len(x))]
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',text=hidden_data, customdata=hidden_data )])
    axis_names=['Acc.','Balanc. Acc', 'MMM-fair']
    fig.update_layout(title='objectives Pareto Plot',
                         scene=dict(
                                    xaxis_title='X:'+axis_names[0],
                                    yaxis_title='Y:'+ axis_names[1],
                                    zaxis_title='Z:'+ axis_names[2]
                                ),
                                width=1000,  # Set the width of the plot
                                height=700 )
    return fig.to_html(full_html=False)  # Returns HTML as a string


@fb.core.Transform
def categories(iterable):
    # print(iterable)
    is_numeric = True
    values = list()
    for value in iterable:
        try:
            values.append(float(value))
        except Exception:
            is_numeric = False
            break
    if is_numeric:
        values = np.array(values)
        mx = values.max()
        mn = values.min()
        if mx == mn:
            raise Exception(
                "Numerical sensitive attribute has the same value everywhere"
            )
        values = (values - mn) / (mx - mn)
        return {f"fuzzy min ({mn:.3f})": 1 - values, f"fuzzy max ({mx:.3f})": values}
    return fb.categories @ iterable


@metric(namespace="arjunroyihrpa", version="v005", python="3.11", packages=("fairbench","plotly"))
def Multi_objective_report(
    dataset: CSV,
    model: ONNXEnsemble,
    sensitive: List[str],
    intersectional: bool = False,
    compare_groups: Options("Pairwise", "To the total population") = None,
) -> HTML:
    """Creates an interactive report using the FairBench library. The report creates traceable evaluations that
    you can shift through to find actual sources of unfairness.

    Args:
        intersectional: Whether to consider all non-empty group intersections during analysis. This does nothing if there is only one sensitive attribute.
        compare_groups: Whether to compare groups pairwise, or each group to the whole population.
    """

    # obtain predictions
    if hasattr(model,'pareto') and model.pareto is not None:
        thetas = model.pareto
    else:
        thetas = np.arange(2,len(model.models))
    O_1,O_2,O_3 = [],[],[]
    labs = list(dataset.labels.values())[-1]
    labs = labs.to_numpy() if hasattr(labs, "to_numpy") else np.array(labs)
    for i in thetas:
        predictions = model.predict(dataset, sensitive,theta=i)
        O_1.append(1-float(fb.accuracy(predictions=predictions, labels=labs)))
        O_2.append(1-(float(fb.tpr(predictions=predictions, labels=labs))+ float(fb.tnr(predictions=predictions, labels=labs)))/2)
        mm_fair = []
        for attr in sensitive:
            prots = fb.Fork(fb.categories @ dataset.data[attr])
            groups = list(prots.branches().keys())#[g for g in prots]#
            dfnr = fb.dfnr(predictions=predictions, labels=labs, sensitive=prots)
            dfpr = fb.dfpr(predictions=predictions, labels=labs, sensitive=prots)
            mm_fair.append(max([max([float(dfnr[g]) for g in groups]), max([float(dfpr[g]) for g in groups])]))
            #mm_fair.append(max([fb.areduce(dfpr, fb.max), fb.areduce(dfnr, fb.max)]))
        O_3.append(max(mm_fair))
        #print(O_1[-1], O_2[-1], O_3[-1])
        #obs.append([O_1,O_2,O_3

    plot_html = create_3d_plot(x=O_1,y=O_2,z=O_3)

    # Create an instance of your existing HTML class with the plot content
    #html_content = HTML(body=plot_html)
    html = (
                """
            <h1>About</h1>
            <p> The report highlights three primary objectives: accuracy loss, balanced accuracy loss, and discrimination (MMM-fairness) loss. Each point plotted within the 3D space represents a Pareto-optimal solution, which achieves an optimal balance between these objectives where no single objective can improve without worsening another. \n Users can hover over any solution point to display the corresponding loss values for each objective. Additionally, each point includes a theta value, indicating up to which sequence in the ONNX ensemble the particular solution is achieved. This allows users to observe performance changes throughout different stages of the ensemble, helping them better understand the trade-offs involved in each model configuration.</p>
            """
                + plot_html 
            )
    return HTML(html)#fb.interactive_html(report, show=False, name="Classes"))

