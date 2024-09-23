from mammoth import testing
from catalogue.model_loaders.onnx_ensemble import model_onnx_ensemble


class obs_stats:
    def __init__(self, pareto_all, pseudoW, pareto_fairness, fair_pseudoW,prot_att):
        self.PF= pareto_all
        self.pseudo=pseudoW
        self.FPF=pareto_fairness
        self.Fpseudo=fair_pseudoW
        self.prot_attr=prot_att


def see_pareto(self):
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    def plot3d(x=[1, 2, 3, 4, 5],
               y=[2, 3, 1, 4, 5],
               z=[3, 1, 2, 5, 4],
               weights=[1, 2, 3, 4, 5],
               criteria='all',
               axis_names=['X', 'Y', 'Z']):
        # Create the figure
        hidden_data = [str(np.around(weights[i], 2)) for i in range(len(weights))]
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        # Add 3D scatter plot
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'),
                                   name='Points', text=hidden_data, customdata=hidden_data), row=1, col=1)

        # Define callback function for selection event
        def update_point(trace, points, selector):
            selected_points = [(trace.x[i], trace.y[i], trace.z[i]) for i in points.point_inds]
            selected_hidden_data = [trace.customdata[i] for i in points.point_inds]
            print("Selected Points:", selected_points, selected_hidden_data)

        # Assign callback function to plot
        fig.data[0].on_selection(update_point)

        # Update layout
        fig.update_layout(title=criteria + '-objectives Pareto Plot - Select Points to Return Vector',
                          scene=dict(
                              xaxis_title='X:' + axis_names[0],
                              yaxis_title='Y:' + axis_names[1],
                              zaxis_title='Z:' + axis_names[2]
                          ),
                          width=1000,  # Set the width of the plot
                          height=700)

        # Show plot
        fig.show()

    plot3d(x=self.PF[:, 0], y=self.PF[:, 1], z=self.PF[:, 2], weights=self.pseudo,
           axis_names=['Acc.', 'Balanc. Acc', 'MMM-fair'])
    # self.update_theta(criteria='fairness')
    if len(self.prot_attr) > 2:
        plot3d(x=self.FPF[:, 0], y=self.FPF[:, 1], z=self.FPF[:, 2], weights=self.Fpseudo,
               criteria='Fairness', axis_names=self.prot_attr)
    elif len(self.prot_attr) == 2:
        plot3d(x=self.FPF[:, 0], y=self.FPF[:, 1], z=np.zeros_like(self.FPF[:, 1]), weights=self.Fpseudo,
               criteria='Fairness', axis_names=self.prot_attr + '')

    else:
        print('Not a Multi-fair')



def test_multiattribute_bias_mitigation():
    with testing.Env(model_onnx_ensemble) as env:
        model_path = "./data/mfppb.zip"
        model = env.model_onnx_ensemble(model_path)
        print(model)

        model.update_theta(criteria='all')
        pf = model.PF
        wgts = model.pseudo

        model.update_theta(criteria='fairness')
        fpf = model.PF
        fwgts = model.pseudo


        vis_stats = obs_stats(pf, wgts, fpf, fwgts, ['Sex', 'Marital Status', 'Age'])
        see_pareto(vis_stats)


test_multiattribute_bias_mitigation()
