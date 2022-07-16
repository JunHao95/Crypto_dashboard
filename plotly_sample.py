import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

df = px.data.iris()
all_dims = ['sepal_length', 'sepal_width',
            'petal_length', 'petal_width']

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x}
                 for x in all_dims],
        value=all_dims[:2],
        multi=True
    ),
    dcc.Graph(id="splom"),
])

@app.callback(
    Output("splom", "figure"),
    [Input("dropdown", "value")])
def update_bar_chart(dims):
    fig = px.scatter_matrix(
        df, dimensions=dims, color="species")
    return fig

app.run_server(debug=True)
##############################

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Multi output example'),
    dcc.Dropdown(id='data-dropdown', options=[
        {'label': 'Movies', 'value': 'movies'},
        {'label': 'Series', 'value': 'series'}
    ], value='movies'),
    html.Div([
        dcc.Graph(id='graph'),
        dt.DataTable(id='data-table', columns=[
            {'name': 'Title', 'id': 'title'},
            {'name': 'Score', 'id': 'score'}
        ])
    ])
], id='container')


@app.callback([
    Output('graph', 'figure'),
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('container', 'style')
], [Input('data-dropdown', 'value')])
def multi_output(value):
    if value is None:
        raise PreventUpdate

    selected = sample_data[value]
    data = selected['data']
    columns = [
        {'name': k.capitalize(), 'id': k}
        for k in data[0].keys()
    ]
    figure = go.Figure(
        data=[
            go.Bar(x=[x['score']], text=x['title'], name=x['title'])
            for x in data
        ]
    )

    return figure, data, columns, selected['style']


if __name__ == '__main__':
    app.run_server(debug=True, port=8056)


# dcc.Graph(id="TVL"), 3
        # dcc.Graph(id="Annualized"), 3
        # dcc.Graph(id="Protocol_Revenue"), 5
        # dcc.Graph(id="P_S_Ratio"),2
        # dcc.Graph(id="P_E_Ratio"),2