"""Main Entrypoint and Structure."""

from plotly.subplots import make_subplots

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc, html, no_update

from structure import StructureWithSensors

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# inches
X_LEN = 7.25
Y_LEN = 7.0
Z_LEN = 1.25

sensors = {
    "A": (0, 0, Z_LEN),
    "B": (0, Y_LEN ,Z_LEN),
    "C": (X_LEN, Y_LEN, Z_LEN),
}

sws = StructureWithSensors(X_LEN, Y_LEN, Z_LEN, sensors)
sws.load_data("data/test1.csv")

app.layout = html.Div([
  dbc.Row([
    dbc.Col([
      dbc.Card(
        children=[
          dbc.CardHeader(html.H2("3D Animated Structure")),
          dbc.CardBody(children=[
              dcc.Loading(children=[dcc.Graph(id="main-3d-graph", figure=sws.plot()),])
            ],
          )
        ],
        style={"width": "100%", "height": "auto", "margin": "auto"},
      ),
      dbc.Row(
        [
          dbc.Card(
            children=[
              dbc.CardHeader(html.H2("XYZ Acceleration")),
              dbc.CardBody(children=[
                  dcc.Loading(children=[
                      dcc.Graph(id="xyz-graph", figure=sws.xyz_plot()),
                    ]
                  )
                ],
              )
            ],
            style={"width": "100%", "height": "auto", "margin": "auto"},
          ),
        ]
      )
    ],
    style={"width": "auto", "height": "auto", "margin": "auto"}),
  ])
])


@app.callback(
  [
    Output("xyz-graph", "figure"),
  ],
  [
    Input("play-button", "n_clicks"),
  ]
)
def graphs(n_clicks):
  cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

