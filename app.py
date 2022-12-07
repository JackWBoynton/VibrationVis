"""Main Entrypoint and Structure."""

import os
import random

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

server = app.server

# inches
X_LEN = 7.25
Y_LEN = 7.0
Z_LEN = 1.25

sensors = {
    "A": (0, 0, Z_LEN),
    "B": (0, Y_LEN ,Z_LEN),
    "D": (X_LEN, Y_LEN, Z_LEN),
}

app.layout = html.Div([
  dbc.Row([
          dbc.Card(children=[
                      dbc.CardHeader(html.H2("Load Data")),
                      dcc.Dropdown(os.listdir('data'), random.choice(os.listdir('data')), id='data-dropdown')
                     ],
                   )
           ], className = 'align-self-center'
          ),

  dbc.Card(
    children=[
      dbc.CardHeader(html.H2("3D Animated Structure")),
      dbc.CardBody(children=[
          dcc.Loading(children=[dcc.Graph(id="main-3d-graph"),])
        ], style={"center": "auto"}
      )
    ], style={"width": "auto", "height": "80rem", "justify-content": "center", "align-items": "center"},
  ),
  dbc.Card(
    children=[
      dbc.CardHeader(html.H2("XYZ Acceleration")),
      dbc.CardBody(children=[
          dcc.Loading(children=[
              dcc.Graph(id="xyz-graph-accel"),
            ]
          )
        ],
      )
    ],
  ),
  dbc.Card(
    children=[
      dbc.CardHeader(html.H2("XYZ Acceleration -- FFT")),
      dbc.CardBody(children=[
          dcc.Loading(children=[
              dcc.Graph(id="xyz-graph-fft"),
            ]
          )
        ],
      )
    ],
  ),
  dbc.Card(
    children=[
      dbc.CardHeader(html.H2("XYZ Acceleration -- Power Spectral Density")),
      dbc.CardBody(children=[
          dcc.Loading(children=[
              dcc.Graph(id="xyz-graph-spectral"),
            ]
          )
        ],
      )
    ],
  ),

  dbc.Card(
    children=[
      dbc.CardHeader(html.H2("XYZ Spectrogram")),
      dbc.CardBody(children=[
          dcc.Loading(children=[
              dcc.Graph(id="xyz-graph-spectrogram"),
            ]
          )
        ],
      )
    ],
  )
])


@app.callback(
    [Output('main-3d-graph', 'figure'), Output('xyz-graph-accel', 'figure'), Output('xyz-graph-fft', 'figure'), Output('xyz-graph-spectral', 'figure'), Output('xyz-graph-spectrogram', 'figure')],
    Input('data-dropdown', 'value')
)
def update_output(value):
    sws = StructureWithSensors(X_LEN, Y_LEN, Z_LEN, sensors, {"A": "red", "B": "blue", "D": "green"})
    sws.load_data(f"data/{value}")
    return [sws.plot(), sws.xyz_plot(), sws.fft_xyz_plot(), sws.psd_xyz_plot(), sws.spectogram_xyz_plot()]


if __name__ == '__main__':
    app.run_server(debug=True)

