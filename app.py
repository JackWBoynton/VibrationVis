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
  dcc.Dropdown(os.listdir('data'), random.choice(os.listdir('data')), id='data-dropdown'),
  dcc.Interval(id='slider-interval', interval=1000, n_intervals=0, disabled=True),
  dbc.Button("click me", id="click"),
  dbc.Row([
    dbc.Col([
      dbc.Card(
        children=[
          dbc.CardHeader(html.H2("3D Animated Structure")),
          dbc.CardBody(children=[
              dcc.Loading(children=[dcc.Graph(id="main-3d-graph"),])
            ],
          )
        ],
        style={"width": "100%", "height": "60%", "margin": "auto"},
      ),
      dbc.Row(
        [
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
            style={"width": "100%", "height": "auto", "margin": "auto"},
          ),
        ]
      ),
      dbc.Row(
        [
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
            style={"width": "100%", "height": "auto", "margin": "auto"},
          ),
        ]
      ),
      dbc.Row(
        [
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
            style={"width": "100%", "height": "auto", "margin": "auto"},
          ),
        ]
      ),
      dbc.Row(
        [
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
            style={"width": "100%", "height": "auto", "margin": "auto"},
          ),
        ]
      )
    ],
    style={"width": "auto", "height": "100%", "margin": "auto"}),
  ])
])


@app.callback(
  [
    Output("slider-interval", "disabled"),
    Output('main-3d-graph', 'figure'),
    Output('xyz-graph-accel', 'figure'),
    Output('xyz-graph-fft', 'figure'),
    Output('xyz-graph-spectral', 'figure'),
    Output('xyz-graph-spectrogram', 'figure')
  ],
  [
    Input('data-dropdown', 'value'),
    Input('slider-interval', 'n_intervals'),
    Input("xyz-graph-accel", "figure"),
    Input('main-3d-graph', 'figure'),
    Input('xyz-graph-accel', 'figure'),
    Input('xyz-graph-fft', 'figure'),
    Input('xyz-graph-spectral', 'figure'),
    Input('xyz-graph-spectrogram', 'figure')
  ],
)
def debug(value, pas, val, main_3d_graph, xyz_graph_accel, xyz_graph_fft, xyz_graph_spectral, xyz_graph_spectrogram):
  cbcontext = [p['prop_id'] for p in dash.callback_context.triggered][0]
  rets = [main_3d_graph, xyz_graph_accel, xyz_graph_fft, xyz_graph_spectral, xyz_graph_spectrogram]
  print(cbcontext)
  if cbcontext == "data-dropdown.value":
    print("data-dropdown.value")
    sws = StructureWithSensors(X_LEN, Y_LEN, Z_LEN, sensors, {"A": "red", "B": "blue", "D": "green"})
    sws.load_data(f"data/{value}")
    return False, sws.plot(), sws.xyz_plot(), sws.fft_xyz_plot(), sws.psd_xyz_plot(), sws.spectogram_xyz_plot()

  elif cbcontext == "slider-interval.n_intervals":
    if val is not None and "layout" in val and "sliders" in val["layout"]:
      # get the current value of the slider
      print(val["layout"]["sliders"][0]["active"])
      slider = int(val["layout"]["sliders"][0]["active"])
      if slider == 0:
        return dash.no_update


      # add a vertical line to the graph at the time that corresponds to the slider selection
      val["layout"]["shapes"] = [
        {
          "type": "line",
          "xref": "x1",
          "yref": "y1",
          "x0": val["data"][0]["x"][slider],
          "y0": 0,
          "x1": val["data"][0]["x"][slider],
          "y1": 1,
          "line": {
            "color": "rgb(55, 128, 191)",
            "width": 3,
          },
        }
      ]
      rets[1] = val
      # prevent the slider from updating the graph
      return False, *rets

  return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True, threaded=False)
