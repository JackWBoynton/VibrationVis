

import mpi4py.MPI as MPI
import plotly.graph_objects as go
import numpy as np
import json
import pickle
import pandas as pd

SAMPLE_RATE = 2000
DATA_PATH = "data/shaky.csv"
SENSORS = {'A': (0, 0, 1.25), 'B': (0, 7.0, 1.25), 'D': (7.25, 7.0, 1.25)}

def make_structure(length_inches, width_inches, height_inches, points_per_inch):
  x, y, z = np.meshgrid(
      np.linspace(0, length_inches, int(points_per_inch[0] * length_inches)),
      np.linspace(0, width_inches, int(points_per_inch[1] * width_inches)),
      np.linspace(0, height_inches, int(points_per_inch[2] * height_inches)),
      indexing="ij",
      sparse=False,
  )
  return x.flatten(), y.flatten(), z.flatten()

def one_timestep_all_sensors(x, y, z, local_data_sel):
  sensor_plots = [go.Scatter3d(x=x,y=y,z=z, mode="markers", marker=dict(color="lightgrey",opacity=0.5), name="Structure")]
  intensity = np.zeros(len(x))
  for n, (sensor, trimmed_sensor) in enumerate(zip(SENSORS.keys(), local_data_sel)):
      sensor_x, sensor_y, sensor_z = SENSORS[sensor]
      print(local_data_sel.shape)
      trimmed_sensor_data_x, trimmed_sensor_data_y, trimmed_sensor_data_z = local_data_sel[n, 0], local_data_sel[n, 1], local_data_sel[n, 2]
      print(trimmed_sensor_data_x)
      distance = np.sqrt((x-sensor_x)**2 + (y-sensor_y)**2 + (z-sensor_z)**2)
      intensity += np.array((trimmed_sensor_data_x**2 + trimmed_sensor_data_y**2 + trimmed_sensor_data_z**2)**0.5 / distance)

  intensity = np.log(intensity)
  sensor_plots.append(go.Scatter3d(x=x[intensity > np.median(intensity)], y=y[intensity > np.median(intensity)], z=z[intensity > np.median(intensity)], mode="markers", marker=dict(color=intensity[intensity > np.median(intensity)], colorscale="Viridis", opacity=0.7)))

  # update the graph title with the time of the frame  "{} T={:.2f} s".format(sensor, i/SAMPLE_RATE)
  return go.Frame(data=sensor_plots)

def load(data_path):
  data = pd.read_csv(data_path).sort_values(by="timestamp_sent")
  return data

def build_frames() -> None:
  data = load(DATA_PATH)
  x, y, z = make_structure(7.25, 7.0, 1.25, (3, 3, 8))

  min_len = min([len(data[(data["field"] == sensor) & (data["channel"] == channel)].values) for sensor in SENSORS.keys() for channel in ["X", "Y", "Z"]])

  local_data = np.zeros((len(SENSORS.keys()), 3, min_len))
  for n, sensor in enumerate(SENSORS.keys()):
    for m, axis in enumerate(["X", "Y", "Z"]):
      local_data[n][m] = data[(data["field"] == sensor) & (data["channel"] == axis)].reading.values[:min_len]

  # split trimmed into chunks for each processor and scatter with mpi
  # local_data = np.array_split(local_data, 10, axis=2)
  local_data = MPI.COMM_WORLD.Scatterv(local_data, root=0, axis=2)

  frames = []
  for i in range(local_data.shape[2]):
      local_data_sel = local_data[:, :, i]
      frames.append(one_timestep_all_sensors(x, y, z, local_data_sel))
  # gather frames from each processor

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Gathering frames")
      frames = MPI.COMM_WORLD.gather(frames, root=0)
      frames = [frame for frames in frames for frame in frames]
      print(len(frames))

if __name__ == "__main__":
  frames = build_frames()

  # make it pickle
  with open("frames.pkl", "wb") as f:
      pickle.dump(frames, f)
