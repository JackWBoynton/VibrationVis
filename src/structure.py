"""Structure Class and Utils."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

class StructureWithSensors:
    """Class to Handle an Instance of a Structure to Simulate."""

    def __init__(
        self,
        length_inches: float,
        width_inches: float,
        height_inches: float,
        sensors: dict[str, tuple[float, float, float]],
        sensor_colors: dict[str, str],
        density: float = 0.75,
        points_per_inch: int = 25
    ) -> None:
        self.length_inches = length_inches
        self.width_inches = width_inches
        self.height_inches = height_inches
        self.sensors = sensors
        self.sensor_colors = sensor_colors
        self.density = density
        self.points_per_inch = points_per_inch

        x, y, z = np.meshgrid(
            np.linspace(0, self.length_inches, self.points_per_inch),
            np.linspace(0, self.width_inches, self.points_per_inch),
            np.linspace(0, self.height_inches, self.points_per_inch),
            indexing="ij",
            sparse=False,
        )
        self.x = x.flatten()
        self.y = y.flatten()
        self.z = z.flatten()

        self.fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "xy"}, {"type": "surface"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        )

    def load_data(self, data_path: str) -> None:
        self.data = pd.read_csv(data_path).sort_values(by="timestamp_sent")



    def build_frames(self) -> None:
        by_sensors = [self.data[self.data["field"] == sensor] for sensor in self.sensors]
        min_len = min([len(sensor) for sensor in by_sensors])
        trimmed = [sensor[:min_len] for sensor in by_sensors]

        frames = []

        for i in tqdm(range(min_len)[:500]):
            sensor_plots = [go.Scatter3d(x=self.x, y=self.y, z=self.z, mode="markers", name="Structure")]
            for sensor, trimmed_sensor in zip(self.sensors.keys(), trimmed):
                sensor_x, sensor_y, sensor_z = self.sensors[sensor]
                propagation_x = np.abs(trimmed_sensor[trimmed_sensor["channel"] == "X"]["reading"].values[i]) / self.density
                propagation_y = np.abs(trimmed_sensor[trimmed_sensor["channel"] == "Y"]["reading"].values[i]) / self.density
                propagation_z = np.abs(trimmed_sensor[trimmed_sensor["channel"] == "Z"]["reading"].values[i] + 1) / self.density

                # get indicies of points within the propagation radius
                x_indicies = np.where(np.abs(self.x - sensor_x) < propagation_x)[0]
                y_indicies = np.where(np.abs(self.y - sensor_y) < propagation_y)[0]
                z_indicies = np.where(np.abs(self.z - sensor_z) < propagation_z)[0]

                sensor_plots.append(go.Scatter3d(x=self.x[x_indicies], y=self.y[y_indicies], z=self.z[z_indicies], mode="markers", marker=dict(color=self.sensor_colors[sensor])))

            frames.append(go.Frame(data=sensor_plots))

        return frames

    def plot(self) -> None:
        fig = go.Figure(
            data=[
                go.Scatter3d(x=self.x, y=self.y, z=self.z, mode="markers", name="Structure"),
                *[go.Scatter3d(x=[self.sensors[sensor][0]], y=[self.sensors[sensor][1]], z=[self.sensors[sensor][2]], mode="markers", name=f"Sensor {sensor}", marker=dict(color=self.sensor_colors[sensor])) for sensor in self.sensors]
            ],
            layout=go.Layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None]
                            )
                        ]
                    )
                ]
            ),
            frames=self.build_frames()
        )
        fig.show()

    def show(self):
        self.fig.show()

# inches
X_LEN = 7.25
Y_LEN = 7.0
Z_LEN = 1.25

sensors = {
    "A": (0, 0, Z_LEN),
    "B": (0, Y_LEN ,Z_LEN),
    "D": (X_LEN, Y_LEN, Z_LEN),
}
sensor_colors = {
    "A": "green",
    "B": "purple",
    "D": "red",
}

my_structure = StructureWithSensors(X_LEN, Y_LEN, Z_LEN, sensors, sensor_colors)
my_structure.load_data("data/test1.csv")
# my_structure.build_frames()
my_structure.plot()
# my_structure.show()