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
        density: float = 0.15,
        points_per_inch: int = 30
    ) -> None:
        self.length_inches = length_inches
        self.width_inches = width_inches
        self.height_inches = height_inches
        self.sensors = sensors
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
        import matplotlib.pyplot as plt
        by_sensors = [self.data[self.data["field"] == sensor] for sensor in self.sensors]
        min_len = min([len(sensor) for sensor in by_sensors])
        trimmed = [sensor[:min_len] for sensor in by_sensors]

        frames = []
        # make me faster
        for i in tqdm(range(0, 9203, 100)):
            sensor_plots = [go.Scatter3d(x=self.x, y=self.y, z=self.z, mode="markers", marker=dict(color="lightgrey",opacity=0.5), name="Structure")]
            for sensor, trimmed_sensor in zip(self.sensors.keys(), trimmed):
                sensor_x, sensor_y, sensor_z = self.sensors[sensor]
                propagataion_x = np.abs(trimmed_sensor[trimmed_sensor["channel"] == "X"]["reading"].values[i]) / self.density
                propagataion_y = np.abs(trimmed_sensor[trimmed_sensor["channel"] == "Y"]["reading"].values[i]) / self.density
                propagataion_z = np.abs(trimmed_sensor[trimmed_sensor["channel"] == "Z"]["reading"].values[i] + 1) / self.density

                # color points based on the magnitude of the propagation and the distance from the sensor
                color = np.sqrt(np.square(self.x - sensor_x) + np.square(self.y - sensor_y) + np.square(self.z - sensor_z))
                color = color / np.max(color)
                color = color * 255
                color = np.minimum(color, 255 / propagataion_z)
                color = np.minimum(color, 255 / propagataion_y)
                color = np.minimum(color, 255 / propagataion_x)
                color = np.maximum(color, 0)
                color = np.round(color)

                # build the 3d scatter plot
                sensor_plots.append(go.Scatter3d(
                    x=self.x[np.logical_and(np.logical_and(np.logical_and(np.logical_and(self.x >= sensor_x - propagataion_x, self.x <= sensor_x + propagataion_x), self.y >= sensor_y - propagataion_y), self.y <= sensor_y + propagataion_y), self.z >= sensor_z - propagataion_z, self.z <= sensor_z + propagataion_z)],
                    y=self.y[np.logical_and(np.logical_and(np.logical_and(np.logical_and(self.x >= sensor_x - propagataion_x, self.x <= sensor_x + propagataion_x), self.y >= sensor_y - propagataion_y), self.y <= sensor_y + propagataion_y), self.z >= sensor_z - propagataion_z, self.z <= sensor_z + propagataion_z)],
                    z=self.z[np.logical_and(np.logical_and(np.logical_and(np.logical_and(self.x >= sensor_x - propagataion_x, self.x <= sensor_x + propagataion_x), self.y >= sensor_y - propagataion_y), self.y <= sensor_y + propagataion_y), self.z >= sensor_z - propagataion_z, self.z <= sensor_z + propagataion_z)],
                    mode="markers",
                    marker=dict(
                        color=color[np.logical_and(np.logical_and(np.logical_and(np.logical_and(self.x >= sensor_x - propagataion_x, self.x <= sensor_x + propagataion_x), self.y >= sensor_y - propagataion_y), self.y <= sensor_y + propagataion_y), self.z >= sensor_z - propagataion_z, self.z <= sensor_z + propagataion_z)],
                        opacity=1,
                        colorscale="Viridis"
                    ),
                    name=f"Sensor {sensor}"
                ))
                # get frame

            frames.append(go.Frame(data=sensor_plots))
            # append frame
        return frames

    def plot(self) -> None:
        fig = go.Figure(
            data=[
                go.Scatter3d(x=self.x, y=self.y, z=self.z, mode="markers", name="Structure"),
                *[go.Scatter3d(x=[self.sensors[sensor][0]], y=[self.sensors[sensor][1]], z=[self.sensors[sensor][2]], mode="markers", name=f"Sensor {sensor}") for sensor in self.sensors]
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
        # fig.show()
        return fig

    def show(self):
        self.fig.show()


if __name__ == "__main__":
    # inches
    X_LEN = 7.25
    Y_LEN = 7.0
    Z_LEN = 1.25

    sensors = {
        "A": (0, 0, Z_LEN),
        "B": (0, Y_LEN ,Z_LEN),
        "D": (X_LEN, Y_LEN, Z_LEN),
    }
    my_structure = StructureWithSensors(X_LEN, Y_LEN, Z_LEN, sensors)
    my_structure.load_data("data/test1.csv")
    # my_structure.build_frames()
    my_structure.plot()
    # my_structure.show()