"""Structure Class and Utils."""

import numpy as np
import pywt
import scipy.signal
import scipy
from scipy import signal
from scipy.fft import fftshift
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

SAMPLE_RATE = 2000

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

    def xyz_plot(self):
        fig = make_subplots(rows=len(self.sensors), cols=len(self.data.channel.unique()), shared_xaxes=True)
        for m, sensor in enumerate(self.sensors):
            for n, channel in enumerate(("X", "Y", "Z")):
                sel = self.data[(self.data["field"] == sensor) & (self.data["channel"] == channel)]
                fig.add_trace(
                    go.Scatter(x=pd.to_datetime(sel.timestamp_sent, unit="s"), y=sel.reading.values, name="{} {}".format(sensor, channel)),
                    row=m+1, col=n+1
                )
                fig.update_yaxes(title_text=sensor, row=m+1, col=n+1)
                fig.update_xaxes(title_text=channel, row=m+1, col=n+1)
        fig.update_xaxes(title_text=" Y <br></br>  Time", row=len(self.sensors), col=2)
        fig.update_yaxes(title_text="Acceleration (g)<br></br>   B", row=2, col=1)
        return fig

    def fft_xyz_plot(self):
        fig = make_subplots(rows=len(self.sensors), cols=len(self.data.channel.unique()), shared_xaxes=True)
        for m, sensor in enumerate(self.sensors):
            for n, channel in enumerate(("X", "Y", "Z")):
                sel = self.data[(self.data["field"] == sensor) & (self.data["channel"] == channel)]
                # compute the fft of the signal
                fft = np.fft.fft(sel.reading.values)
                # get the frequencies
                freq = np.fft.fftfreq(sel.shape[0])
                freq_axis = np.linspace(0, SAMPLE_RATE, len(np.abs(fft)))

                fig.add_trace(
                    go.Scatter(x=freq_axis, y=np.abs(fft), name="{} {}".format(sensor, channel)),
                    row=m+1, col=n+1
                )
                fig.update_yaxes(title_text=sensor, row=m+1, col=n+1)
                fig.update_xaxes(title_text=channel, row=m+1, col=n+1)

        fig.update_xaxes(title_text="   Y    <br></br>FFT frequency", row=len(self.sensors), col=2)
        fig.update_yaxes(title_text="Intensity <br></br>   B", row=2, col=1)
        return fig

    def psd_xyz_plot(self):
        fig = make_subplots(rows=len(self.sensors), cols=len(self.data.channel.unique()), shared_xaxes=True)
        for m, sensor in enumerate(self.sensors):
            for n, channel in enumerate(("X", "Y", "Z")):
                sel = self.data[(self.data["field"] == sensor) & (self.data["channel"] == channel)]

                (f, S)= scipy.signal.welch(sel.reading.values, SAMPLE_RATE, nperseg=4*1024)

                fig.add_trace(
                    go.Scatter(x=f, y=S, name="{} {}".format(sensor, channel)),
                    row=m+1, col=n+1
                )
                # yaxis is log scale
                fig.update_yaxes(type="log", row=m+1, col=n+1)

                fig.update_yaxes(title_text=sensor, row=m+1, col=n+1)
                fig.update_xaxes(title_text=channel, row=m+1, col=n+1)

        fig.update_xaxes(title_text="  Y    <br></br>frequency [Hz]", row=len(self.sensors), col=2)
        fig.update_yaxes(title_text="PSD [V**2/Hz] <br></br>   B", row=2, col=1)
        return fig

    def spectogram_xyz_plot(self):
        fig = make_subplots(rows=len(self.sensors), cols=len(self.data.channel.unique()), shared_xaxes=True)
        for m, sensor in enumerate(self.sensors):
            for n, channel in enumerate(("X", "Y", "Z")):
                sel = self.data[(self.data["field"] == sensor) & (self.data["channel"] == channel)]

                # compute the vibration spectrogram of the signal
                freq_axis, time_axis, S = scipy.signal.spectrogram(sel.reading.values, SAMPLE_RATE, nperseg=256, noverlap=255)

                fig.add_trace(
                    go.Heatmap(z=S, x=time_axis, y=freq_axis, name="{} {}".format(sensor, channel), colorscale="Jet", showscale=True if m == len(self.sensors)-1 and n == len(self.data.channel.unique()) else False),
                    row=m+1, col=n+1
                )
                # yaxis is log scale
                # fig.update_yaxes(type="log", row=m+1, col=n+1)

                fig.update_yaxes(title_text=sensor, row=m+1, col=n+1)
                fig.update_xaxes(title_text=channel, row=m+1, col=n+1)


        fig.update_xaxes(title_text="  Y    <br></br>Time", row=len(self.sensors), col=2)
        fig.update_yaxes(title_text="Frequency [Hz] <br></br>    B", row=2, col=1)
        return fig

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
        # set bounds on zyz
        # fig.update_layout(
        #     scene = dict(
        #         xaxis = dict(nticks=4, range=[0, 10],),
        #         yaxis = dict(nticks=4, range=[0, 10],),
        #         zaxis = dict(nticks=4, range=[0, 10],),
        #         ),
        #     width=700,
        # )
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