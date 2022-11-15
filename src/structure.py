"""Structure Class and Utils."""

import numpy as np
import plotly.graph_objects as go


class StructureWithSensors:
    """Class to Handle an Instance of a Structure to Simulate."""

    def __init__(
        self,
        length_inches: float,
        width_inches: float,
        height_inches: float,
        sensors: dict[str, tuple[float, float, float]],
        density: float = 1.0,
        points_per_inch: int = 25
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

    def load_data(self, data_path: str) -> None:
        pass

    def build_frames(self) -> None:
        pass

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
        fig.show()


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
my_structure.plot()
