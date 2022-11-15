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
        points_per_inch: int = 10
    ) -> None:
        self.length_inches = length_inches
        self.width_inches = width_inches
        self.height_inches = height_inches
        self.density = density
        self.points_per_inch = points_per_inch

        self.x, self.y, self.z = np.meshgrid(
            np.linspace(0, self.length_inches, self.points_per_inch),
            np.linspace(0, self.width_inches, self.points_per_inch),
            np.linspace(0, self.height_inches, self.points_per_inch),
            indexing="ij"
        )

    def load_data(self, data_path: str) -> None:
        pass

    def build_frames(self) -> None:
        pass

    def plot(self) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=self.x, y=self.y, z=self.z, opacity=1))
        fig.show()


# inches
X_LEN = 7.25
Y_LEN = 7.0
Z_LEN = 1.25

sensors = {
    "A": (0, 0, Z_LEN),
    "B": (0, Y_LEN ,Z_LEN),
    "D": (Z_LEN, Y_LEN, Z_LEN),
}
my_structure = StructureWithSensors(X_LEN, Y_LEN, Z_LEN, sensors)
my_structure.plot()
