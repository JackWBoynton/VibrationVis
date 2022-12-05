
import plotly.graph_objects as go
import numpy as np
  
# Data for three-dimensional scattered points
z = 15 * np.random.random(100)
x = np.sin(z) + 0.1 * np.random.randn(100)
y = np.cos(z) + 0.1 * np.random.randn(100)
print(x, y, z)
fig = go.Figure(data=[go.Mesh3d(
  x=x, y=y, z=z, color='green', opacity=0.20)])
  
fig.show()