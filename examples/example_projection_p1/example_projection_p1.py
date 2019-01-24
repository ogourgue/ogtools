import matplotlib.pyplot as plt
import numpy as np

from gmsh2telemac import load_msh
from projection import projection_p1


# high-resolution data
X, Y = np.meshgrid(np.linspace(-.5, .5, 201), np.linspace(-.5, .5, 201))
X = X.reshape(-1)
Y = Y.reshape(-1)
F = np.exp(-(X * X + Y * Y))

# low-resolution grid
x, y, tri, bnd, physical = load_msh('../data/circle_100.msh')

# analytical
f = np.exp(-(x * x + y * y))

# figure analytical
plt.figure()
plt.tripcolor(x, y, tri, f, vmin = .65, vmax = 1)
plt.colorbar()
plt.axis('scaled')
plt.savefig('data_analytical.png', dpi = 300)
plt.close()

# projection
f = projection_p1(x, y, tri, X, Y, F)

# figure projection
plt.figure()
plt.tripcolor(x, y, tri, f, vmin = .65, vmax = 1)
plt.colorbar()
plt.axis('scaled')
plt.savefig('data_projection.png', dpi = 300)
plt.close()