import matplotlib.pyplot as plt
import numpy as np

from projection import projection_p1

# to do: move gmsh2telemac module in ogtools
from gmsh2telemac import load_msh


# data
X, Y = np.meshgrid(np.linspace(-.5, .5, 101), np.linspace(-.5, .5, 101))
F = np.exp(-(X ** 2 + Y ** 2))

# figure data
plt.figure()
plt.pcolormesh(X, Y, F, vmin = .65, vmax = 1.)
plt.colorbar()
plt.axis('scaled')
plt.savefig('data.png', dpi = 300)
plt.close()


# grid
x, y, tri, bnd, physical = load_msh('circle.msh')

# reshape data
X = X.reshape(-1)
Y = Y.reshape(-1)
F = F.reshape(-1)

# projection
f = projection_p1(x, y, tri, X, Y, F)

# figure projection
plt.figure()
plt.tripcolor(x, y, tri, f, vmin = .65, vmax = 1.)
plt.colorbar()
plt.axis('scaled')
plt.savefig('projection.png', dpi = 300)
plt.close()