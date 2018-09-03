import matplotlib.pyplot as plt
import numpy as np

from gmsh2telemac import load_msh
from curvature import compute_mini_cloud, export_mini_cloud, \
                      import_mini_cloud, gaussian_curvature_p1


# grid
x, y, tri, bnd, physical = load_msh('../data/circle_100.msh')

# data
f = np.exp(-(x * x + y * y))

# compute list of mini-clouds
cloud = compute_mini_cloud(x, y, tri)

# export list of mini-clouds
export_mini_cloud(cloud, 'cloud.bin')

# import list of mini-clouds
cloud = import_mini_cloud('cloud.bin')

# numerical curvature
r = gaussian_curvature_p1(x, y, f, tri, cloud)

# analytical curvature
fx = -2. * x * f
fy = -2. * y * f
fxx = -2. * f + 4. * x * x * f
fxy = 4. * x * y * f
fyy = -2. * f + 4. * y * y * f
k = (fxx * fyy - fxy * fxy) / (1. + fx * fx + fy * fy) ** 2.

# figure numerical curvature
plt.figure()
plt.tripcolor(x, y, tri, r, vmin = 1., vmax = 3.5)
plt.colorbar()
plt.axis('scaled')
plt.savefig('curvature_numerical.png', dpi = 300)
plt.close()

# figure analytical curvature
plt.figure()
plt.tripcolor(x, y, tri, k, vmin = 1., vmax = 3.5)
plt.colorbar()
plt.axis('scaled')
plt.savefig('curvature_analytical.png', dpi = 300)
plt.close()
