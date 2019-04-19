import numpy as np


def voronoi(x, y, tri, X, Y):

  """
  compute the voronoi array from an unstructured triangular grid and a structured squared grid

  triangular grid defined by:
    x, y: arrays of shape (npoin) giving node coordinates
    tri: array of shape (nelem, 3) giving node connectivity table

  structured grid defined by:
    X: array of size (nx) giving center cell coordinates in the x direction
    Y: array of size (ny) giving center cell coordinates in the y direction

  return:
    vor: array of shape (nx, ny) giving index of closest triangular grid node (-1 if outside triangular grid)
  """


  # initialize voronoi array
  vor = np.zeros((len(X), len(Y)), dtype = int) - 1

  # for each triangle
  for i in range(tri.shape[0]):

    # triangle vertex coordinates
    x0 = x[tri[i, 0]]
    x1 = x[tri[i, 1]]
    x2 = x[tri[i, 2]]
    y0 = y[tri[i, 0]]
    y1 = y[tri[i, 1]]
    y2 = y[tri[i, 2]]

    # triangle bounding box and corresponding indices on the structured grid
    xmin = min([x0, x1, x2])
    xmax = max([x0, x1, x2])
    ymin = min([y0, y1, y2])
    ymax = max([y0, y1, y2])
    try:imin = int(np.argwhere(X <= xmin)[-1])
    except:imin = 0
    try:imax = int(np.argwhere(X >= xmax)[0])
    except:imax = len(X) - 1
    try:jmin = int(np.argwhere(Y <= ymin)[-1])
    except:jmin = 0
    try:jmax = int(np.argwhere(Y >= ymax)[0])
    except:jmax = len(Y) - 1

    # local grid of the bounding box
    Xloc, Yloc = np.meshgrid(X[imin:imax + 1], Y[jmin:jmax + 1], \
                             indexing = 'ij')

    # compute barycentric coordinates
    s0 = ((y1 - y2) * (Xloc  - x2) + (x2 - x1) * (Yloc  - y2)) \
       / ((y1 - y2) * (x0    - x2) + (x2 - x1) * (y0    - y2))
    s1 = ((y2 - y0) * (Xloc  - x2) + (x0 - x2) * (Yloc  - y2)) \
       / ((y1 - y2) * (x0    - x2) + (x2 - x1) * (y0    - y2))
    s2 = 1. - s0 - s1

    # s[i,j] is True if barycentric coordinates are all positive, and the
    # corresponding structured grid cell is inside the triangle
    s = (s0 >= 0.) * (s1 >= 0.) * (s2 >= 0.)

    # distance to triangle vertices
    d = np.array([(x0 - Xloc) * (x0 - Xloc) + (y0 - Yloc) * (y0 - Yloc), \
                  (x1 - Xloc) * (x1 - Xloc) + (y1 - Yloc) * (y1 - Yloc), \
                  (x2 - Xloc) * (x2 - Xloc) + (y2 - Yloc) * (y2 - Yloc)])

    # tmp[i,j] is the number of the closest vertex...
    tmp = tri[i, np.argmin(d, 0)]

    # ... but outside the triangle, tmp[i,j] is the value of the voronoi array
    vor_loc = vor[imin:imax + 1, jmin:jmax + 1]
    tmp[s == False] = vor_loc[s == False]

    # update voronoi array for structured grid cells inside the triangle
    vor[imin:imax + 1, jmin:jmax + 1] = tmp

  return vor