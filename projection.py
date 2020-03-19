""" Projection

This module allows to calculate projections of discrete fields

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl



################################################################################
# projection p1 ################################################################
################################################################################

def projection_p1(x, y, tri, X, Y, F):

  """ Project the point cloud data on a P1 finite element field

  Required parameters:
  x, y (NumPy arrays of shape (n)): triangle grid node coordinates
  tri (NumPy array of shape (m, 3)): triangle connectivity table
  X, Y (NumPy arrays of shape (N)): point cloud coordinates
  F (NumPy array of shape (N) or (N, M)): field values to project at points (axis 0) and different time steps (axis 1)

  Returns:
  NumPy array of shape (n) or (n, M): projected data

  """

  # number of triangles
  ntri = len(tri)

  # number of grid nodes
  nnode = len(x)

  # number of data points to project
  ndata = len(X)

  # extent F to two dimensions if needed
  if F.ndim == 1:
    F = np.reshape(F, (F.shape[0], 1))

  # second dimension of F
  nt = F.shape[1]

  # initialization (lil matrix for sparse matrix fast building)
  a = sp.lil_matrix((ndata, nnode), dtype = float)

  # for each triangle
  for i in range(ntri):

    # triangle node coordinates
    x0 = x[tri[i, 0]]
    x1 = x[tri[i, 1]]
    x2 = x[tri[i, 2]]
    y0 = y[tri[i, 0]]
    y1 = y[tri[i, 1]]
    y2 = y[tri[i, 2]]

    # only keep data within the triangle bounding box
    xmin = np.min([x0, x1, x2])
    xmax = np.max([x0, x1, x2])
    ymin = np.min([y0, y1, y2])
    ymax = np.max([y0, y1, y2])
    ind = list(np.where((X >= xmin) * (X <= xmax) * \
                        (Y >= ymin) * (Y <= ymax))[0])
    Xi = X[ind]
    Yi = Y[ind]
    Fi = F[ind, :]

    # barycentric coordinates of remaining data points
    s1 = ((y1 - y2) * (Xi - x2) + (x2 - x1) * (Yi - y2)) / \
         ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
    s2 = ((y2 - y0) * (Xi - x2) + (x0 - x2) * (Yi - y2)) / \
         ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
    s3 = 1. - s1 - s2

    # only keep data within the triangle
    ind_ = list(np.where((s1 > -1e-12) * (s2 > -1e-12) * (s3 > -1e-12))[0])
    ind = list(np.array(ind)[ind_])
    Xi = X[ind]
    Yi = Y[ind]
    Fi = F[ind, :]

    # data point coordinates in the parent element
    xi  = ((Xi - x0) * (y2 - y0) - (Yi - y0) * (x2 - x0)) / \
          ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    eta = ((Xi - x0) * (y1 - y0) - (Yi - y0) * (x1 - x0)) / \
          ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))

    # basis functions (P1)
    phi0 = 1. - xi - eta
    phi1 = xi
    phi2 = eta

    # feed matrix a
    for j in range(len(ind)):
      a[ind[j], tri[i, 0]] = phi0[j]
      a[ind[j], tri[i, 1]] = phi1[j]
      a[ind[j], tri[i, 2]] = phi2[j]

  # convert a to sparse matrix
  a = sp.csr_matrix(a)

  # solve system
  f = np.zeros((nnode, nt))
  for j in range(nt):
    f[:, j], _, _, _, _, _, _, _, _, _ = spl.lsqr(a, F[:, j])

  # return projection
  if nt == 1:
    return f[:, 0]
  else:
    return f

################################################################################
# projection q0 ################################################################
################################################################################

def projection_q0(x, y, X, Y, F):

  """ Project point cloud data on a structured rectangular grid (data are projected on the center of the rectangle cells)

  Required parameters:
  x (NumPy array of shape (nx + 1)): x-coordinates of structured grid cells
  y (NumPy array of shape (ny + 1)): y-coordinates of structured grid cells
  X, Y (NumPy arrays of shape (N)): point cloud coordinates
  F (NumPy array of shape (N) or (N, M)): field values to project at points (axis 0) and different time steps (axis 1)

  Returns:
  f (NumPy array of shape (nx, ny) or (nx, ny, M)): projected data
  """

  # grid dimensions
  nx = len(x) - 1
  ny = len(y) - 1

  # extent F to two dimensions if needed
  if F.ndim == 1:
    F = np.reshape(F, (F.shape[0], 1))

  # second dimension of F
  nt = F.shape[1]

  # initialize
  f = np.zeros((nx, ny, nt))

  # for each structured grid cell
  for i in range(nx):
    for j in range(ny):

      # vertex coordinates
      xmin = x[i]
      ymin = y[j]
      xmax = x[i + 1]
      ymax = y[j + 1]

      # data inside the rectangle
      ind = list(np.where((X >= xmin) * (X <= xmax) * \
                          (Y >= ymin) * (Y <= ymax))[0])
      f[i, j, :] = np.mean(F[ind, :], axis = 0)

  # return projection
  if nt == 1:
    return f[:, :, 0]
  else:
    return f



