# author: O. Gourgue (University of Antwerp, Belgium)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def projection_p1(x, y, tri, X, Y, F):

  """
  project the data F(X, Y) on a P1 finite element field on the grid defined by x, y (tria ngle vertex coordinates) and tri (connectivity table)
  x: array of shape (n)
  y: array of shape (n)
  tri: array of shape (m, 3)
  X: array of size (N)
  Y: array of size (N)
  F: array of size (N)
  """

  # test input data

  # number of triangles
  ntri = len(tri)

  # number of grid nodes
  nnode = len(x)

  # number of data points to project
  ndata = len(X)

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
    Fi = F[ind]

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
    Fi = F[ind]

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
  f = spl.lsqr(a, F)[0]

  # return projection
  return f