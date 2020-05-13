""" Interpolation

This module allows to calculate interpolation of discrete fields

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np



################################################################################
# interpolation barycentric ####################################################
################################################################################

def interpolation_barycentric(x, y, f, tri, X, Y, nodata_value = np.nan):

  """ Interpolate a P1 finite element field on a structured grid using barycentric coordinates

  Required parameters:
  x, y (NumPy arrays of shape (n)): triangle grid node coordinates
  f (NumPy array of shape (n)): field values to interpolate
  tri (NumPy array of shape (m, 3)): triangle connectivity table
  X, Y (NumPy arrays of shape (M) and (N)): structured grid coordinates

  Optional parameter:
  nodata_value (number, default = np.nan): field value outside the unstructured grid

  Returns:
  NumPy array of shape (M, N): interpolated data

  """

  # initialize output
  F = np.zeros((len(X), len(Y))) + nodata_value

  # for each triangle
  for i in range(tri.shape[0]):

    # triangle vertex coordinates
    x0 = x[tri[i, 0]]
    x1 = x[tri[i, 1]]
    x2 = x[tri[i, 2]]
    y0 = y[tri[i, 0]]
    y1 = y[tri[i, 1]]
    y2 = y[tri[i, 2]]

    # field values on triangle vertex coordinates
    f0 = f[tri[i, 0]]
    f1 = f[tri[i, 1]]
    f2 = f[tri[i, 2]]

    # bounding box coordinates
    xmin = np.min([x0, x1, x2])
    xmax = np.max([x0, x1, x2])
    ymin = np.min([y0, y1, y2])
    ymax = np.max([y0, y1, y2])

    # indices of structured grid points inside the bounding box
    INDX = np.where((X > xmin) * (X < xmax))[0]
    INDY = np.where((Y > ymin) * (Y < ymax))[0]

    # mini mesh grid
    XLOC, YLOC = np.meshgrid(X[INDX], Y[INDY], indexing = 'ij')

    # barycentric coordinates
    S0 = ((y1 - y2) * (XLOC - x2) + (x2 - x1) * (YLOC - y2)) \
       / ((y1 - y2) * (x0   - x2) + (x2 - x1) * (y0   - y2))
    S1 = ((y2 - y0) * (XLOC - x2) + (x0 - x2) * (YLOC - y2)) \
       / ((y1 - y2) * (x0   - x2) + (x2 - x1) * (y0   - y2))
    S2 = 1 - S0 - S1

    # barycentric interpolation
    FLOC = f0 * S0 + f1 * S1 + f2 * S2

    # update output array for structured grid points inside triangle
    # (that is, for which barycentric coordinates are all positive)
    for j in range(len(INDX)):
      for k in range(len(INDY)):
        if S0[j, k] >= 0 and S1[j, k] >= 0 and S2[j, k] >= 0:
          F[INDX[j], INDY[k]] = FLOC[j, k]

  return F