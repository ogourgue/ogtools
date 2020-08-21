""" Diffusion

This module allows to smooth data fields by applying a diffusion operator

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


################################################################################
# diffusion p1 #################################################################
################################################################################

def diffusion_p1(x, y, f, tri, nu, dt, t = 1):

  """ Apply a diffusion operator to a P1 finite element field. The diffusion coefficient is applied over a period t, using a series of time steps dt < t.

  Required parameters:
  x, y (NumPy arrays of shape (n)): grid node coordinates
  f (NumPy array of shape (n)): field values before diffusion
  tri (NumPy array of shape (m, 3)): triangle connectivity table
  nu (float): diffusion coefficient (m^2/s)
  dt (float): time step (s)

  Optional parameters:
  t (float, default = 1) duration of diffusion

  Returns
  NumPy array of shape (n): field values after diffusion

  """

  ##################
  # initialization #
  ##################

  npoin = len(x)
  ntri = len(tri)
  a = lil_matrix((npoin, npoin))
  b = lil_matrix((npoin, npoin))
  jac = np.zeros(ntri)


  ########################
  # jacobian determinant #
  ########################

  for i in range(len(tri)):

    # local coordinates
    x0 = x[tri[i, 0]]
    x1 = x[tri[i, 1]]
    x2 = x[tri[i, 2]]
    y0 = y[tri[i, 0]]
    y1 = y[tri[i, 1]]
    y2 = y[tri[i, 2]]

    # jacobian determinant
    jac[i] = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


  #################
  # feed matrix a #
  #################

  for i in range(ntri):

    # local matrix
    a_loc = jac[i] / 24. * np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])

    # global indices of local nodes
    i0 = tri[i, 0]
    i1 = tri[i, 1]
    i2 = tri[i, 2]

    # feed matrix
    a[i0, i0] += a_loc[0, 0]
    a[i0, i1] += a_loc[0, 1]
    a[i0, i2] += a_loc[0, 2]
    a[i1, i0] += a_loc[1, 0]
    a[i1, i1] += a_loc[1, 1]
    a[i1, i2] += a_loc[1, 2]
    a[i2, i0] += a_loc[2, 0]
    a[i2, i1] += a_loc[2, 1]
    a[i2, i2] += a_loc[2, 2]


  #################
  # feed matrix b #
  #################

  for i in range(ntri):

    # local coordinates
    x0 = x[tri[i, 0]]
    x1 = x[tri[i, 1]]
    x2 = x[tri[i, 2]]
    y0 = y[tri[i, 0]]
    y1 = y[tri[i, 1]]
    y2 = y[tri[i, 2]]

    # shape function derivatives
    phi0x = (y1 - y2) / jac[i]
    phi1x = (y2 - y0) / jac[i]
    phi2x = (y0 - y1) / jac[i]
    phi0y = (x2 - x1) / jac[i]
    phi1y = (x0 - x2) / jac[i]
    phi2y = (x1 - x0) / jac[i]

    # local matrix
    b_loc = np.zeros((3, 3))
    b_loc[0, 0] = -.5 * jac[i] * (phi0x * phi0x + phi0y * phi0y)
    b_loc[0, 1] = -.5 * jac[i] * (phi0x * phi1x + phi0y * phi1y)
    b_loc[0, 2] = -.5 * jac[i] * (phi0x * phi2x + phi0y * phi2y)
    b_loc[1, 0] = -.5 * jac[i] * (phi1x * phi0x + phi1y * phi0y)
    b_loc[1, 1] = -.5 * jac[i] * (phi1x * phi1x + phi1y * phi1y)
    b_loc[1, 2] = -.5 * jac[i] * (phi1x * phi2x + phi1y * phi2y)
    b_loc[2, 0] = -.5 * jac[i] * (phi2x * phi0x + phi2y * phi0y)
    b_loc[2, 1] = -.5 * jac[i] * (phi2x * phi1x + phi2y * phi1y)
    b_loc[2, 2] = -.5 * jac[i] * (phi2x * phi2x + phi2y * phi2y)

    # global indices of local nodes
    i0 = tri[i, 0]
    i1 = tri[i, 1]
    i2 = tri[i, 2]

    # feed matrix
    b[i0, i0] += b_loc[0, 0]
    b[i0, i1] += b_loc[0, 1]
    b[i0, i2] += b_loc[0, 2]
    b[i1, i0] += b_loc[1, 0]
    b[i1, i1] += b_loc[1, 1]
    b[i1, i2] += b_loc[1, 2]
    b[i2, i0] += b_loc[2, 0]
    b[i2, i1] += b_loc[2, 1]
    b[i2, i2] += b_loc[2, 2]


  ##################
  # diffusion loop #
  ##################

  # number of time steps
  nt = np.int(np.ceil(t / dt))

  # convert to sparse matrices
  a = csr_matrix(a)
  b = csr_matrix(b)

  # initialization
  f1 = f.copy()

  # diffusion loop
  for i in range(nt):

    # initialization
    f0 = f1.copy()

    # solve linear matrix equation
    f1 = spsolve(a, a.dot(f0) + nu * dt * b.dot(f0))

  # finalization
  return f1