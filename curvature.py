""" Curvature

This module allows to calculate the local curvature of a P1 finite element field, based on local quadratic approximation

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np
import scipy.linalg

from mini_cloud import compute_mini_cloud



################################################################################
# curvature p1  ################################################################
################################################################################

def curvature_p1(x, y, f, tri, cloud = None, type = 'gaussian'):

  """ Calculate the local curvature of a P1 finite element field, based on calculation of local quadratic approximation

  Required parameters:
  x, y (NumPy arrays of size (n)): grid node coordinates
  f (NumPy array of size (n)): field values
  tri (NumPy array of size (m, 3)): triangle connectivity table

  Optional parameters:
  cloud (list of mini-cloud sub-lists)
  type (string, default = 'gaussian'): type of curvature
    'gaussian' --> r = (fxx * fyy - fxy ** 2) / (1 + fx ** 2 + fy ** 2) ** 2
    'laplacian' --> r = fxx + fyy

  Returns
  NumPy array of size (n)

  """

  # number of triangles
  ntri = len(tri)

  # number of nodes
  nnode = len(x)

  # if list of mini-clouds not provided:
  if cloud is None:

    # compute list of mini-clouds
    cloud = compute_mini_cloud(x, y, tri)

  # initialize gaussian curvature
  k = np.zeros(nnode)

  # for each node
  for i in range(nnode):

    # number of nodes in the mini-cloud
    ncloud = len(cloud[i])

    # coordinates of nodes in the mini-cloud
    xcloud = x[cloud[i]]
    ycloud = y[cloud[i]]

    # field values in the mini-cloud
    fcloud = f[cloud[i]]

    # initialize linear system
    a = np.zeros((ncloud, 6))
    b = np.zeros(ncloud)

    # build linear system
    a[:, 0] = xcloud * xcloud
    a[:, 1] = ycloud * ycloud
    a[:, 2] = xcloud * ycloud
    a[:, 3] = xcloud
    a[:, 4] = ycloud
    a[:, 5] = 1
    b[:] = fcloud

    # solve linear system
    coef, _, _, _ = scipy.linalg.lstsq(a, b, lapack_driver = 'gelsy')

    # first and second derivatives
    fx = 2 * coef[0] * x[i] + coef[2] * y[i] + coef[3]
    fy = 2 * coef[1] * y[i] + coef[2] * x[i] + coef[4]
    fxx = 2 * coef[0]
    fyy = 2 * coef[1]
    fxy = coef[2]

    # compute curvature
    if type == 'gaussian':
      k[i] = (fxx * fyy - fxy * fxy) / (1 + fx * fx + fy * fy) ** 2
    elif type == 'laplacian':
      k[i] = fxx + fyy

  # return curvature
  return k


################################################################################
# curvature p1 (contour integrals) #############################################
################################################################################

def curvature_p1_contour(x, y, f, tri):

  """ Calculate the local curvature of a P1 finite element field, based on calculation of local contour integrals

  Acknowledgments: V. Legat (Universite catholqiue de Louvain, Belgium) for giving the conceptual idea

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!! This function does not work !!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  """

  # number of triangles
  ntri = len(tri)

  # number of nodes
  nnode = len(x)

  # initialization
  r = np.zeros(nnode)

  # for each triangle
  for i in range(ntri):

    # global index of triangle vertices
    i0 = tri[i, 0]
    i1 = tri[i, 1]
    i2 = tri[i, 2]

    # coordinates of triangle vertices
    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    y0 = y[i0]
    y1 = y[i1]
    y2 = y[i2]

    # length of triangle edges
    d01 = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
    d12 = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    d20 = np.sqrt((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2))

    # outward normal components of triangle edges
    n01x = (y1 - y0) / d01
    n01y = (x0 - x1) / d01
    n12x = (y2 - y1) / d12
    n12y = (x1 - x2) / d12
    n20x = (y0 - y2) / d20
    n20y = (x2 - x0) / d20

    # jacobian determinant
    jac = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    # shape function derivatives
    phi0x = (y1 - y2) / jac
    phi1x = (y2 - y0) / jac
    phi2x = (y0 - y1) / jac
    phi0y = (x2 - x1) / jac
    phi1y = (x0 - x2) / jac
    phi2y = (x1 - x0) / jac

    # field values at triangle vertices
    f0 = f[i0]
    f1 = f[i1]
    f2 = f[i2]

    # contour integral edge 01
    r[i0] += .5 * d01 * (f0 * (n01x * phi0x + n01y * phi0y) + \
                         f1 * (n01x * phi1x + n01y * phi1y) + \
                         f2 * (n01x * phi2x + n01y * phi2y))
    r[i1] += .5 * d01 * (f0 * (n01x * phi0x + n01y * phi0y) + \
                         f1 * (n01x * phi1x + n01y * phi1y) + \
                         f2 * (n01x * phi2x + n01y * phi2y))

    # contour integral edge 12
    r[i1] += .5 * d12 * (f0 * (n12x * phi0x + n12y * phi0y) + \
                         f1 * (n12x * phi1x + n12y * phi1y) + \
                         f2 * (n12x * phi2x + n12y * phi2y)) / np.sqrt(2)
    r[i2] += .5 * d12 * (f0 * (n12x * phi0x + n12y * phi0y) + \
                         f1 * (n12x * phi1x + n12y * phi1y) + \
                         f2 * (n12x * phi2x + n12y * phi2y)) / np.sqrt(2)

    # contour integral edge 20
    r[i2] += .5 * d20 * (f0 * (n20x * phi0x + n20y * phi0y) + \
                         f1 * (n20x * phi1x + n20y * phi1y) + \
                         f2 * (n20x * phi2x + n20y * phi2y))
    r[i0] += .5 * d20 * (f0 * (n20x * phi0x + n20y * phi0y) + \
                         f1 * (n20x * phi1x + n20y * phi1y) + \
                         f2 * (n20x * phi2x + n20y * phi2y))

  return r

