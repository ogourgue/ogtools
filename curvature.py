# author: O. Gourgue (University of Antwerp, Belgium)

import numpy as np
import scipy.linalg


def gaussian_curvature_p1(x, y, f, tri, cloud = None):

  """
  calculate the local gaussian curvature of the P1 finite element field f(x, y) on the grid defined by x, y (triangle vertex coordinates) and tri (connectivity table), based on calculation of local quadratic approximation

  input:
  x: array of shape (n)
  y: array of shape (n)
  f: array of shape (n)
  tri: array of shape (m, 3)

  output:
  r: array of shape (n)
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
    a[:, 5] = 1.
    b[:] = fcloud

    # solve linear system
    coef, _, _, _ = scipy.linalg.lstsq(a, b, lapack_driver = 'gelsy')

    # first and second derivatives
    fx = 2. * coef[0] * x[i] + coef[2] * y[i] + coef[3]
    fy = 2. * coef[1] * y[i] + coef[2] * x[i] + coef[4]
    fxx = 2. * coef[0]
    fyy = 2. * coef[1]
    fxy = coef[2]

    # gaussian curvature
    k[i] = (fxx * fyy - fxy * fxy) / (1. + fx * fx + fy * fy) ** 2.

  # return gaussian curvature
  return k


def compute_mini_cloud(x, y, tri):

  """
  computes mini-cloud of each node of a trinagular grid (the mini-cloud of a node is the list of all neighboring nodes sharing at least one triangle with it)

  input:
  x: array of shape (n)
  y: array of shape (n)
  tri: array of shape (m, 3)

  output:
  cloud: list of n sub-lists (each sub-list has differents lengths)
  """

  # number of triangles
  ntri = len(tri)

  # number of nodes
  nnode = len(x)

  # initialize list of mini-clouds
  cloud = [None] * nnode
  for i in range(nnode):
    cloud[i] = []

  # for each triangle
  for i in range(ntri):

    # for each triangle vertex
    for j in tri[i, :]:

      # for each triangle vertex
      for k in tri[i, :]:

        # add node j in mini-cloud k if not already in
        if j not in cloud[k]:
          cloud[k].append(j)

  return cloud


def export_mini_cloud(cloud, filename):

  """
  export list of mini-clouds

  input:
  cloud: list of n sub-lists (each sub-list has differents lengths)
  filename: binary file name
  """

  # open file
  file = open(filename, 'w')

  # number of mini-clouds
  np.array(len(cloud), dtype = int).tofile(file)

  # for each mini-cloud
  for i in range(len(cloud)):

    # number of nodes
    np.array(len(cloud[i]), dtype = int).tofile(file)

    # node indices
    np.array(cloud[i], dtype = int).tofile(file)

  # close file
  file.close()


def import_mini_cloud(filename):

  """
  import list of mini-clouds

  input:
  cloud: list of n sub-lists (each sub-list has differents lengths)
  filename: binary file name
  """

  # open file
  file = open(filename, 'r')

  # number of mini-clouds
  ncloud = np.fromfile(file, dtype = int, count = 1)[0]

  # initialize list of mini-clouds
  cloud = [None] * ncloud
  for i in range(ncloud):
    cloud[i] = []

  # for each mini-cloud
  for i in range(ncloud):

    # number of nodes
    n = np.fromfile(file, dtype = int, count = 1)[0]

    # node indices
    cloud[i] = list(np.fromfile(file, dtype = int, count = n))

  # close file
  file.close()



############################################
# !!! the function below does not work !!! #
############################################

def curvature_p1_contour(x, y, f, tri):

  """
  calculate the local curvature of the P1 finite element field f(x, y) on the grid defined by x, y (triangle vertex coordinates) and tri (connectivity table), based on calculation of local contour integrals

  input:
  x: array of shape (n)
  y: array of shape (n)
  f: array of shape (n)
  tri: array of shape (m, 3)

  output:
  r: array of shape (n)

  acknowledgments: V. Legat (Universite catholqiue de Louvain, Belgium) for giving the conceptual idea
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
                         f2 * (n12x * phi2x + n12y * phi2y)) / np.sqrt(2.)
    r[i2] += .5 * d12 * (f0 * (n12x * phi0x + n12y * phi0y) + \
                         f1 * (n12x * phi1x + n12y * phi1y) + \
                         f2 * (n12x * phi2x + n12y * phi2y)) / np.sqrt(2.)

    # contour integral edge 20
    r[i2] += .5 * d20 * (f0 * (n20x * phi0x + n20y * phi0y) + \
                         f1 * (n20x * phi1x + n20y * phi1y) + \
                         f2 * (n20x * phi2x + n20y * phi2y))
    r[i0] += .5 * d20 * (f0 * (n20x * phi0x + n20y * phi0y) + \
                         f1 * (n20x * phi1x + n20y * phi1y) + \
                         f2 * (n20x * phi2x + n20y * phi2y))

  return r

