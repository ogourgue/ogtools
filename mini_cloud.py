# author: O. Gourgue (University of Antwerp, Belgium)

import numpy as np


################################################################################

def compute_mini_cloud(x, y, tri):

  """
  compute mini-cloud of each node of a trinagular grid, as the list of all neighboring nodes sharing at least one triangle with it

  input:
    - x: array of shape (n) with x-coordinates of grid nodes
    - y: array of shape (n) with y-coordinates of grid nodes
    - tri: array of shape (m, 3) with connectivity table of grid triangles

  output:
    - cloud: list of n arrays of different shapes with mini-cloud node ids
  """

  # number of triangles
  ntri = len(tri)

  # number of nodes
  nnode = len(x)

  # integer type
  if nnode <= 127:
    dtype = np.int8
  elif nnode <= 32767:
    dtype = np.int16
  elif nnode <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

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

  # for each node
  for i in range(nnode):
    cloud[i] = np.array(cloud[i], dtype = dtype)

  return cloud


################################################################################

def compute_mini_cloud_radius(x, y, r, nmax = None):

  """
  compute mini-cloud of each node of a trinagular grid, as the list of all nodes in a certain radius

  input:
    - x: array of shape (n) with x-coordinates grid nodes
    - y: array of shape (n) with y-coordinates grid nodes
    - tri: array of shape (m, 3) with connectivity table of grid triangles
    - nmax: maximum number of nodes per mini-cloud (random selection if necessary)

  output:
    - cloud: list of n arrays of different shapes with mini-cloud node ids
  """

  # number of nodes
  nnode = len(x)

  # integer type
  if nnode <= 127:
    dtype = np.int8
  elif nnode <= 32767:
    dtype = np.int16
  elif nnode <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # initialize list of mini-clouds
  cloud = [None] * nnode

  # for each node
  for i in range(nnode):

    # node coordinates
    x0 = x[i]
    y0 = y[i]

    # data within square bounding box of length (2r)
    tmp_j = (x >= x0 - r) * (x <= x0 + r) * (y >= y0 - r) * (y <= y0 + r)
    tmp_x = x[tmp_j]
    tmp_y = y[tmp_j]

    # data within circle of radius r
    d2 = np.zeros(nnode)
    d2[tmp_j] = (tmp_x - x0) * (tmp_x - x0) + (tmp_y - y0) * (tmp_y - y0)
    tmp_j *= (d2 <= r * r)
    j = np.array(np.argwhere(tmp_j).reshape(-1), dtype = dtype)

    # random selection if too many nodes in the mini-cloud
    if len(j) > nmax and nmax is not None:
      j = j[np.random.randint(0, len(j), nmax)]

    # add node indices to mini-cloud
    cloud[i] = j

  return cloud


################################################################################

def export_mini_cloud(cloud, filename):

  """
  export list of mini-clouds

  input:
    - cloud: list of n arrays with mini-cloud node ids
    - filename: binary file name
  """

  # number of mini-clouds
  ncloud = len(cloud)

  # integer type
  if ncloud <= 127:
    dtype = np.int8
  elif ncloud <= 32767:
    dtype = np.int16
  elif ncloud <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # open file
  file = open(filename, 'w')

  # number of mini-clouds
  np.array(ncloud, dtype = int).tofile(file)

  # for each mini-cloud
  for i in range(ncloud):

    # number of nodes
    np.array(len(cloud[i]), dtype = dtype).tofile(file)

    # node indices
    np.array(cloud[i], dtype = dtype).tofile(file)

  # close file
  file.close()


################################################################################

def import_mini_cloud(filename):

  """
  import list of mini-clouds

  input:
    - filename: binary file name

  output:
    - cloud: list of n arrays with mini-cloud node ids
  """

  # open file
  file = open(filename, 'r')

  # number of mini-clouds
  ncloud = np.fromfile(file, dtype = int, count = 1)[0]

  # integer type
  if ncloud <= 127:
    dtype = np.int8
  elif ncloud <= 32767:
    dtype = np.int16
  elif ncloud <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # initialize list of mini-clouds
  cloud = [None] * ncloud
  for i in range(ncloud):
    cloud[i] = []

  # for each mini-cloud
  for i in range(ncloud):

    # number of nodes
    n = np.fromfile(file, dtype = dtype, count = 1)[0]

    # node indices
    cloud[i] = np.fromfile(file, dtype = dtype, count = n)

  # close file
  file.close()

  return cloud