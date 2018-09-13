import numpy as np


################################################################################

def compute_mini_cloud(x, y, tri):

  """
  computes mini-cloud of each node of a trinagular grid, as the list of all neighboring nodes sharing at least one triangle with it

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


################################################################################

def compute_mini_cloud_radius(x, y, tri, r):

  """
  computes mini-cloud of each node of a trinagular grid, as the list of all nodes in a certain radius

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
    d2 = (x_tmp - x0) * (x_tmp - x0) + (y_tmp - y0) * (y_tmp - y0)
    j = tmp_j[d2 <= r * r]

    # add node indices to mini-cloud
    cloud[i] = j

  print cloud

  return cloud


################################################################################

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


################################################################################

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