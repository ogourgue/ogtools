# author: O. Gourgue
# (University of Antwerp, Belgium & Boston University, MA, USA)

import numpy as np
from scipy import spatial


################################################################################

def compute_upl(x, y, creek):

  """
  compute unchanneled path length from boolean creek field

  input:
    - x: array of shape (n) with x-coordinates of grid nodes
    - y: array of shape (n) with y-coordinates of grid nodes
    - creek: array of shape (n, m) with boolean (1 is creek, 0 is not) - m is number of time steps

  output:
    - upl: array of shape (n, m) with unchanneled path length - m is number of time steps
  """

  # initialize
  upl = np.zeros(creek.shape)

  # case of one time step
  if creek.ndim == 1:
    creek = creek.reshape((creek.shape[0], 1))
    upl = upl.reshape((upl.shape[0], 1))

  # for each time step
  for i in range(creek.shape[1]):

    # creek nodes
    creek_ind = np.flatnonzero(creek[:, i])
    creek_xy = np.array([x[creek_ind], y[creek_ind]]).T

    # non-creek nodes
    non_creek_ind = np.flatnonzero(creek[:, i] == 0)
    non_creek_xy = np.array([x[non_creek_ind], y[non_creek_ind]]).T

    # unchanneled path length
    tree = spatial.KDTree(creek_xy)
    non_creek_upl, ind = tree.query(non_creek_xy)
    upl[non_creek_ind, i] = non_creek_upl

  if upl.shape[1] == 1:
    upl = upl.reshape((upl.shape[0]))

  # return
  return upl