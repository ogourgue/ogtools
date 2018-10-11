# author: O. Gourgue (University of Antwerp, Belgium)

import numpy as np


################################################################################

def integral_p1(x, y, v, tri, mask = None, div_logical = False):

  """
  compute integral p1-field v

  input:
    - x: array of shape (n) with x-coordinates of grid nodes
    - y: array of shape (n) with y-coordinates of grid nodes
    - v: array of shape (n, m) with field value at grid nodes (axis 0) and different time step (axis 1) - array of shape (n) if only 1 time step
    - tri: array of shape (p, 3) with connectivity table of grid triangles
    - mask: logical array of shape (n) or (n, m) - true where integral must be computed
    - div_logical: integral divided by surface if true

  output:
    - v_int: integral of v over the entire domain
  """

  # number of nodes
  nnode = len(x)

  # number of triangles
  ntri = tri.shape[0]

  # number of time steps
  if v.ndim == 1:
    nt = 1
  else:
    nt = v.shape[1]

  # triangle node coordinates
  x0 = x[tri[:, 0]]
  x1 = x[tri[:, 1]]
  x2 = x[tri[:, 2]]
  y0 = y[tri[:, 0]]
  y1 = y[tri[:, 1]]
  y2 = y[tri[:, 2]]

  # triangle surface
  s = .5 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))

  # apply mask
  if mask is not None:
    if v.ndim == mask.ndim:
      tmp = v * mask
    else:
      tmp = v * np.reshape(mask, (nnode, 1))
  else:
    tmp = v

  # average over triangles
  if v.ndim == 1:
    v_mean = np.mean(tmp[tri])
  else:
    v_mean = np.mean(tmp[tri, :], axis = 1)

  # integral
  if v.ndim == 1:
    v_int = np.sum(v_mean * s)
  else:
    v_int = np.sum(v_mean * np.reshape(s, (ntri, 1)), axis = 0)

  # division by surface
  if div_logical:
    if mask is None:
      s_tot = np.sum(s)
    elif mask.ndim == 1:
      s_tot = np.sum(s * np.mean(mask[tri]))
    else:
      s_tot = np.sum(np.reshape(s, (ntri, 1)) * np.mean(mask[tri], axis = 1), \
                     axis = 0)
    v_int /= s_tot

  # return value
  return v_int