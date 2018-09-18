# author: O. Gourgue (University of Antwerp, Belgium)

import numpy as np
import os
import scipy.linalg
import sys

import mini_cloud as mc


################################################################################

def compute_boolean_creek(x, y, z, hc, radius = None, cloud_fn = None, \
                          combine_radius_logical = True, \
                          combine_time_logical = False, \
                          linear_detrend_logical = False):

  """
  extract tidal creek from elevation maps, based on median neighborhood analysis - inspired by first step of multi-step approach by Liu et al. (2015, dx.doi.org/10.1016/j.jhydrol.2015.05.058)

  input:
    - x: array of shape (n) with x-coordinates of grid nodes
    - y: array of shape (n) with y-coordinates of grid nodes
    - z: array of shape (n, m) with bottome elevation at grid nodes (axis 0) and different time steps (axis 1)
    - hc: threshold residual (median(z) - z) above which node is considered as creek
    - radius (optional): (list of) mini-cloud radii(-us)
    - cloud_fn (optional): (list of) mini-cloud binary file name(s)
    - combine_radius_logical (default is True): combine the results for each mini-cloud radius
    - combine_time_logical (default is False): combine the results for each time step (combine_radius_logical must be True)
    - linear_detrend_logical (default is False): combine median analysis results with a median analysis applied to a linearly detrent elevation within each mini-cloud (experimental, not tested in depth)

  output:
    - creek: array of shape (n, m) with boolean (1 is creek, 0 is not) - m is number of radius considered - creek shape is (n) if only one radius
  """


  ##################
  # initialization #
  ##################

  # radius as a list
  if radius is not None and type(radius) is not list:
    radius = [radius]

  # cloud_fn as a list
  if cloud_fn is not None and type(cloud_fn) is not list:
    cloud_fn = [cloud_fn]

  # number of radius and/or mini-cloud file names
  if radius is not None:
    nr = len(radius)
  elif cloud_fn is not None:
    nr = len(cloud_fn)

  # check optional input
  if radius is not None and cloud_fn is not None:
    if len(radius) is not len(cloud_fn):
      print 'number of radius must the same as number cloud_fn'
      sys.exit()
  if radius is None and cloud_fn is None:
    print 'radius or cloud_fn must be defined (both also accepted)'
    sys.exit()

  # check combine options
  if combine_time_logical and not combine_radius_logical:
    print 'combined in radius is mandatory if combined in time'
    sys.exit()

  # number of nodes
  nnode = len(x)

  # number of time steps
  if z.ndim == 1:
    nt = 1
  else:
    nt = z.shape[1]

  # initialize boolean creek
  if combine_radius_logical:
    if combine_time_logical:
      creek = np.zeros(nnode)
    else:
      creek = np.zeros(z.shape)
  else:
    creek = []


  # for each radius
  for i in range(nr):


    #####################################
    # compute/import/export mini-clouds #
    #####################################

    # if list of radius and mini-cloud file names are given
    if radius is not None and cloud_fn is not None:

      # import mini-cloud
      if os.path.isfile(cloud_fn[i]):
        cloud = mc.import_mini_cloud(cloud_fn[i])

      # or compute & export mini-cloud
      else:
        cloud = mc.compute_mini_cloud_radius(x, y, r)
        mc.export_mini_cloud(cloud, cloud_fn[i])

    # if list of mini-cloud file names is not given
    elif cloud_fn is None:

      # compute mini-cloud
      cloud = mc.compute_mini_cloud_radius(x, y, r)

    # if list of radius is not given
    elif radius is None:

      # import mini-cloud
      cloud = mc.import_mini_cloud(cloud_fn[i])


    #########################
    # compute boolean creek #
    #########################

    # compute median
    median = np.zeros(z.shape)
    for j in range(len(x)):
      median[j, :] = np.median(z[cloud[j]], axis = 0)

    # compute residual
    res = median - z

    # compute boolean creek
    tmp_1 = (res > hc)


    #################################
    # compute detrent boolean creek #
    #################################

    if linear_detrend_logical:

      # for each mini-cloud
      for j in range(nnode):

        # number of nodes in the mini-cloud
        ncloud = len(cloud[j])

        # coordinates of nodes in the mini-cloud
        xcloud = x[cloud[j]]
        ycloud = y[cloud[j]]

        # initialization
        zl = np.zeros(z.shape)

        # bottom elevation in the mini-cloud
        if nt == 1:
          zcloud = z[cloud[j]]
        else:
          zcloud = z[cloud[j], :]

        # initialize linear system
        a = np.zeros((ncloud, 3))
        if nt == 1:
          b = np.zeros(ncloud)
        else:
          b = np.zeros((ncloud, nt))

        # build linear system
        a[:, 0] = xcloud
        a[:, 1] = ycloud
        a[:, 2] = 1.
        b[:] = zcloud

        # solve linear system
        coef, _, _, _ = scipy.linalg.lstsq(a, b, lapack_driver = 'gelsy')

        # linear detrent elevation
        if nt == 1:
          zl[j] = coef[0] * x[j] + coef[1] * y[j] + coef[2]
        else:
          zl[j, :] = coef[0, :] * x[j] + coef[1, :] * y[j] + coef[2, :]

      # compute median
      median = np.zeros(z.shape)
      for j in range(nnode):
        median[j, :] = np.median(z[cloud[j]] - zl[j], axis = 0)

      # compute residual
      res = median - z

      # compute boolean creek
      tmp_2 = (res > hc)

    else:
      tmp_2 = np.zeros(z.shape, dtype = bool)


    ##########################
    # combine boolean creeks #
    ##########################

    # compute boolean creek
    if combine_radius_logical:
      if combine_time_logical:
        creek += np.sum(tmp_1 + tmp_2, axis = 1)
      else:
        creek += tmp_1 + tmp_2
    else:
      creek.append(tmp_1 + tmp_2)


  # return boolean creek
  return creek