import numpy as np

import ppmodules.selafin_io_pp as pps


################################################################################

class Telemac(object):

  def __init__(self, filename):

    """
    object to import and export Telemac (selafin) geometry/output files
    example:

    from read_write_telemac import Telemac
    tel = Telemac(filename)
    tel.import_header()
    v = tel.impor_data()

    """

    # open file
    slf = pps.ppSELAFIN(filename)

    # class atributes
    self.slf = slf


  ############################################################################

  def import_header(self):

    # class attributes
    slf = self.slf

    # read header
    slf.readHeader()
    slf.readTimes()

    # store into arrays
    times = slf.getTimes()
    vnames = slf.getVarNames()
    vunits = slf.getVarUnits()
    float_type, float_size = slf.getPrecision()
    nelem, npoin, ndp, ikle, ipobo, x, y = slf.getMesh()

    # class attributes
    self.times = times
    self.vnames = vnames
    self.vunits = vunits
    self.float_type = float_type
    self.float_size = float_size
    self.nelem = nelem
    self.npoin = npoin
    self.ndp = ndp
    self.ikle = ikle
    self.ipobo = ipobo
    self.x = x
    self.y = y


  ############################################################################

  def import_data(self, vname = None, step = None):

    # class attributes
    slf = self.slf
    times = self.times
    vnames = self.vnames
    npoin = self.npoin

    # variables
    if vname is None:
      vname = vnames
    elif type(vname) is str:
      vname = [vname]

    # number of variables
    if type(vname) is str:
      nv = 1
    else:
      nv = len(vname)

    # indices of variables to keep
    vid = []
    for i in range(nv):
      vid.append(map(str.strip, vnames).index(vname[i].strip()))
    vid = np.array(vid)

    # time steps
    if step is None:
      step = range(len(times))
    elif step == -1:
      step = [len(times) - 1]
    elif type(step) is int:
      step = [step]

    # number of time steps
    nt = len(step)

    # initialization of temporary array
    tmp = np.zeros((nt, nv, npoin))

    # read data
    for i in range(nt):
      slf.readVariables(step[i])
      tmp[i, :, :] = slf.getVarValues()[vid, :]

    # reshape data
    if nv == 1:
      if nt == 1:
        v = tmp[0, 0, :]
      else:
        v = tmp[:, 0, :].T
    else:
      v = []
      for i in range(nv):
        if nt == 1:
          v.append(tmp[0, i, :])
        else:
          v.append(tmp[:, i, :].T)

    # return data
    return v