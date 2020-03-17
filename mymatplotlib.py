""" My Matplotlib

This module redefines some useful matplotlib functions

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


from mpl_toolkits.axes_grid1 import make_axes_locatable




################################################################################
# my colorbar ##################################################################
################################################################################

def mycolorbar(mappable, label = None, nticks = 0, orientation = 'vertical'):

  """ Create a custom colorbar

  Required parameter:
  mappable:

  Optional parameters:
  label:
  nticks:
  orientation (default = 'vertical')

  Returns:
  colorbar

  """




  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  if orientation is 'vertical':
    cax = divider.append_axes("right", size = "5%", pad = .05)
  elif orientation is 'horizontal':
    cax = divider.append_axes("bottom", size = "5%", pad = .05)

  if label is None:
    cb = fig.colorbar(mappable, cax = cax, orientation = orientation)
  else:
    cb = fig.colorbar(mappable, cax = cax, label = label, \
                      orientation = orientation)

  # change number of ticks
  if nticks > 0:
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins = nticks)
    cb.locator = tick_locator
    cb.update_ticks()

  return cb