import numpy as np
import sys

# pputils-v1.07
import ppmodules.selafin_io_pp as pps



def convert(msh_fn, slf_fn, cli_fn = None, bc = {}):

  # read gmsh file
  x, y, ikle, bnd, physical = load_msh(msh_fn)
  ipobo = generate_ipobo(x, y, bnd)

  # write telemac geometry file
  slf = pps.ppSELAFIN(slf_fn)
  slf.setPrecision('f', 4)
  slf.setTitle('')
  slf.setVarNames([])
  slf.setVarUnits([])
  slf.setIPARAM([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
  slf.setMesh(ikle.shape[0], x.shape[0], 3, ikle + 1, ipobo, x, y)
  slf.writeHeader()
  slf.close()

  # write telemac boundary condition file
  if cli_fn is not None:
    write_cli(cli_fn, ipobo, bnd, physical, bc)



def load_msh(filename):

  # read file
  lines = [line.rstrip('\n') for line in open(filename)]

  # mesh format
  i = lines.index('$MeshFormat')
  if int(lines[i + 1][0]) != 2:
    print()
    print('error in ogtools/gmsh2telemac.py:')
    print('error in load_msh: gmsh mesh file format must be version 2')
    print()
    sys.exit()

  # physical names (boundaries)
  i = lines.index('$PhysicalNames')
  n = int(lines[i + 1])
  physical = {}
  for line in lines[i + 2: i + 2 + n]:
    items = line.split(' ')
    physical[items[2][1:-1]] = int(items[1])

  # nodes
  i = lines.index('$Nodes')
  n = int(lines[i + 1])
  x = np.zeros(n)
  y = np.zeros(n)
  for line in lines[i + 2: i + 2 + n]:
    items = line.split(' ')
    x[int(items[0]) - 1] = float(items[1])
    y[int(items[0]) - 1] = float(items[2])

  # elements
  i = lines.index('$Elements')
  n = int(lines[i + 1])

  # count number of boundary segments (nbs) and number of triangles (nt)
  nbs = 0
  for line in lines[i + 2: i + 2 + n]:
    items = line.split(' ')
    if int(items[1]) == 1:
      nbs += 1
  nt = n - nbs

  # boundary segments
  bnd = np.zeros((nbs, 3), dtype = int)
  for line in lines[i + 2: i + 2 + nbs]:
    items = line.split(' ')
    bnd[int(items[0]) - 1, 0] = int(items[5]) - 1 # first node
    bnd[int(items[0]) - 1, 1] = int(items[6]) - 1 # second node
    bnd[int(items[0]) - 1, 2] = int(items[3]) - 1 # physical

  # triangles
  ikle = np.zeros((nt, 3), dtype = int)
  for line in lines[i + 2 + nbs: -1]:
    items = line.split(' ')
    n0 = int(items[5]) - 1 # first node
    n1 = int(items[6]) - 1 # second node
    n2 = int(items[7]) - 1 # third node
    if isclockwise([x[n0], x[n1], x[n2]], [y[n0], y[n1], y[n2]]):
      ikle[int(items[0]) - nbs - 1, 0] = n0
      ikle[int(items[0]) - nbs - 1, 1] = n2
      ikle[int(items[0]) - nbs - 1, 2] = n1
    else:
      ikle[int(items[0]) - nbs - 1, 0] = n0
      ikle[int(items[0]) - nbs - 1, 1] = n1
      ikle[int(items[0]) - nbs - 1, 2] = n2

  return x, y, ikle, bnd, physical



def generate_ipobo(x, y, bnd):

  #######################################################
  # !!! not tested for meshes with inner boundaries !!! #
  #######################################################

  ipobo = np.zeros(x.shape, dtype = int)

  first_loop = True
  loops = np.array([], dtype = int)

  # first node is lower left node
  i0 = np.argmin(x + y)
  loop = np.array([i0])

  while bnd.shape[0] > 0:
    # !!! might be a problem if len(i1) > 1
    j = int(np.argwhere(bnd[:, 0] == i0))
    i1 = bnd[j, 1]
    bnd = np.delete(bnd, j, 0)
    if i1 != loop[0]:
      loop = np.append(loop, i1)
      i0 = i1
    else:
      # outer loop nodes must be order anti-clockwise
      if first_loop:
        first_loop = False
        if isclockwise(x[loop], y[loop]):
          loop = np.append(loop[0], np.flipud(loop[1:]))
      # inner loop nodes must be order clockwise
      else:
        if not isclockwise(x[loop], y[loop]):
          loop = np.flipud(loop)
      loops = np.append(loops, loop)
      if bnd.shape[0] > 0:
        loop = np.array(bnd[0, 0])

  for i in range(len(loops)):
    ipobo[loops[i]] = i + 1

  return ipobo



def write_cli(filename, ipobo, bnd, physical, bc):

  lihbor = np.zeros(np.max(ipobo))
  liubor = np.zeros(np.max(ipobo))
  livbor = np.zeros(np.max(ipobo))
  hbor   = np.zeros(np.max(ipobo))
  ubor   = np.zeros(np.max(ipobo))
  vbor   = np.zeros(np.max(ipobo))
  aubor  = np.zeros(np.max(ipobo))
  litbor = np.zeros(np.max(ipobo))
  tbor   = np.zeros(np.max(ipobo))
  atbor  = np.zeros(np.max(ipobo))
  btbor  = np.zeros(np.max(ipobo))
  n      = np.zeros(np.max(ipobo))
  k      = np.zeros(np.max(ipobo))

  # solid boundaries by default
  for i in range(bnd.shape[0]):
    for j in range(2):
      global_id = bnd[i, j]
      bnd_id = ipobo[global_id]
      lihbor[bnd_id - 1] = 2
      liubor[bnd_id - 1] = 2
      livbor[bnd_id - 1] = 2
      litbor[bnd_id - 1] = 2
      n[bnd_id - 1] = global_id + 1
      k[bnd_id - 1] = bnd_id

  # other boundaries
  for key in bc.keys():
    seg_ids = np.argwhere(bnd[:, 2] == physical[key] - 1)
    global_ids = np.unique(bnd[seg_ids, :-1].flatten())
    bnd_ids = ipobo[global_ids]
    lihbor[bnd_ids - 1] = bc[key][0]
    liubor[bnd_ids - 1] = bc[key][1]
    livbor[bnd_ids - 1] = bc[key][2]
    hbor[bnd_ids - 1] = bc[key][3]
    ubor[bnd_ids - 1] = bc[key][4]
    vbor[bnd_ids - 1] = bc[key][5]
    aubor[bnd_ids - 1] = bc[key][6]
    litbor[bnd_ids - 1] = bc[key][7]
    tbor[bnd_ids - 1] = bc[key][8]
    atbor[bnd_ids - 1] = bc[key][9]
    btbor[bnd_ids - 1] = bc[key][10]
    n[bnd_ids - 1] = global_ids + 1
    k[bnd_ids - 1] = bnd_ids

  # write file
  cli = open(filename, 'w')
  for i in range(np.max(ipobo)):
    cli.write('%d %d %d %g %g %g %g %d %g %g %g %d %d \n' % \
              (lihbor[i], liubor[i], livbor[i], hbor[i], ubor[i], vbor[i], \
               aubor[i], litbor[i], tbor[i], atbor[i], btbor[i], n[i], k[i]))



def isclockwise(x, y):

  area = 0.
  for i in range(len(x)):
    j = np.mod(i + 1, len(x))
    area += (x[j] - x[i]) * (y[j] + y[i])

  if area > 0:
    return True
  else:
    return False

