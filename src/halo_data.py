#######################################################################
#
# This module is dedicated to routines related to the halo data
# of the cubed-sphere grid
#
# Luan Santos - October 2022
######################################################################

import numpy as np
from constants import nbfaces

######################################################################
# This routine get the halo data of Q needed for interpolation
######################################################################
def get_halo_data_interpolation(Q, cs_grid):
    N   = cs_grid.N
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Interior cells index
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Halo data
    halodata_east  = np.zeros((ngl, N+ng, nbfaces))
    halodata_west  = np.zeros((ngl, N+ng, nbfaces))
    halodata_north = np.zeros((N+ng, ngl, nbfaces))
    halodata_south = np.zeros((N+ng, ngl, nbfaces))

    # --------------------- Panel 0 ----------------------------
    p = 0
    north = 4
    south = 5
    east  = 1
    west  = 3

    # Data of panel 0 from east
    halodata_east[:,:,p] = Q[i0:i0+ngr,:,east] # Panel 1

    # Data of panel 0 from  west
    halodata_west[:,:,p] = Q[iend-ngl:iend,:, west] # Panel 3

    # Data of panel 0 from north
    halodata_north[:,:,p] = Q[:,j0:j0+ngr, north] # Panel 4

    # Data of panel 0 from south
    halodata_south[:,:,p] = Q[:,jend-ngl:jend, south] # Panel 5

    # --------------------- Panel 1 ----------------------------
    p = 1
    north = 4
    south = 5
    east  = 2
    west  = 0

    # Data of panel 1 from east
    halodata_east[:,:,p] = Q[i0:i0+ngr,:,east] # Panel 2

    # Data of panel 1 from west
    halodata_west[:,:,p] = Q[iend-ngl:iend,:, west] # Panel 0

    # Data of panel 1 from north
    support_values = Q[iend-ngr:iend,:, north] # Panel 4
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_north[:,:,p] = support_values

    # Data of panel 1 from south
    support_values = Q[iend-ngl:iend, :, south] # Panel 5
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=0)
    halodata_south[:,:,p] = support_values

    # --------------------- Panel 2 ----------------------------
    p = 2
    north = 4
    south = 5
    east  = 3
    west  = 1

    # Data of panel 2 from east
    halodata_east[:,:,p] = Q[i0:i0+ngr,:,east] # Panel 3

    # Data of panel 2 from west
    halodata_west[:,:,p] = Q[iend-ngl:iend,:, west] # Panel 1

    # Data of panel 2 from north
    support_values = Q[:,jend-ngr:jend,north] # Panel 4
    support_values = np.flip(support_values,axis=0)
    support_values = np.flip(support_values,axis=1)
    halodata_north[:,:,p] = support_values

    # Data of panel 2 from south
    support_values = Q[:,j0:j0+ngr, south] # Panel 5
    support_values = np.flip(support_values,axis=1)
    support_values = np.flip(support_values,axis=0)
    halodata_south[:,:,p] = support_values

    # --------------------- Panel 3 ----------------------------
    p = 3
    north = 4
    south = 5
    east  = 0
    west  = 2

    # Data of panel 3 from east
    halodata_east[:,:,p] = Q[i0:i0+ngr,:,east] # Panel 0

    # Data of panel 3 from west
    halodata_west[:,:,p]= Q[iend-ngl:iend,:, west] # Panel 2

    # Data of panel 3 from north
    support_values = Q[i0:i0+ngr,:,north] # Panel 4
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=0)
    halodata_north[:,:,p] = support_values

    # Data of panel 0 from south
    support_values = Q[i0:i0+ngr,:, south] # Panel 5
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_south[:,:,p] = support_values

    # --------------------- Panel 4 ----------------------------
    p = 4
    north = 2
    south = 0
    east  = 1
    west  = 3

    # Data of panel 4 from east
    support_values = Q[:,jend-ngl:jend,east] # Panel 3
    support_values = np.flip(support_values,axis=1)
    support_values = np.transpose(support_values)
    halodata_east[:,:,p] = support_values

    # Data of panel 4 from west
    support_values = Q[:,jend-ngl:jend, west] # Panel 1
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_west[:,:,p] = support_values

    # Data of panel 4 from north
    support_values = Q[:,jend-ngr:jend, north] # Panel 2
    support_values = np.flip(support_values,axis=0)
    support_values = np.flip(support_values,axis=1)
    halodata_north[:,:,p] = support_values

    # Interpolate ghost cells of panel 4 at south
    halodata_south[:,:,p] = Q[:,jend-ngl:jend, south] # Panel 0


    # --------------------- Panel 5 ----------------------------
    p = 5
    north = 0
    south = 2
    east  = 1
    west  = 3

    # Data of panel 5 from east
    support_values = Q[:,j0:j0+ngr,east] # Panel 1
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_east[:,:,p] = support_values

    # Data of panel 5 from west
    support_values = Q[:,j0:j0+ngr, west] # Panel 3
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=0)
    halodata_west[:,:,p] = support_values

    # Data of panel 5 from north
    halodata_north[:,:,p] = Q[:,j0:j0+ngr, north] # Panel 0

    # Data of panel 5 at south
    support_values = Q[:,j0:j0+ngr, south] # Panel 2
    support_values = np.flip(support_values,axis=0)
    support_values = np.flip(support_values,axis=1)
    halodata_south[:,:,p] = support_values

    halodata = halodata_east, halodata_west, halodata_north, halodata_south
    return halodata

######################################################################
# This routine get the halo data of Q needed for interpolation
# only at adjacent panels from south and north directions
######################################################################
def get_halo_data_interpolation_NS(Qx, Qy, cs_grid):
    N   = cs_grid.N
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Interior cells index
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Halo data
    halodata_north = np.zeros((N+ng, ngl, nbfaces))
    halodata_south = np.zeros((N+ng, ngl, nbfaces))

    # --------------------- Panel 0 ----------------------------
    p = 0
    north = 4
    south = 5

    # Data of panel 0 from north
    halodata_north[:,:,p] = Qx[:,j0:j0+ngr,north] # Panel 4

    # Data of panel 0 from south
    halodata_south[:,:,p] = Qx[:,jend-ngl:jend,south] # Panel 5

    # --------------------- Panel 1 ----------------------------
    p = 1
    north = 4
    south = 5

    # Data of panel 1 from north
    support_values = Qy[iend-ngr:iend,:,north] # Panel 4
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_north[:,:,p] = support_values

    # Data of panel 1 from south
    support_values = Qy[iend-ngl:iend,:,south] # Panel 5
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=0)
    halodata_south[:,:,p] = support_values

    # --------------------- Panel 2 ----------------------------
    p = 2
    north = 4
    south = 5

    # Data of panel 2 from north
    support_values = Qx[:,jend-ngr:jend,north] # Panel 4
    support_values = np.flip(support_values,axis=0)
    support_values = np.flip(support_values,axis=1)
    halodata_north[:,:,p] = support_values

    # Data of panel 2 from south
    support_values = Qx[:,j0:j0+ngr, south] # Panel 5
    support_values = np.flip(support_values,axis=1)
    support_values = np.flip(support_values,axis=0)
    halodata_south[:,:,p] = support_values

    # --------------------- Panel 3 ----------------------------
    p = 3
    north = 4
    south = 5

    # Data of panel 3 from north
    support_values = Qy[i0:i0+ngr,:,north] # Panel 4
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=0)
    halodata_north[:,:,p] = support_values

    # Data of panel 0 from south
    support_values = Qy[i0:i0+ngr,:,south] # Panel 5
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_south[:,:,p] = support_values

    # --------------------- Panel 4 ----------------------------
    p = 4
    north = 2
    south = 0

    # Data of panel 4 from north
    support_values = Qx[:,jend-ngr:jend,north] # Panel 2
    support_values = np.flip(support_values,axis=0)
    support_values = np.flip(support_values,axis=1)
    halodata_north[:,:,p] = support_values

    # Interpolate ghost cells of panel 4 at south
    halodata_south[:,:,p] = Qx[:,jend-ngl:jend,south] # Panel 0


    # --------------------- Panel 5 ----------------------------
    p = 5
    north = 0
    south = 2

    # Data of panel 5 from north
    halodata_north[:,:,p] = Qx[:,j0:j0+ngr,north] # Panel 0

    # Data of panel 5 at south
    support_values = Qx[:,j0:j0+ngr, south] # Panel 2
    support_values = np.flip(support_values,axis=0)
    support_values = np.flip(support_values,axis=1)
    halodata_south[:,:,p] = support_values

    halodata = halodata_north, halodata_south
    return halodata

######################################################################
# This routine get the halo data of Q needed for interpolation
# only at adjacent panels from east and west directions
######################################################################
def get_halo_data_interpolation_WE(Qx, Qy, cs_grid):
    N   = cs_grid.N
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Interior cells index
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Halo data
    halodata_east  = np.zeros((ngl, N+ng, nbfaces))
    halodata_west  = np.zeros((ngl, N+ng, nbfaces))

    # --------------------- Panel 0 ----------------------------
    p = 0
    east  = 1
    west  = 3

    # Data of panel 0 from east
    halodata_east[:,:,p] = Qy[i0:i0+ngr,:,east] # Panel 1

    # Data of panel 0 from  west
    halodata_west[:,:,p] = Qy[iend-ngl:iend,:,west] # Panel 3

    # --------------------- Panel 1 ----------------------------
    p = 1
    east  = 2
    west  = 0

    # Data of panel 1 from east
    halodata_east[:,:,p] = Qy[i0:i0+ngr,:,east] # Panel 2

    # Data of panel 1 from west
    halodata_west[:,:,p] = Qy[iend-ngl:iend,:,west] # Panel 0

    # --------------------- Panel 2 ----------------------------
    p = 2
    east  = 3
    west  = 1

    # Data of panel 2 from east
    halodata_east[:,:,p] = Qy[i0:i0+ngr,:,east] # Panel 3

    # Data of panel 2 from west
    halodata_west[:,:,p] = Qy[iend-ngl:iend,:,west] # Panel 1

    # --------------------- Panel 3 ----------------------------
    p = 3
    east  = 0
    west  = 2

    # Data of panel 3 from east
    halodata_east[:,:,p] = Qy[i0:i0+ngr,:,east] # Panel 0

    # Data of panel 3 from west
    halodata_west[:,:,p] = Qy[iend-ngl:iend,:,west] # Panel 2

    # --------------------- Panel 4 ----------------------------
    p = 4
    east  = 1
    west  = 3

    # Data of panel 4 from east
    support_values = Qx[:,jend-ngl:jend,east] # Panel 3
    support_values = np.flip(support_values,axis=1)
    support_values = np.transpose(support_values)
    halodata_east[:,:,p] = support_values

    # Data of panel 4 from west
    support_values = Qx[:,jend-ngl:jend,west] # Panel 1
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_west[:,:,p] = support_values

    # --------------------- Panel 5 ----------------------------
    p = 5
    east  = 1
    west  = 3

    # Data of panel 5 from east
    support_values = Qx[:,j0:j0+ngr,east] # Panel 1
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=1)
    halodata_east[:,:,p] = support_values

    # Data of panel 5 from west
    support_values = Qx[:,j0:j0+ngr,west] # Panel 3
    support_values = np.transpose(support_values)
    support_values = np.flip(support_values,axis=0)
    halodata_west[:,:,p] = support_values

    halodata = halodata_east, halodata_west
    return halodata
