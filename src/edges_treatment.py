#######################################################################
#
# This module is dedicated to routines related to the data near
# the edges of the cubed-sphere grid
#
# Luan Santos - 2023
####################################################################

import numpy as np

####################################################################################
#  The quadrilateral points are labeled as below
#
#  po-------pv--------po
#  |                  |
#  |                  |
#  |                  |
#  pu       pc        pu
#  |                  |
#  |                  |
#  |                  |
#  po--------pv-------po
#
#  Given the PPM parabolas px and py, this routine
#  average the values of the edge reconstruction at the
#  cube edge points
####################################################################################
def average_parabola_cube_edges(px, py, cs_grid):
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # Average panels 0-1,1-2,2-3,3-4
    px.q_R[iend-1,j0:jend,0:3] = (px.q_R[iend-1,j0:jend,0:3] + px.q_L[i0,j0:jend,1:4])*0.5
    px.q_L[i0,j0:jend,1:4] = px.q_R[iend-1,j0:jend,0:3]
    px.q_R[iend-1,j0:jend,3] = (px.q_R[iend-1,j0:jend,3] + px.q_L[i0,j0:jend,0])*0.5
    px.q_L[i0,j0:jend,0] = px.q_R[iend-1,j0:jend,3]

    # Average panels 0-4
    py.q_L[i0:iend,j0,4] = (py.q_L[i0:iend,j0,4] + py.q_R[i0:iend,jend-1,0])*0.5
    py.q_R[i0:iend,jend-1,0] = py.q_L[i0:iend,j0,4]

    # Average panels 0-5
    py.q_R[i0:iend,jend-1,5] = (py.q_R[i0:iend,jend-1,5] + py.q_L[i0:iend,j0,0])*0.5
    py.q_L[i0:iend,j0,0] = py.q_R[i0:iend,jend-1,5]

    # Average panels 1-4
    px.q_R[iend-1,j0:jend,4] = (px.q_R[iend-1,j0:jend,4] + py.q_R[i0:iend,jend-1,1])*0.5
    py.q_R[i0:iend,jend-1,1] = px.q_R[iend-1,j0:jend,4]

    # Average panels 1-5
    py.q_L[i0:iend,j0,1] = (py.q_L[i0:iend,j0,1] + np.flip(px.q_R[iend-1,j0:jend,5]))*0.5
    px.q_R[iend-1,j0:jend,5] = np.flip(py.q_L[i0:iend,j0,1])

    # Average panels 2-4
    py.q_R[i0:iend,jend-1,4] = (py.q_R[i0:iend,jend-1,4] + np.flip(py.q_R[i0:iend,jend-1,2]))*0.5
    py.q_R[i0:iend,jend-1,2] = np.flip(py.q_R[i0:iend,jend-1,4])

    # Average panels 2-5
    py.q_L[i0:iend,j0,2] = (py.q_L[i0:iend,j0,2] + np.flip(py.q_L[i0:iend,j0,5]))*0.5
    py.q_L[i0:iend,j0,5] = np.flip(py.q_L[i0:iend,j0,2])

    # Average panels 3-4
    px.q_L[i0,j0:jend,4] = np.flip(py.q_R[i0:iend,jend-1,3])
    py.q_R[i0:iend,jend-1,3] = np.flip(px.q_L[i0,j0:jend,4])

    # Average panels 3-5
    py.q_L[i0:iend,j0,3] = (px.q_L[i0,j0:jend,5] + py.q_L[i0:iend,j0,3])*0.5
    px.q_L[i0,j0:jend,5] = py.q_L[i0:iend,j0,3]


####################################################################################
# Perform the extrapolation from Putman and Lin 07 paper (PL07)
# described in appendix C from this paper
####################################################################################
def edges_extrapolation(Q, px, py, cs_grid, simulation):
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # Formula 47 from PL07
    # x direction
    px.q_L[i0,j0:jend,0:6] = 1.5*Q[i0,j0:jend,0:6] - 0.5*Q[i0+1,j0:jend,0:6]
    px.q_R[iend-1,j0:jend,0:6] = 1.5*Q[iend-1,j0:jend,0:6] - 0.5*Q[iend-2,j0:jend,0:6]

    # y direction
    py.q_L[i0:iend,j0,0:6] = 1.5*Q[i0:iend,j0,0:6] - 0.5*Q[i0:iend,j0+1,0:6]
    py.q_R[i0:iend,jend-1,0:6] = 1.5*Q[i0:iend,jend-1,0:6] - 0.5*Q[i0:iend,jend-2,0:6]

    # Formula 49 from PL07
    # x direction
    px.q_R[i0,j0:jend,0:6] = (3.0*Q[i0,j0:jend,0:6] + 11.0*Q[i0+1,j0:jend,0:6] - 2.0*(Q[i0+2,j0:jend,0:6] - Q[i0,j0:jend,0:6]))/14.0
    px.q_L[i0+1,j0:jend,0:6] = px.q_R[i0,j0:jend,0:6]
    px.q_L[iend-1,j0:jend,0:6] = (3.0*Q[iend-1,j0:jend,0:6] + 11.0*Q[iend-2,j0:jend,0:6] - 2.0*(Q[iend-3,j0:jend,0:6] - Q[iend-1,j0:jend,0:6]))/14.0
    px.q_R[iend-2,j0:jend,0:6] = px.q_L[iend-1,j0:jend,0:6]

    # y direction
    py.q_R[i0:iend,j0,0:6] = (3.0*Q[i0:iend,j0,0:6] + 11.0*Q[i0:iend,j0+1,0:6] - 2.0*(Q[i0:iend,j0+2,0:6] - Q[i0:iend,j0,0:6]))/14.0
    py.q_L[i0:iend,j0+1,0:6] = py.q_R[i0:iend,j0,0:6]
    py.q_L[i0:iend,jend-1,0:6] = (3.0*Q[i0:iend,jend-1,0:6] + 11.0*Q[i0:iend,jend-2,0:6] - 2.0*(Q[i0:iend,jend-3,0:6] - Q[i0:iend,jend-1,0:6]))/14.0
    py.q_R[i0:iend,jend-2,0:6] = py.q_L[i0:iend,jend-1,0:6]

    if simulation.recon_name=='PPM-PL07':
        px.q_R[i0+1,j0:jend,:] = px.q_L[i0+2,j0:jend,:]
        px.q_L[iend-2,j0:jend,:] = px.q_R[iend-3,j0:jend,:]
        py.q_R[i0:iend,j0+1,:] = py.q_L[i0:iend,j0+2,:]
        py.q_L[i0:iend,jend-2,:] = py.q_R[i0:iend,jend-3,:]
