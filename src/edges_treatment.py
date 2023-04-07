#######################################################################
#
# This module is dedicated to routines related to the data near
# the edges of the cubed-sphere grid
#
# Luan Santos - 2023
####################################################################

import numpy as np
import numexpr as ne
from interpolation import ghost_cells_adjacent_panels, ghost_cells_lagrange_interpolation, ghost_cells_lagrange_interpolation_NS, ghost_cells_lagrange_interpolation_WE

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
def average_parabola_cube_edges(Qx, Qy, px, py, cs_grid):
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

    # Average panels 1-4
    px.q_R[iend-1,j0:jend,4] = (px.q_R[iend-1,j0:jend,4] + py.q_R[i0:iend,jend-1,1])*0.5
    py.q_R[i0:iend,jend-1,1] = px.q_R[iend-1,j0:jend,4]

    # Average panels 2-4
    py.q_R[i0:iend,jend-1,4] = (py.q_R[i0:iend,jend-1,4] + np.flip(py.q_R[i0:iend,jend-1,2]))*0.5
    py.q_R[i0:iend,jend-1,2] = np.flip(py.q_R[i0:iend,jend-1,4])

    # Average panels 3-4
    px.q_L[i0,j0:jend,4] = np.flip(py.q_R[i0:iend,jend-1,3])
    py.q_R[i0:iend,jend-1,3] = np.flip(px.q_L[i0,j0:jend,4])

    # Average panels 0-5
    py.q_R[i0:iend,jend-1,5] = (py.q_R[i0:iend,jend-1,5] + py.q_L[i0:iend,j0,0])*0.5
    py.q_L[i0:iend,j0,0] = py.q_R[i0:iend,jend-1,5]

    # Average panels 1-5
    py.q_L[i0:iend,j0,1] = (py.q_L[i0:iend,j0,1] + np.flip(px.q_R[iend-1,j0:jend,5]))*0.5
    px.q_R[iend-1,j0:jend,5] = np.flip(py.q_L[i0:iend,j0,1])

    # Average panels 2-5
    py.q_L[i0:iend,j0,2] = (py.q_L[i0:iend,j0,2] + np.flip(py.q_L[i0:iend,j0,5]))*0.5
    py.q_L[i0:iend,j0,5] = np.flip(py.q_L[i0:iend,j0,2])

    # Average panels 3-5
    py.q_L[i0:iend,j0,3] = (px.q_L[i0,j0:jend,5] + py.q_L[i0:iend,j0,3])*0.5
    px.q_L[i0,j0:jend,5] = py.q_L[i0:iend,j0,3]

    # Update coeffs
    # x direction
    q_L =  px.q_L[i0-1:iend+1,:,:]
    q_R =  px.q_R[i0-1:iend+1,:,:]
    q = Qx[i0-1:iend+1,:,:]
    px.dq[i0-1:iend+1,:,:] = ne.evaluate('q_R-q_L')
    px.q6[i0-1:iend+1,:,:] = ne.evaluate('6*q- 3*(q_R + q_L)')
    # y direction
    q_L =  py.q_L[:,j0-1:jend+1,:]
    q_R =  py.q_R[:,j0-1:jend+1,:]
    q = Qy[:,j0-1:jend+1,:]
    py.dq[:,j0-1:jend+1,:] = ne.evaluate('q_R-q_L')
    py.q6[:,j0-1:jend+1,:] = ne.evaluate('6*q- 3*(q_R + q_L)')

####################################################################################
# Perform the extrapolation from Putman and Lin 07 paper (PL07)
# described in appendix C from this paper
####################################################################################
def edges_extrapolation(Qx, Qy, px, py, cs_grid, simulation):
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # Formula 47 from PL07
    # x direction
    px.q_L[i0,j0:jend,0:6] = 1.5*Qx[i0,j0:jend,0:6] - 0.5*Qx[i0+1,j0:jend,0:6]
    px.q_R[iend-1,j0:jend,0:6] = 1.5*Qx[iend-1,j0:jend,0:6] - 0.5*Qx[iend-2,j0:jend,0:6]

    # y direction
    py.q_L[i0:iend,j0,0:6] = 1.5*Qy[i0:iend,j0,0:6] - 0.5*Qy[i0:iend,j0+1,0:6]
    py.q_R[i0:iend,jend-1,0:6] = 1.5*Qy[i0:iend,jend-1,0:6] - 0.5*Qy[i0:iend,jend-2,0:6]

   # Formula 49 from PL07
    # x direction
    px.q_R[i0,j0:jend,0:6] = (3.0*Qx[i0,j0:jend,0:6] + 11.0*Qx[i0+1,j0:jend,0:6] - 2.0*(Qx[i0+2,j0:jend,0:6] - Qx[i0,j0:jend,0:6]))/14.0
    px.q_L[i0+1,j0:jend,0:6] = px.q_R[i0,j0:jend,0:6]
    px.q_L[iend-1,j0:jend,0:6] = (3.0*Qx[iend-1,j0:jend,0:6] + 11.0*Qx[iend-2,j0:jend,0:6] - 2.0*(Qx[iend-3,j0:jend,0:6] - Qx[iend-1,j0:jend,0:6]))/14.0
    px.q_R[iend-2,j0:jend,0:6] = px.q_L[iend-1,j0:jend,0:6]

    # y direction
    py.q_R[i0:iend,j0,0:6] = (3.0*Qy[i0:iend,j0,0:6] + 11.0*Qy[i0:iend,j0+1,0:6] - 2.0*(Qy[i0:iend,j0+2,0:6] - Qy[i0:iend,j0,0:6]))/14.0
    py.q_L[i0:iend,j0+1,0:6] = py.q_R[i0:iend,j0,0:6]
    py.q_L[i0:iend,jend-1,0:6] = (3.0*Qy[i0:iend,jend-1,0:6] + 11.0*Qy[i0:iend,jend-2,0:6] - 2.0*(Qy[i0:iend,jend-3,0:6] - Qy[i0:iend,jend-1,0:6]))/14.0
    py.q_R[i0:iend,jend-2,0:6] = py.q_L[i0:iend,jend-1,0:6]

    if simulation.recon_name=='PPM-PL07':
        px.q_R[i0+1,j0:jend,:] = px.q_L[i0+2,j0:jend,:]
        px.q_L[iend-2,j0:jend,:] = px.q_R[iend-3,j0:jend,:]
        py.q_R[i0:iend,j0+1,:] = py.q_L[i0:iend,j0+2,:]
        py.q_L[i0:iend,jend-2,:] = py.q_R[i0:iend,jend-3,:]

    average_parabola_cube_edges(Qx, Qy, px, py, cs_grid)

    # Ghost cell updates
    # 0-1; 1-2; 2-3
    px.q_L[iend,j0:jend,0:3] = px.q_L[i0,j0:jend,1:4]
    px.q_R[iend,j0:jend,0:3] = px.q_R[i0,j0:jend,1:4]
    px.q_L[i0-1,j0:jend,1:4] = px.q_L[iend-1,j0:jend,0:3]
    px.q_R[i0-1,j0:jend,1:4] = px.q_R[iend-1,j0:jend,0:3]

    # 3-0
    px.q_L[iend,j0:jend,3] = px.q_L[i0,j0:jend,0]
    px.q_R[iend,j0:jend,3] = px.q_R[i0,j0:jend,0]
    px.q_L[i0-1,j0:jend,0] = px.q_L[iend-1,j0:jend,3]
    px.q_R[i0-1,j0:jend,0] = px.q_R[iend-1,j0:jend,3]

    # 0-4
    py.q_L[i0:iend,jend,0] = py.q_L[i0:iend,j0,4]
    py.q_R[i0:iend,jend,0] = py.q_R[i0:iend,j0,4]
    py.q_L[i0:iend,j0-1,4] = py.q_L[i0:iend,jend-1,0]
    py.q_R[i0:iend,j0-1,4] = py.q_R[i0:iend,jend-1,0]

    # 1-4
    py.q_L[i0:iend,jend,1] = px.q_R[iend-1,j0:jend,4]
    py.q_R[i0:iend,jend,1] = px.q_L[iend-1,j0:jend,4]
    px.q_R[iend,j0:jend,4] = py.q_L[i0:iend,jend-1,1]
    px.q_L[iend,j0:jend,4] = py.q_R[i0:iend,jend-1,1]

    # 2-4
    py.q_L[i0:iend,jend,2] = np.flip(py.q_R[i0:iend,jend-1,4])
    py.q_R[i0:iend,jend,2] = np.flip(py.q_L[i0:iend,jend-1,4])
    py.q_R[i0:iend,jend,4] = np.flip(py.q_L[i0:iend,jend-1,2])
    py.q_L[i0:iend,jend,4] = np.flip(py.q_R[i0:iend,jend-1,2])

    # 3-4
    py.q_L[i0:iend,jend,3] = np.flip(px.q_L[i0,j0:jend,4])
    py.q_R[i0:iend,jend,3] = np.flip(px.q_R[i0,j0:jend,4])
    px.q_L[i0-1,j0:jend,4] = np.flip(py.q_L[i0:iend,jend-1,3])
    px.q_R[i0-1,j0:jend,4] = np.flip(py.q_R[i0:iend,jend-1,3])

    # 0-5
    py.q_L[i0:iend,jend,5] = py.q_L[i0:iend,j0,0]
    py.q_R[i0:iend,jend,5] = py.q_R[i0:iend,j0,0]
    py.q_L[i0:iend,j0-1,0] = py.q_L[i0:iend,jend-1,5]
    py.q_R[i0:iend,j0-1,0] = py.q_R[i0:iend,jend-1,5]

    # 1-5
    px.q_L[iend,j0:jend,5] = np.flip(py.q_L[i0:iend,j0,1])
    px.q_R[iend,j0:jend,5] = np.flip(py.q_R[i0:iend,j0,1])
    py.q_L[i0:iend,j0-1,1] = np.flip(px.q_L[iend-1,j0:jend,5])
    py.q_R[i0:iend,j0-1,1] = np.flip(px.q_R[iend-1,j0:jend,5])

    # 2-5
    py.q_L[i0:iend,j0-1,5] = np.flip(py.q_R[i0:iend,j0,2])
    py.q_R[i0:iend,j0-1,5] = np.flip(py.q_L[i0:iend,j0,2])
    py.q_L[i0:iend,j0-1,2] = np.flip(py.q_R[i0:iend,j0,5])
    py.q_R[i0:iend,j0-1,2] = np.flip(py.q_L[i0:iend,j0,5])

    # 3-5
    px.q_L[i0-1,j0:jend,5] = py.q_R[i0:iend,j0,3]
    px.q_R[i0-1,j0:jend,5] = py.q_L[i0:iend,j0,3]
    py.q_R[i0:iend,j0-1,3] = px.q_L[i0,j0:jend,5]
    py.q_L[i0:iend,j0-1,3] = px.q_R[i0,j0:jend,5]

    # Update coeffs
    # x direction
    q_L =  px.q_L[i0-1:iend+1,:,:]
    q_R =  px.q_R[i0-1:iend+1,:,:]
    q = Qx[i0-1:iend+1,:,:]
    px.dq[i0-1:iend+1,:,:] = ne.evaluate('q_R-q_L')
    px.q6[i0-1:iend+1,:,:] = ne.evaluate('6*q- 3*(q_R + q_L)')

    # y direction
    q_L =  py.q_L[:,j0-1:jend+1,:]
    q_R =  py.q_R[:,j0-1:jend+1,:]
    q = Qy[:,j0-1:jend+1,:]
    py.dq[:,j0-1:jend+1,:] = ne.evaluate('q_R-q_L')
    py.q6[:,j0-1:jend+1,:] = ne.evaluate('6*q- 3*(q_R + q_L)')


####################################################################################
#  Given the PPM parabolas px and py, this routine
#  average the values of the upwind flux at the
#  cube edge points
####################################################################################
def average_flux_cube_edges(px, py, cs_grid):
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # Average panels 0-1,1-2,2-3,3-4
    px.f_upw[iend,j0:jend,0:3] = (px.f_upw[iend,j0:jend,0:3] + px.f_upw[i0,j0:jend,1:4])*0.5
    px.f_upw[i0,j0:jend,1:4] = px.f_upw[iend,j0:jend,0:3]
    px.f_upw[iend,j0:jend,3] = (px.f_upw[iend,j0:jend,3] + px.f_upw[i0,j0:jend,0])*0.5
    px.f_upw[i0,j0:jend,0] = px.f_upw[iend,j0:jend,3]

    # Average panels 0-4
    py.f_upw[i0:iend,j0,4] = (py.f_upw[i0:iend,j0,4] + py.f_upw[i0:iend,jend,0])*0.5
    py.f_upw[i0:iend,jend,0] = py.f_upw[i0:iend,j0,4]

    # Average panels 1-4
    px.f_upw[iend,j0:jend,4] = (px.f_upw[iend,j0:jend,4] + py.f_upw[i0:iend,jend,1])*0.5
    py.f_upw[i0:iend,jend,1] = px.f_upw[iend,j0:jend,4]

    # Average panels 2-4
    py.f_upw[i0:iend,jend,4] = (py.f_upw[i0:iend,jend,4] + np.flip(py.f_upw[i0:iend,jend-1,2]))*0.5
    py.f_upw[i0:iend,jend,2] = np.flip(py.f_upw[i0:iend,jend,4])

    # Average panels 3-4
    px.f_upw[i0,j0:jend,4] = (px.f_upw[i0,j0:jend,4] + np.flip(py.f_upw[i0:iend,jend,3]))*0.5
    py.f_upw[i0:iend,jend,3] = np.flip(px.f_upw[i0,j0:jend,4])

    # Average panels 0-5
    py.f_upw[i0:iend,jend,5] = (py.f_upw[i0:iend,jend,5] + py.f_upw[i0:iend,j0,0])*0.5
    py.f_upw[i0:iend,j0,0] = py.f_upw[i0:iend,jend,5]

    # Average panels 1-5
    py.f_upw[i0:iend,j0,1] = (py.f_upw[i0:iend,j0,1] + np.flip(px.f_upw[iend,j0:jend,5]))*0.5
    px.f_upw[iend,j0:jend,5] = np.flip(py.f_upw[i0:iend,j0,1])

    # Average panels 2-5
    py.f_upw[i0:iend,j0,2] = (py.f_upw[i0:iend,j0,2] + np.flip(py.f_upw[i0:iend,j0,5]))*0.5
    py.f_upw[i0:iend,j0,5] = np.flip(py.f_upw[i0:iend,j0,2])

    # Average panels 3-5
    py.f_upw[i0:iend,j0,3] = (px.f_upw[i0,j0:jend,5] + py.f_upw[i0:iend,j0,3])*0.5
    px.f_upw[i0,j0:jend,5] = py.f_upw[i0:iend,j0,3]

####################################################################################
# This routine fill the halo data using the scheme given in simulation
# Qx and Qy are scalar fields
####################################################################################
def edges_ghost_cell_treatment_scalar(Qx, Qy, cs_grid, simulation, transformation, lagrange_poly, Kmin, Kmax):
    if simulation.et_name=='ET-S72' or simulationet_name=='ET-PL07': # Uses adjacent cells values
        ghost_cells_adjacent_panels(Qx, Qy, cs_grid, simulation)

    elif simulation.et_name=='ET-R96' or simulation.et_name=='ET-R96-AF': # Uses ghost cells interpolation - ignoring corner ghost cells
        # Interpolate to ghost cells - north and south neighbors
        ghost_cells_lagrange_interpolation_NS(Qx, Qy, cs_grid, transformation, simulation, lagrange_poly, Kmin, Kmax)

        # Interpolate to ghost cells - west and east neighbors
        ghost_cells_lagrange_interpolation_WE(Qx, Qy, cs_grid, transformation, simulation, lagrange_poly, Kmin, Kmax)

    elif simulation.et_name=='ET-Z21' or simulation.et_name=='ET-Z21-AF': # Uses ghost cells interpolation - using corner ghost cells
        ghost_cells_lagrange_interpolation(Qx, Qy, cs_grid, transformation, simulation,\
                                           lagrange_poly, Kmin, Kmax)

####################################################################################
# This routine fill the halo data with the scheme given in simulation
# u and v are components of the velocity fields at edges
####################################################################################
def edges_ghost_cell_treatment_vector(u, v, cs_grid, simulation):
    if simulation.dp_name == 'RK2':
        i0, iend = cs_grid.i0, cs_grid.iend
        j0, jend = cs_grid.j0, cs_grid.jend
        ngl = cs_grid.nghost_left

        if simulation.et_name=='ET-S72' or simulation.et_name=='ET-PL07': # Uses adjacent cells values
            # Panels 0-1,1-2,2-3,3-4
            u[iend+1,j0:jend,0:3] = u[i0+1,j0:jend,1:4]
            u[iend+1,j0:jend,3]   = u[i0+1,j0:jend,0]
            u[i0-1,j0:jend,1:4] = u[iend-1,j0:jend,0:3]
            u[i0-1,j0:jend,0]   = u[iend-1,j0:jend,3]

            # Panels 0-4
            v[i0:iend,jend+1,0] = v[i0:iend,j0+1,4]
            v[i0:iend,j0-1,4]   = v[i0:iend,jend-1,0]

            # Panels 1-4
            u[iend+1,j0:jend,4] = -v[i0:iend,jend-1,1]
            v[i0:iend,jend+1,1] = -u[iend-1,j0:jend,4]

            # Panels 2-4
            v[i0:iend,jend+1,4] = -np.flip(v[i0:iend,jend-1,2])
            v[i0:iend,jend+1,2] = -np.flip(v[i0:iend,jend-1,4])

            # Panels 3-4
            u[i0-1,j0:jend,4]   = np.flip(v[i0:iend,jend-1,3])
            v[i0:iend,jend+1,3] = np.flip(u[i0+1,j0:jend,4])

            # Panels 0-5
            v[i0:iend,jend+1,5] = v[i0:iend,j0+1,0]
            v[i0:iend,j0-1,0]   = v[i0:iend,jend-1,5]

            # Panels 1-5
            v[i0:iend,j0-1,1]   = np.flip(u[iend-1,j0:jend,5])
            u[iend+1,j0:jend,5] = np.flip(v[i0:iend,j0+1,1])

            # Panels 2-5
            v[i0:iend,j0-1,2] = -np.flip(v[i0:iend,j0+1,5])
            v[i0:iend,j0-1,5] = -np.flip(v[i0:iend,j0+1,2])

            # Average panels 3-5
            v[i0:iend,j0-1,3] = -u[i0+1,j0:jend,5]
            u[i0-1,j0:jend,5] = -v[i0:iend,j0+1,3]
