####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from interpolation          import ghost_cells_lagrange_interpolation
from flux                   import compute_fluxes
from advection_ic           import velocity_adv
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from discrete_operators     import divergence
from averaged_velocity      import time_averaged_velocity
from edges_treatment        import edges_ghost_cell_treatment

####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, Q, gQ, div, px, py, cx, cy, \
                  U_pu, U_pv, transformation, lagrange_poly, Kmin, Kmax, k, t,):

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Compute the velocity need for the departure point
    time_averaged_velocity(U_pu, U_pv, k, t, cs_grid, simulation)

    # CFL
    cx[:,:,:] = cfl_x(U_pu.ucontra_averaged[:,:,:], cs_grid, simulation)
    cy[:,:,:] = cfl_y(U_pv.vcontra_averaged[:,:,:], cs_grid, simulation)

    # Multiply the field Q by metric tensor
    #gQ[:,:,:] = Q[:,:,:]*cs_grid.metric_tensor_centers[:,:,:]

    # Compute the divergence
    divergence(Q, gQ, div, px, py, cx, cy, cs_grid, simulation,\
               transformation, lagrange_poly, Kmin, Kmax)
    # Q update
    Q[i0:iend,j0:jend,:] = Q[i0:iend,j0:jend,:] - simulation.dt*div[i0:iend,j0:jend,:]

####################################################################################
# This routine computes advection updates for the next timestep
####################################################################################
def update_adv(cs_grid, simulation, t, cx, cy, U_pu, U_pv):
    # Updates for next time step - only for time dependent velocity
    if simulation.vf >= 3:
        # Velocity
        U_pu.ulon[:,:,:], U_pu.vlat[:,:,:] = velocity_adv(cs_grid.edx.lon,cs_grid.edx.lat, t, simulation)
        U_pv.ulon[:,:,:], U_pv.vlat[:,:,:] = velocity_adv(cs_grid.edy.lon,cs_grid.edy.lat, t, simulation)

        # Store old velocity
        U_pu.ucontra_old[:,:,:] = U_pu.ucontra[:,:,:]
        U_pv.vcontra_old[:,:,:] = U_pv.vcontra[:,:,:]

        # Latlon to contravariant
        U_pu.ucontra[:,:,:], U_pu.vcontra[:,:,:] = latlon_to_contravariant(U_pu.ulon, U_pu.vlat, cs_grid.prod_ex_elon_edx, cs_grid.prod_ex_elat_edx, cs_grid.prod_ey_elon_edx, cs_grid.prod_ey_elat_edx, cs_grid.determinant_ll2contra_edx)
        U_pv.ucontra[:,:,:], U_pv.vcontra[:,:,:] = latlon_to_contravariant(U_pv.ulon, U_pv.vlat, cs_grid.prod_ex_elon_edy, cs_grid.prod_ex_elat_edy, cs_grid.prod_ey_elon_edy, cs_grid.prod_ey_elat_edy, cs_grid.determinant_ll2contra_edy)


