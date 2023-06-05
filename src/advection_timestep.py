####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from flux                   import compute_fluxes
from advection_ic           import velocity_adv
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from discrete_operators     import divergence
from averaged_velocity      import time_averaged_velocity

####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, Q, gQ, div, px, py, cx, cy, \
                  U_pu, U_pv, transformation, lagrange_poly_ghost_pc, stencil_ghost_pc, k, t,):

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Compute the velocity need for the departure point
    time_averaged_velocity(U_pu, U_pv, cs_grid, simulation)

    # Compute the divergence
    divergence(Q, gQ, div, U_pu, U_pv, px, py, cx, cy, cs_grid, simulation,\
               transformation, lagrange_poly_ghost_pc, stencil_ghost_pc)

    # Q update
    Q[i0:iend,j0:jend,:] = Q[i0:iend,j0:jend,:] - simulation.dt*div[i0:iend,j0:jend,:]

####################################################################################
# This routine computes advection updates for the next timestep
####################################################################################
def update_adv(cs_grid, simulation, t, cx, cy, U_pu, U_pv):
    # Updates for next time step - only for time dependent velocity
    if simulation.vf >= 3:
        # Velocity
        U_pu.ulon[:,:,:], U_pu.vlat[:,:,:] = velocity_adv(cs_grid.pu.lon,cs_grid.pu.lat, t, simulation)
        U_pv.ulon[:,:,:], U_pv.vlat[:,:,:] = velocity_adv(cs_grid.pv.lon,cs_grid.pv.lat, t, simulation)

        # Store old velocity
        U_pu.ucontra_old[:,:,:] = U_pu.ucontra[:,:,:]
        U_pv.vcontra_old[:,:,:] = U_pv.vcontra[:,:,:]

        # Latlon to contravariant
        U_pu.ucontra[:,:,:], U_pu.vcontra[:,:,:] = latlon_to_contravariant(U_pu.ulon, U_pu.vlat, cs_grid.prod_ex_elon_pu, cs_grid.prod_ex_elat_pu, cs_grid.prod_ey_elon_pu, cs_grid.prod_ey_elat_pu, cs_grid.determinant_ll2contra_pu)
        U_pv.ucontra[:,:,:], U_pv.vcontra[:,:,:] = latlon_to_contravariant(U_pv.ulon, U_pv.vlat, cs_grid.prod_ex_elon_pv, cs_grid.prod_ex_elat_pv, cs_grid.prod_ey_elon_pv, cs_grid.prod_ey_elat_pv, cs_grid.determinant_ll2contra_pv)


