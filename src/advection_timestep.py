####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from interpolation          import ghost_cells_lagrange_interpolation
from flux                   import compute_fluxes, fix_fluxes_at_cube_edges, average_fluxes_at_cube_edges


####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, g_metric, Q_old, Q_new, k, dt, t, cx, cy, \
                  flux_x, flux_y, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
                  F_gQ, G_gQ, GF_gQ, FG_gQ, transformation,\
                  lagrange_poly, Kmin, Kmax):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Interpolate on ghost cells
    ghost_cells_lagrange_interpolation(Q_old, cs_grid, transformation, simulation,\
                                      lagrange_poly, Kmin, Kmax)

    # Multiply the field Q by metric tensor
    gQ = Q_old*g_metric

    Q_new[i0:iend,j0:jend,:] = Q_old[i0:iend,j0:jend,:] + FG_gQ[i0:iend,j0:jend,:] + GF_gQ[i0:iend,j0:jend,:]

    # Update
    Q_old = Q_new

    # Updates for next time step - only for time dependent velocity
    if simulation.vf >= 3:
        # Velocity
        ulon_edx[:,:,:], vlat_edx[:,:,:] = velocity_adv(edx_lon, edx_lat, t, simulation)
        ulon_edy[:,:,:], vlat_edy[:,:,:] = velocity_adv(edy_lon, edy_lat, t, simulation)

        # Latlon to contravariant
        ucontra_edx, vcontra_edx = latlon_to_contravariant(ulon_edx, vlat_edx, ex_elon_edx, ex_elat_edx, ey_elon_edx, ey_elat_edx, det_edx)
        ucontra_edy, vcontra_edy = latlon_to_contravariant(ulon_edy, vlat_edy, ex_elon_edy, ex_elat_edy, ey_elon_edy, ey_elat_edy, det_edy)

        # CFL
        cx[:,:,:] = cfl_x(ucontra_edx, cs_grid, simulation)
        cy[:,:,:] = cfl_y(vcontra_edy, cs_grid, simulation)


