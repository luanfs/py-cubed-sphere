####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from dimension_splitting    import F_operator, G_operator
from interpolation          import ghost_cells_interpolation

####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, g_metric, Q_old, Q_new, k, dt, t, ax, ay, cx, cx2, cy, cy2, \
                  flux_x, flux_y, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
                  F_gQ, G_gQ, GF_gQ, FG_gQ):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Interpolate to ghost cells
    ghost_cells_interpolation(Q_old, cs_grid, t, simulation)

    # Multiply the field Q by metric tensor
    gQ = Q_old*g_metric

    # Applies F and G operators in each panel
    F_operator(F_gQ, gQ, ucontra_edx, flux_x, ax, cs_grid, simulation)
    G_operator(G_gQ, gQ, vcontra_edy, flux_y, ay, cs_grid, simulation)

    # Applies F and G operators in each panel again
    F_operator(FG_gQ, gQ+0.5*G_gQ, ucontra_edx, flux_x, ax, cs_grid, simulation)
    G_operator(GF_gQ, gQ+0.5*F_gQ, vcontra_edy, flux_y, ay, cs_grid, simulation)

    Q_new[i0:iend,j0:jend,:] = Q_old[i0:iend,j0:jend,:] + FG_gQ[i0:iend,j0:jend,:] + GF_gQ[i0:iend,j0:jend,:]

    # Update
    Q_old = Q_new
