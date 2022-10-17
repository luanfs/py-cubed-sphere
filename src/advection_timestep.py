####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from dimension_splitting    import F_operator, G_operator, fix_splitting_operator_ghost_cells
from interpolation          import ghost_cells_lagrange_interpolation
from flux                   import compute_fluxes, fix_fluxes_at_cube_edges, average_fluxes_at_cube_edges


####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, g_metric, Q_old, Q_new, k, dt, t, ax, ay, cx, cx2, cy, cy2, \
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

    # Compute the fluxes
    compute_fluxes(gQ, gQ, ucontra_edx, vcontra_edy, flux_x, flux_y, ax, ay, cs_grid, simulation)

    # Applies F and G operators in each panel
    F_operator(F_gQ, gQ, ucontra_edx, flux_x, ax, cs_grid, simulation)
    G_operator(G_gQ, gQ, vcontra_edy, flux_y, ay, cs_grid, simulation)

    # Compute the fluxes
    compute_fluxes(gQ+0.5*G_gQ, gQ+0.5*F_gQ, ucontra_edx, vcontra_edy, flux_x, flux_y, ax, ay, cs_grid, simulation)

    # Applies F and G operators in each panel again
    F_operator(FG_gQ, gQ+0.5*G_gQ, ucontra_edx, flux_x, ax, cs_grid, simulation)
    G_operator(GF_gQ, gQ+0.5*F_gQ, vcontra_edy, flux_y, ay, cs_grid, simulation)

    # Exact operators
    #N = cs_grid.N
    #ng = cs_grid.nghost
    #F_gQ_exact  = np.zeros((N+6,N+6,6))
    #G_gQ_exact  = np.zeros((N+6,N+6,6))
    #FG_gQ_exact = np.zeros((N+6,N+6,6))

    #F_gQ_exact[:,:,:] = (cs_grid.dy/cs_grid.areas[:,:,:])*(ucontra_edx[1:N+ng+1,:,:]*cs_grid.metric_tensor_edx[1:N+ng+1,:,:] - ucontra_edx[0:N+ng,:,:]*cs_grid.metric_tensor_edx[0:N+ng,:,:])
    #G_gQ_exact[:,:,:] = (cs_grid.dx/cs_grid.areas[:,:,:])*(vcontra_edy[:,1:N+ng+1,:]*cs_grid.metric_tensor_edy[:,1:N+ng+1,:] - vcontra_edy[:,0:N+ng,:]*cs_grid.metric_tensor_edy[:,0:N+ng,:])
    #FG_gQ_exact = G_gQ_exact + F_gQ_exact

    Q_new[i0:iend,j0:jend,:] = Q_old[i0:iend,j0:jend,:] + FG_gQ[i0:iend,j0:jend,:] + GF_gQ[i0:iend,j0:jend,:]
    #Q_new[i0:iend,j0:jend,:] = Q_old[i0:iend,j0:jend,:] + FG_gQ_exact[i0:iend,j0:jend,:]
    # Update
    Q_old = Q_new
