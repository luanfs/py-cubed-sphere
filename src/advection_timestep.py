####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from interpolation          import ghost_cells_lagrange_interpolation
from flux                   import compute_fluxes, fix_fluxes_at_cube_edges, average_fluxes_at_cube_edges
from advection_ic           import velocity_adv
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from discrete_operators     import divergence

####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, Q, gQ, div, px, py, cx, cy, \
                  ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
                  transformation, lagrange_poly, Kmin, Kmax, k, t,):

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Interpolate on ghost cells
    ghost_cells_lagrange_interpolation(Q, cs_grid, transformation, simulation,\
                                      lagrange_poly, Kmin, Kmax)
    # Multiply the field Q by metric tensor
    gQ[:,:,:] = Q[:,:,:]*cs_grid.metric_tensor_centers[:,:,:]

    # Compute the divergence
    divergence(gQ, div, ucontra_edx, vcontra_edy, px, py, cx, cy, cs_grid, simulation)

    # Q update
    Q[i0:iend,j0:jend,:] = Q[i0:iend,j0:jend,:] -simulation.dt*div[i0:iend,j0:jend,:]


    # Updates for next time step - only for time dependent velocity
    if simulation.vf >= 3:
        # Velocity
        ulon_edx, vlat_edx = velocity_adv(cs_grid.edx.lon,cs_grid.edx.lat, t, simulation)
        ulon_edy, vlat_edy = velocity_adv(cs_grid.edy.lon,cs_grid.edy.lat, t, simulation)

        # Latlon to contravariant
        ucontra_edx, vcontra_edx = latlon_to_contravariant(ulon_edx, vlat_edx, cs_grid.prod_ex_elon_edx,  cs_grid.prod_ex_elat_edx,  cs_grid.prod_ey_elon_edx,  cs_grid.prod_ey_elat_edx, cs_grid.determinant_ll2contra_edx)
        ucontra_edy, vcontra_edy = latlon_to_contravariant(ulon_edy, vlat_edy,  cs_grid.prod_ex_elon_edy,  cs_grid.prod_ex_elat_edy,  cs_grid.prod_ey_elon_edy,  cs_grid.prod_ey_elat_edy, cs_grid.determinant_ll2contra_edy)

        # CFL
        cx[:,:,:] = cfl_x(ucontra_edx, cs_grid, simulation)
        cy[:,:,:] = cfl_y(vcontra_edy, cs_grid, simulation)


