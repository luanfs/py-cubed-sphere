####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from advection_ic           import velocity_adv, q0_adv
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid, ppm_parabola, velocity_edges
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from lagrange               import lagrange_poly_ghostcells

####################################################################################
# This routine initializates the advection routine variables
####################################################################################
def init_vars_adv(cs_grid, simulation, transformation):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # other grid vars
    N = cs_grid.N
    nghost = cs_grid.nghost

    # Velocity at edges
    U_pu = velocity_edges(cs_grid, 'pu')
    U_pv = velocity_edges(cs_grid, 'pv')

    # Get velocities
    U_pu.ulon[:,:,:], U_pu.vlat[:,:,:] = velocity_adv(cs_grid.edx.lon, cs_grid.edx.lat, 0.0, simulation)
    U_pv.ulon[:,:,:], U_pv.vlat[:,:,:] = velocity_adv(cs_grid.edy.lon, cs_grid.edy.lat, 0.0, simulation)

    # Convert latlon to contravariant at pu
    U_pu.ucontra[:,:,:], U_pu.vcontra[:,:,:] = latlon_to_contravariant(U_pu.ulon, U_pu.vlat, cs_grid.prod_ex_elon_edx, cs_grid.prod_ex_elat_edx,\
                                                       cs_grid.prod_ey_elon_edx, cs_grid.prod_ey_elat_edx, cs_grid.determinant_ll2contra_edx)

    # Convert latlon to contravariant at pv
    U_pv.ucontra[:,:,:], U_pv.vcontra[:,:,:] = latlon_to_contravariant(U_pv.ulon, U_pv.vlat, cs_grid.prod_ex_elon_edy, cs_grid.prod_ex_elat_edy,\
                                                       cs_grid.prod_ey_elon_edy, cs_grid.prod_ey_elat_edy, cs_grid.determinant_ll2contra_edy)

    # CFL at edges - x direction
    cx = cfl_x(U_pu.ucontra, cs_grid, simulation)

    # CFL at edges - y direction
    cy = cfl_y(U_pv.vcontra, cs_grid, simulation)

    # CFL number
    CFL_x = np.amax(cx)
    CFL_y = np.amax(cy)
    CFL = max(abs(CFL_x),abs(CFL_y))

    # PPM parabolas
    px = ppm_parabola(cs_grid,simulation,'x')
    py = ppm_parabola(cs_grid,simulation,'y')

    # Compute average values of Q (initial condition) at cell centers
    Q = np.zeros((N+nghost, N+nghost, nbfaces))
    gQ = np.zeros((N+nghost, N+nghost, nbfaces))
    Q[i0:iend,j0:jend,:] = q0_adv(cs_grid.centers.lon[i0:iend,j0:jend,:], cs_grid.centers.lat[i0:iend,j0:jend,:], simulation)

    # Numerical divergence
    div_numerical = np.zeros((N+nghost, N+nghost, nbfaces))

    # Get Lagrange polynomials
    lagrange_poly, Kmin, Kmax = lagrange_poly_ghostcells(cs_grid, simulation, transformation)

    return Q, gQ, div_numerical, px, py, cx, cy, \
           U_pu, U_pv, lagrange_poly, Kmin, Kmax, CFL
