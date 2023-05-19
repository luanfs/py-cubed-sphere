####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from advection_ic           import velocity_adv, q0_adv
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid, ppm_parabola, velocity
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from lagrange               import lagrange_poly_ghostcell_pc, wind_edges2center_lagrange_poly, wind_center2ghostedges_lagrange_poly_ghost
from edges_treatment    import edges_ghost_cell_treatment_vector

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
    ng = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Velocity at edges
    U_pu = velocity(cs_grid, 'pu')
    U_pv = velocity(cs_grid, 'pv')

    # Get velocities
    U_pu.ulon[:,:,:], U_pu.vlat[:,:,:] = velocity_adv(cs_grid.pu.lon, cs_grid.pu.lat, 0.0, simulation)
    U_pv.ulon[:,:,:], U_pv.vlat[:,:,:] = velocity_adv(cs_grid.pv.lon, cs_grid.pv.lat, 0.0, simulation)

    # Convert latlon to contravariant at pu
    U_pu.ucontra[:,:,:], U_pu.vcontra[:,:,:] = latlon_to_contravariant(U_pu.ulon, U_pu.vlat, cs_grid.prod_ex_elon_pu, cs_grid.prod_ex_elat_pu,\
                                                       cs_grid.prod_ey_elon_pu, cs_grid.prod_ey_elat_pu, cs_grid.determinant_ll2contra_pu)

    # Convert latlon to contravariant at pv
    U_pv.ucontra[:,:,:], U_pv.vcontra[:,:,:] = latlon_to_contravariant(U_pv.ulon, U_pv.vlat, cs_grid.prod_ex_elon_pv, cs_grid.prod_ex_elat_pv,\
                                                       cs_grid.prod_ey_elon_pv, cs_grid.prod_ey_elat_pv, cs_grid.determinant_ll2contra_pv)

    # Fill ghost cell - velocity field
    edges_ghost_cell_treatment_vector(U_pu.ucontra, U_pv.vcontra, cs_grid, simulation)

    U_pu.ucontra_old[:,:,:] = U_pu.ucontra[:,:,:]
    U_pv.vcontra_old[:,:,:] = U_pv.vcontra[:,:,:]

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

    # Compute average values of Q (initial condition) at cell pc
    Q = np.zeros((N+ng, N+ng, nbfaces))
    gQ = np.zeros((N+ng, N+ng, nbfaces))
    Q[i0:iend,j0:jend,:] = q0_adv(cs_grid.pc.lon[i0:iend,j0:jend,:], cs_grid.pc.lat[i0:iend,j0:jend,:], simulation)

    # Numerical divergence
    div_numerical = np.zeros((N+ng, N+ng, nbfaces))

    # Compute the Lagrange polynomials
    if cs_grid.projection=="gnomonic_equiangular":
        lagrange_poly_edge, stencil_edge = wind_edges2center_lagrange_poly(cs_grid, simulation, transformation)
        lagrange_poly_ghost_pc, stencil_ghost_pc = lagrange_poly_ghostcell_pc(cs_grid, simulation, transformation)
        lagrange_poly_ghost_edge, stencil_ghost_edge = wind_center2ghostedges_lagrange_poly_ghost(cs_grid, simulation, transformation)
    else:
        lagrange_poly_edge, stencil_edge = None, None 
        lagrange_poly_ghost_pc, stencil_ghost_pc =  None, None 
        lagrange_poly_ghost_edge, stencil_ghost_edge = None, None 

    # Edge treatment may modify the metric tensor in ghost cells using adjacent panel values
    if simulation.et_name=='ET-S72' or simulation.et_name=='ET-PL07': # Uses adjacent cells values
        # x direction
        cs_grid.metric_tensor_pc[iend:N+ng,:] = cs_grid.metric_tensor_pc[i0:i0+ngr,:]
        cs_grid.metric_tensor_pc[0:i0,:]      = cs_grid.metric_tensor_pc[N:N+ngl,:]

        # y direction
        cs_grid.metric_tensor_pc[:,jend:N+ng] = cs_grid.metric_tensor_pc[:,j0:j0+ngr]
        cs_grid.metric_tensor_pc[:,0:j0]      = cs_grid.metric_tensor_pc[:,N:N+ngl]

    return Q, gQ, div_numerical, px, py, cx, cy, \
           U_pu, U_pv, lagrange_poly_ghost_pc, stencil_ghost_pc,\
           lagrange_poly_edge, stencil_edge,\
           lagrange_poly_ghost_edge, stencil_ghost_edge, CFL
