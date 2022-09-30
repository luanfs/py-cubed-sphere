####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from advection_ic           import velocity_adv, q0_adv
from cs_datastruct          import scalar_field
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from stencil                import flux_ppm_x_stencil_coefficients, flux_ppm_y_stencil_coefficients

def init_vars_adv(cs_grid, simulation, N, nghost, Nsteps):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Get edges position in lat/lon system
    edx_lon = cs_grid.edx.lon
    edx_lat = cs_grid.edx.lat
    edy_lon = cs_grid.edy.lon
    edy_lat = cs_grid.edy.lat

    # Get center position in lat/lon system
    center_lon = cs_grid.centers.lon
    center_lat = cs_grid.centers.lat

    # Metric tensor
    g_metric = cs_grid.metric_tensor_centers

    # Velocity (latlon) at edges
    ulon_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    vlat_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    ulon_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))
    vlat_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))

    # Velocity (contravariant) at edges
    ucontra_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    vcontra_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    ucontra_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))
    vcontra_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))

    # Get velocities
    ulon_edx[:,:,:], vlat_edx[:,:,:] = velocity_adv(edx_lon, edx_lat, 0.0, simulation)
    ulon_edy[:,:,:], vlat_edy[:,:,:] = velocity_adv(edy_lon, edy_lat, 0.0, simulation)

    # Integration variable
    Q_old = np.zeros((N+nghost, N+nghost, nbfaces))
    Q_new = np.zeros((N+nghost, N+nghost, nbfaces))

    # Convert latlon to contravariant at ed_x
    ex_elon_edx = cs_grid.prod_ex_elon_edx
    ex_elat_edx = cs_grid.prod_ex_elat_edx
    ey_elon_edx = cs_grid.prod_ey_elon_edx
    ey_elat_edx = cs_grid.prod_ey_elat_edx
    det_edx    = cs_grid.determinant_ll2contra_edx
    ucontra_edx, vcontra_edx = latlon_to_contravariant(ulon_edx, vlat_edx, ex_elon_edx, ex_elat_edx, ey_elon_edx, ey_elat_edx, det_edx)
    #ulon_edx2, vlat_edx2 = contravariant_to_latlon(ucontra_edx, vcontra_edx, ex_elon_edx, ex_elat_edx, ey_elon_edx, ey_elat_edx)

    # Convert latlon to contravariant at ed_y
    ex_elon_edy = cs_grid.prod_ex_elon_edy
    ex_elat_edy = cs_grid.prod_ex_elat_edy
    ey_elon_edy = cs_grid.prod_ey_elon_edy
    ey_elat_edy = cs_grid.prod_ey_elat_edy
    det_edy     = cs_grid.determinant_ll2contra_edy
    ucontra_edy, vcontra_edy = latlon_to_contravariant(ulon_edy, vlat_edy, ex_elon_edy, ex_elat_edy, ey_elon_edy, ey_elat_edy, det_edy)
    ulon_edy2, vlat_edy2 = contravariant_to_latlon(ucontra_edy, vcontra_edy, ex_elon_edy, ex_elat_edy, ey_elon_edy, ey_elat_edy)

    # CFL at edges - x direction
    cx, cx2 = cfl_x(ucontra_edx, cs_grid, simulation)

    # CFL at edges - y direction
    cy, cy2 = cfl_y(vcontra_edy, cs_grid, simulation)

    # CFL number
    CFL_x = np.amax(cx)
    CFL_y = np.amax(cy)
    CFL = np.sqrt(CFL_x**2 + CFL_y**2)
    #print('CFL number = ', CFL)
    #exit()

    # Flux at edges
    flux_x = np.zeros((N+7, N+6, nbfaces))
    flux_y = np.zeros((N+6, N+7, nbfaces))

    # Stencil coefficients
    ax = np.zeros((6, N+7, N+6, nbfaces))
    ay = np.zeros((6, N+6, N+7, nbfaces))

    # Compute the coefficients
    flux_ppm_x_stencil_coefficients(ucontra_edx, ax, cx, cx2, simulation)
    flux_ppm_y_stencil_coefficients(vcontra_edy, ay, cy, cy2, simulation)

    # Compute average values of Q (initial condition) at cell centers
    Q = scalar_field(cs_grid, 'Q', 'center')
    Q_old[i0:iend,j0:jend,:] = q0_adv(center_lon[i0:iend,j0:jend,:], center_lat[i0:iend,j0:jend,:], simulation)
    Q.f[:,:,:] = Q_old[i0:iend,j0:jend,:]

    # Exact field
    q_exact = scalar_field(cs_grid, 'q_exact', 'center')

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    # Dimension splitting operators
    F_gQ  = np.zeros((N+nghost, N+nghost, nbfaces))
    G_gQ  = np.zeros((N+nghost, N+nghost, nbfaces))
    GF_gQ = np.zeros((N+nghost, N+nghost, nbfaces))
    FG_gQ = np.zeros((N+nghost, N+nghost, nbfaces))
    
    return Q_new, Q_old, Q, q_exact, flux_x, flux_y, ax, ay, cx, cy, cx2, cy2, \
           error_linf, error_l1, error_l2, F_gQ, G_gQ, GF_gQ, FG_gQ, \
           ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, g_metric
