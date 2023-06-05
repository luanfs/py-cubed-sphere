####################################################################################
# This module contains the routine that computes the discrete
# differential operator using finite volume discretization
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
import numexpr as ne
from flux               import compute_fluxes
from cfl                import cfl_x, cfl_y
from edges_treatment    import edges_ghost_cell_treatment_scalar,\
average_flux_cube_edges, edges_ghost_cell_treatment_vector, average_parabola_cube_edges
from averaged_velocity  import time_averaged_velocity

####################################################################################
# Given gQ (g = metric tensor), compute div(UgQ), where U = (u,v), and cx and cy
# are the cfl numbers (must be already computed)
# The divergence is given by px.dF + py.dF
####################################################################################
def divergence(Q, gQ, div, U_pu, U_pv, px, py, cx, cy, cs_grid, simulation,\
               transformation, lagrange_poly_ghost_pc, stencil_ghost_pc):

    # Interior of the grid
    i0, iend = cs_grid.i0, cs_grid.iend
    j0, jend = cs_grid.j0, cs_grid.jend
    dt = simulation.dt
    N = cs_grid.N
    ng = cs_grid.ng
    dx = cs_grid.dx
    dy = cs_grid.dy
    metric_tensor = cs_grid.metric_tensor_pc

    # Fill ghost cell values - scalar field
    edges_ghost_cell_treatment_scalar(Q, Q, cs_grid, simulation, transformation, \
    lagrange_poly_ghost_pc, stencil_ghost_pc)

    # Multiply the field Q by metric tensor
    gQ[:,:,:] = Q[:,:,:]*cs_grid.metric_tensor_pc[:,:,:]

    # CFL
    cx[:,:,:] = cfl_x(U_pu.ucontra_averaged[:,:,:], cs_grid, simulation)
    cy[:,:,:] = cfl_y(U_pv.vcontra_averaged[:,:,:], cs_grid, simulation)

    # compute the fluxes
    compute_fluxes(Q, Q, px, py, U_pu, U_pv, cx, cy, cs_grid, simulation)

    # Applies F and G operators in each panel
    F_operator(px.dF, cx, px.f_upw, cs_grid, simulation)
    G_operator(py.dF, cy, py.f_upw, cs_grid, simulation)

    pxdF = px.dF
    pydF = py.dF

    # Splitting scheme
    if simulation.opsplit_name=='SP-AVLT':
        gQx = ne.evaluate("gQ+0.5*pxdF")
        gQy = ne.evaluate("gQ+0.5*pydF")
        # divide by the metric tensor at centers
        Qx, Qy = gQx/metric_tensor, gQy/metric_tensor
        #Qx, Qy = gQx, gQy

    elif simulation.opsplit_name=='SP-L04':
        # L04 equation 7 and 8
        c1x, c2x = cs_grid.metric_tensor_pu[1:,:,:]*cx[1:,:,:], cs_grid.metric_tensor_pu[:N+ng,:,:]*cx[:N+ng,:,:]
        c1y, c2y = cs_grid.metric_tensor_pv[:,1:,:]*cy[:,1:,:], cs_grid.metric_tensor_pv[:,:N+ng,:]*cy[:,:N+ng,:]
        gQx = ne.evaluate('(gQ + 0.5*(pxdF + (c1x-c2x)*Q))')
        gQy = ne.evaluate('(gQ + 0.5*(pydF + (c1y-c2y)*Q))')
        # divide by the metric tensor at centers
        Qx, Qy = gQx/metric_tensor, gQy/metric_tensor

    elif simulation.opsplit_name=='SP-PL07':
        # PL07 - equation 17 and 18
        c1x, c2x = cs_grid.metric_tensor_pu[1:,:,:]*cx[1:,:,:], cs_grid.metric_tensor_pu[:N+ng,:,:]*cx[:N+ng,:,:]
        c1y, c2y = cs_grid.metric_tensor_pv[:,1:,:]*cy[:,1:,:], cs_grid.metric_tensor_pv[:,:N+ng,:]*cy[:,:N+ng,:]
        Qx = ne.evaluate('0.5*(Q + (Q + pxdF)/(1.0-(c1x-c2x)))')
        Qy = ne.evaluate('0.5*(Q + (Q + pydF)/(1.0-(c1y-c2y)))')

    # applies edge treatment if needed
    if simulation.et_name=='ET-S72' or simulation.et_name=='ET-PL07':
        # Fill ghost cell values
        edges_ghost_cell_treatment_scalar(Qx, Qy, cs_grid, simulation, transformation, lagrange_poly_ghost_pc, stencil_ghost_pc)

    # Compute the fluxes
    compute_fluxes(Qy, Qx, px, py, U_pu, U_pv, cx, cy, cs_grid, simulation)

    ##############################################################################
    # Flux averaging
    ##############################################################################
    if simulation.et_name=='ET-Z21-AF':
        average_flux_cube_edges(px, py, cs_grid)

    # Applies F and G operators in each panel again
    F_operator(px.dF, cx, px.f_upw, cs_grid, simulation)
    G_operator(py.dF, cy, py.f_upw, cs_grid, simulation)

    # Compute the divergence
    pxdF = px.dF
    pydF = py.dF
    div[:,:,:] = ne.evaluate("-(pxdF + pydF)/(dt*metric_tensor)")
    print(np.sum(div[i0:iend,j0:jend,:]*metric_tensor[i0:iend,j0:jend,:]*cs_grid.dx*cs_grid.dx))

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(F, cx, flux_x, cs_grid, simulation):
    N = cs_grid.N
    i0 = cs_grid.i0
    iend = cs_grid.iend
    iend = cs_grid.iend
    dx = cs_grid.dx
    dt = simulation.dt

    f1 = flux_x[i0+1:iend+1,:,:]
    f2 = flux_x[i0:iend,:,:]
    F[i0:iend,:,:] = ne.evaluate("-(f1-f2)")
    F[i0:iend,:,:] = F[i0:iend,:,:]*dt/dx 

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(G, cy, flux_y, cs_grid, simulation):
    M = cs_grid.N
    j0 = cs_grid.j0
    jend = cs_grid.jend
    dy = cs_grid.dy
    dt = simulation.dt

    g1 = flux_y[:,j0+1:jend+1,:]
    g2 = flux_y[:,j0:jend,:]
    G[:,j0:jend,:] = ne.evaluate("-(g1-g2)")
    G[:,j0:jend,:] = G[:,j0:jend,:]*dt/dy
