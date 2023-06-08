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
def divergence(cs_grid, simulation):

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
    edges_ghost_cell_treatment_scalar(simulation.Q, simulation.Q, cs_grid, simulation)

    # Multiply the field Q by metric tensor
    simulation.gQ[:,:,:] = simulation.Q[:,:,:]*cs_grid.metric_tensor_pc[:,:,:]

    # CFL
    simulation.cx[:,:,:] = cfl_x(simulation.U_pu.ucontra_averaged[:,:,:], cs_grid, simulation)
    simulation.cy[:,:,:] = cfl_y(simulation.U_pv.vcontra_averaged[:,:,:], cs_grid, simulation)

    # compute the fluxes
    compute_fluxes(simulation.Q, simulation.Q, simulation.px, simulation.py, \
    simulation.U_pu, simulation.U_pv, simulation.cx, simulation.cy, cs_grid, simulation)

    # Applies F and G operators in each panel
    F_operator(cs_grid, simulation)
    G_operator(cs_grid, simulation)
    pxdF, pydF = simulation.px.dF, simulation.py.dF
    gQ = simulation.gQ
    Q = simulation.Q

    # Splitting scheme
    if simulation.opsplit_name=='SP-AVLT':
        gQx = ne.evaluate("gQ + 0.5*pxdF")
        gQy = ne.evaluate("gQ + 0.5*pydF")
        # divide by the metric tensor at centers
        Qx, Qy = gQx/metric_tensor, gQy/metric_tensor

    elif simulation.opsplit_name=='SP-L04':
        # L04 equations 7 and 8
        c1x, c2x = cs_grid.metric_tensor_pu[1:,:,:]*simulation.cx[1:,:,:], cs_grid.metric_tensor_pu[:N+ng,:,:]*simulation.cx[:N+ng,:,:]
        c1y, c2y = cs_grid.metric_tensor_pv[:,1:,:]*simulation.cy[:,1:,:], cs_grid.metric_tensor_pv[:,:N+ng,:]*simulation.cy[:,:N+ng,:]
        gQx = ne.evaluate('gQ + 0.5*pxdF + 0.5*(c1x-c2x)*Q')
        gQy = ne.evaluate('gQ + 0.5*pydF + 0.5*(c1y-c2y)*Q')
        # divide by the metric tensor at centers
        Qx, Qy = gQx/metric_tensor, gQy/metric_tensor

    elif simulation.opsplit_name=='SP-PL07':
        # PL07 - equations 17 and 18
        c1x, c2x = cs_grid.metric_tensor_pu[1:,:,:]*simulation.cx[1:,:,:], cs_grid.metric_tensor_pu[:N+ng,:,:]*simulation.cx[:N+ng,:,:]
        c1y, c2y = cs_grid.metric_tensor_pv[:,1:,:]*simulation.cy[:,1:,:], cs_grid.metric_tensor_pv[:,:N+ng,:]*simulation.cy[:,:N+ng,:]
        Qx = ne.evaluate('0.5*(Q + (Q + pxdF)/(1.0-(c1x-c2x)))')
        Qy = ne.evaluate('0.5*(Q + (Q + pydF)/(1.0-(c1y-c2y)))')

    # applies edge treatment if needed
    if simulation.et_name=='ET-S72' or simulation.et_name=='ET-PL07':
        # Fill ghost cell values
        edges_ghost_cell_treatment_scalar(Qx, Qy, cs_grid, simulation)

    # Compute the fluxes
    compute_fluxes(Qy, Qx, simulation.px, simulation.py,\
    simulation.U_pu, simulation.U_pv, simulation.cx, simulation.cy, cs_grid, simulation)

    # Flux averaging
    if simulation.et_name=='ET-Z21-AF':
        average_flux_cube_edges(simulation.px, simulation.py, cs_grid)

    # Applies F and G operators in each panel again
    F_operator(cs_grid, simulation)
    G_operator(cs_grid, simulation)
 
    # Compute the divergence
    pxdF = simulation.px.dF
    pydF = simulation.py.dF
    simulation.div[:,:,:] = ne.evaluate("-(pxdF + pydF)/(dt*metric_tensor)")

    # Applies mass fixer (project divergence in nullspace)
    if simulation.et_name=='ET-Z21-PR':
        m0 = np.sum(simulation.div[i0:iend,j0:jend,:]*metric_tensor[i0:iend,j0:jend,:])
        a2 = np.sum(metric_tensor[i0:iend,j0:jend,:]*metric_tensor[i0:iend,j0:jend,:])
        simulation.div[i0:iend,j0:jend,:] = simulation.div[i0:iend,j0:jend,:] - metric_tensor[i0:iend,j0:jend,:]*m0/a2
    #print(np.sum(simulation.div[i0:iend,j0:jend,:]*metric_tensor[i0:iend,j0:jend,:]*cs_grid.dx*cs_grid.dx))

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(cs_grid, simulation):
    N = cs_grid.N
    i0 = cs_grid.i0
    iend = cs_grid.iend
    iend = cs_grid.iend
    dx = cs_grid.dx
    dt = simulation.dt

    f1 = simulation.px.f_upw[i0+1:iend+1,:,:]
    f2 = simulation.px.f_upw[i0:iend,:,:]
    simulation.px.dF[i0:iend,:,:] = ne.evaluate("-(f1-f2)")
    simulation.px.dF[i0:iend,:,:] = simulation.px.dF[i0:iend,:,:]*dt/dx 

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(cs_grid, simulation):
    M = cs_grid.N
    j0 = cs_grid.j0
    jend = cs_grid.jend
    dy = cs_grid.dy
    dt = simulation.dt

    g1 = simulation.py.f_upw[:,j0+1:jend+1,:]
    g2 = simulation.py.f_upw[:,j0:jend,:]
    simulation.py.dF[:,j0:jend,:] = ne.evaluate("-(g1-g2)")
    simulation.py.dF[:,j0:jend,:] = simulation.py.dF[:,j0:jend,:]*dt/dy
