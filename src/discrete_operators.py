####################################################################################
# This module contains the routine that computes the discrete
# differential operator using finite volume discretization
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
import numexpr as ne
from flux               import compute_fluxes
from edges_treatment    import edges_ghost_cell_treatment_scalar, average_flux_cube_edges, edges_ghost_cell_treatment_vector

####################################################################################
# Given gQ (g = metric tensor), compute div(UgQ), where U = (u,v), and cx and cy
# are the cfl numbers (must be already computed)
# The divergence is given by px.dF + py.dF
####################################################################################
def divergence(Q, gQ, div, px, py, cx, cy, cs_grid, simulation,\
               transformation, lagrange_poly, Kmin, Kmax):
    if simulation.et_name=='ET-S72' or simulation.et_name=='ET-PL07': # Uses adjacent cells values
        # Multiply the field Q by metric tensor
        gQ[:,:,:] = Q[:,:,:]*cs_grid.metric_tensor_centers[:,:,:]
        # Fill ghost cell values - scalar field
        edges_ghost_cell_treatment_scalar(gQ, gQ, cs_grid, simulation, transformation, lagrange_poly, Kmin, Kmax)

    elif simulation.et_name=='ET-R96' or simulation.et_name=='ET-R96-AF' or \
         simulation.et_name=='ET-Z21' or simulation.et_name=='ET-Z21-AF':
        # Fill ghost cell values - scalar field
        edges_ghost_cell_treatment_scalar(Q, Q, cs_grid, simulation, transformation, lagrange_poly, Kmin, Kmax)
        # Multiply the field Q by metric tensor
        gQ[:,:,:] = Q[:,:,:]*cs_grid.metric_tensor_centers[:,:,:]

    # compute the fluxes
    compute_fluxes(gQ, gQ, px, py, cx, cy, cs_grid, simulation)

    # Flux averaging
    if simulation.et_name=='ET-R96-AF' or simulation.et_name=='ET-Z21-AF':
        average_flux_cube_edges(px, py, cs_grid)

    # Applies F and G operators in each panel
    F_operator(px.dF, cx, px.f_upw, cs_grid, simulation)
    G_operator(py.dF, cy, py.f_upw, cs_grid, simulation)

    pxdF = px.dF
    pydF = py.dF
    dt = simulation.dt
    metric_tensor = cs_grid.metric_tensor_centers
    div[:,:,:] = ne.evaluate("-(pxdF + pydF)/(dt*metric_tensor)")

    N = cs_grid.N
    ng = cs_grid.nghost

    # Splitting scheme
    if simulation.opsplit_name=='SP-AVLT':
        #Qx = gQ+0.5*px.dF
        #Qy = gQ+0.5*py.dF
        pxdF = px.dF
        pydF = py.dF
        Qx = ne.evaluate("gQ+0.5*pxdF")
        Qy = ne.evaluate("gQ+0.5*pydF")
    elif simulation.opsplit_name=='SP-L04':
        # L04 equation 7 and 8
        #px.dF = px.dF + (cx[1:,:,:]-cx[:N+ng,:,:])*gQ
        #py.dF = py.dF + (cy[:,1:,:]-cy[:,:N+ng,:])*gQ
        #Qx = gQ+0.5*px.dF
        #Qy = gQ+0.5*py.dF
        pxdF = px.dF
        pydF = py.dF
        c1x, c2x = cx[1:,:,:], cx[:N+ng,:,:]
        c1y, c2y = cy[:,1:,:], cy[:,:N+ng,:]
        Qx = ne.evaluate('(gQ + 0.5*(pxdF + (c1x-c2x)*gQ))')
        Qy = ne.evaluate('(gQ + 0.5*(pydF + (c1y-c2y)*gQ))')
    elif simulation.opsplit_name=='SP-PL07':
        # PL07 - equation 17 and 18
        #Qx = 0.5*(gQ + (gQ + px.dF)/(1.0-(cx[1:,:,:]-cx[:N+ng,:,:])))
        #Qy = 0.5*(gQ + (gQ + py.dF)/(1.0-(cy[:,1:,:]-cy[:,:N+ng,:])))
        pxdF = px.dF
        pydF = py.dF
        c1x, c2x = cx[1:,:,:], cx[:N+ng,:,:]
        c1y, c2y = cy[:,1:,:], cy[:,:N+ng,:]
        Qx = ne.evaluate('0.5*(gQ + (gQ + pxdF)/(1.0-(c1x-c2x)))')
        Qy = ne.evaluate('0.5*(gQ + (gQ + pydF)/(1.0-(c1y-c2y)))')

    if simulation.et_name=='ET-S72' or simulation.et_name=='ET-PL07' or \
       simulation.et_name=='ET-R96' or simulation.et_name=='ET-R96-AF':
        # Fill ghost cell values
        edges_ghost_cell_treatment_scalar(Qx, Qy, cs_grid, simulation, transformation, lagrange_poly, Kmin, Kmax)

    # Compute the fluxes
    compute_fluxes(Qy, Qx, px, py, cx, cy, cs_grid, simulation)

    # Flux averaging
    if simulation.et_name=='ET-R96-AF' or simulation.et_name=='ET-Z21-AF':
        average_flux_cube_edges(px, py, cs_grid)

    # Applies F and G operators in each panel again
    F_operator(px.dF, cx, px.f_upw, cs_grid, simulation)
    G_operator(py.dF, cy, py.f_upw, cs_grid, simulation)

    # Compute the divergence
    #i0, j0, iend, jend  = cs_grid.i0, cs_grid.j0, cs_grid.iend, cs_grid.jend
    #div[i0:iend,j0:jend,:] = -(px.dF[i0:iend,j0:jend,:] + py.dF[i0:iend,j0:jend,:])/simulation.dt/cs_grid.metric_tensor_centers[i0:iend,j0:jend,:]
    pxdF = px.dF
    pydF = py.dF
    dt = simulation.dt
    metric_tensor = cs_grid.metric_tensor_centers
    div[:,:,:] = ne.evaluate("-(pxdF + pydF)/(dt*metric_tensor)")

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

    #F[i0:iend,:,:] = -(simulation.dt/cs_grid.areas[i0:iend,:,:])*cs_grid.dy*(u_edges[i0+1:iend+1,:,:]*flux_x[i0+1:iend+1,:,:] - u_edges[i0:iend,:,:]*flux_x[i0:iend,:,:])
    c1 = cx[i0+1:iend+1,:,:]
    c2 = cx[i0:iend,:,:]
    f1 = flux_x[i0+1:iend+1,:,:]
    f2 = flux_x[i0:iend,:,:]
    F[i0:iend,:,:] = ne.evaluate("-(c1*f1-c2*f2)")


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

    #G[:, j0:jend,:] = -(simulation.dt/cs_grid.areas[:,j0:jend,:])*cs_grid.dy*(v_edges[:,j0+1:jend+1,:]*flux_y[:,j0+1:jend+1,:] - v_edges[:,j0:jend,:]*flux_y[:,j0:jend,:])
    c1 = cy[:,j0+1:jend+1,:]
    c2 = cy[:,j0:jend,:]
    g1 = flux_y[:,j0+1:jend+1,:]
    g2 = flux_y[:,j0:jend,:]
    G[:,j0:jend,:] = ne.evaluate("-(c1*g1-c2*g2)")

