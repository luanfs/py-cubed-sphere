####################################################################################
# Dimension splitting operators implementation
# Luan da Fonseca Santos - June 2022
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
###################################################################################

import numpy as np
from flux import  compute_flux_x, compute_flux_y

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(F, Q, u_edges, flux_x, ax, cs_grid, simulation):
    N = cs_grid.N
    i0 = cs_grid.i0
    iend = cs_grid.iend
    compute_flux_x(flux_x, Q, u_edges, ax, cs_grid, simulation)
    F[3:N+3,:,:] = -(simulation.dt/cs_grid.areas[i0:iend,:,:])*cs_grid.dy*(u_edges[4:N+4,:,:]*flux_x[4:N+4,:,:] - u_edges[3:N+3,:,:]*flux_x[3:N+3,:,:])

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(G, Q, v_edges, flux_y, ay, cs_grid, simulation):
    M = cs_grid.N
    j0 = cs_grid.j0
    jend = cs_grid.jend

    compute_flux_y(flux_y, Q, v_edges, ay, cs_grid, simulation)
    G[:, 3:M+3,:] = -(simulation.dt/cs_grid.areas[:,j0:jend,:])*cs_grid.dx*(v_edges[:,4:M+4,:]*flux_y[:,4:M+4,:] - v_edges[:,3:M+3,:]*flux_y[:,3:M+3,:])


####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
#def G_operator_stencil(Q, v_edges, cs_grid, simulation, p):
#    N  = cs_grid.N
#    ng = cs_grid.nghost

#    flux_y = flux_ppm_y_stencil(Q, v_edges, cs_grid, simulation)

#    G = np.zeros(np.shape(Q))
#    G[:,:] = -(simulation.dt/cs_grid.areas[:,:,p])*cs_grid.dx*(flux_y[:,1:N+ng+1] - flux_y[:,0:N+ng])

#    return G

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
#def F_operator_stencil(Q, u_edges, cs_grid, simulation, p):
#    N  = cs_grid.N
#    ng = cs_grid.nghost

#    flux_x = flux_ppm_x_stencil(Q, u_edges, cs_grid, simulation)

#    F = np.zeros(np.shape(Q))
#    F[:,:] = -(simulation.dt/cs_grid.areas[:,:,p])*cs_grid.dy*(flux_x[1:N+ng+1,:] - flux_x[0:N+ng,:])

#    return F
