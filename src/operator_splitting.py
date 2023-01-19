####################################################################################
# Operator splitting implementation
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
def F_operator(F, Q, u_edges, flux_x, cs_grid, simulation):
    N = cs_grid.N
    i0 = cs_grid.i0
    iend = cs_grid.iend

    F[i0:iend,:,:] = -(simulation.dt/cs_grid.areas[i0:iend,:,:])*cs_grid.dy*(u_edges[i0+1:iend+1,:,:]*flux_x[i0+1:iend+1,:,:] - u_edges[i0:iend,:,:]*flux_x[i0:iend,:,:])

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(G, Q, v_edges, flux_y, cs_grid, simulation):
    M = cs_grid.N
    j0 = cs_grid.j0
    jend = cs_grid.jend

    G[:, j0:jend,:] = -(simulation.dt/cs_grid.areas[:,j0:jend,:])*cs_grid.dx*(v_edges[:,j0+1:jend+1,:]*flux_y[:,j0+1:jend+1,:] - v_edges[:,j0:jend,:]*flux_y[:,j0:jend,:])


####################################################################################
def fix_splitting_operator_ghost_cells(F, G, cs_grid):
    i0 = cs_grid.i0
    iend = cs_grid.iend
    j0 = cs_grid.j0
    jend = cs_grid.jend
    ngl = cs_grid.nghost_left
    ngr = cs_grid.nghost_right

    # Fix F and G at Panel 1/ Panel 4 ghost cells
    F[i0:iend,jend:jend+ngr,1] = G[i0:iend,jend:jend+ngr,1]

    # Fix F at Panel 1/ Panel 5 ghost cells
    F[i0:iend,0:ngl,1] = G[i0:iend,0:ngl,1]

    # Fix F at Panel 3/ Panel 4 ghost cells
    F[i0:iend,jend:jend+ngr,3] = G[i0:iend,jend:jend+ngr,3]

    # Fix F at Panel 3/ Panel 5 ghost cells
    F[i0:iend,0:ngl,3] = G[i0:iend,0:ngl,3]

    # Fix G at Panel 4/ Panel 1 ghost cells
    G[iend:iend+ngr,j0:jend,4] = F[iend:iend+ngr,j0:jend,4]

    # Fix G at Panel 4/ Panel 3 ghost cells
    G[0:ngl,j0:jend,4] = F[0:ngl,j0:jend,4]

    # Fix G at Panel 5/ Panel 3 ghost cells
    G[iend:iend+ngr,j0:jend,5] = F[iend:iend+ngr,j0:jend,5]

    # Fix G at Panel 5/ Panel 1 ghost cells
    G[0:ngl,j0:jend,5] = F[0:ngl,j0:jend,5]


    #exit()


####################################################################################
####################################################################################
# Flux operator in y direction
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
