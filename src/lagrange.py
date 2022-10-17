####################################################################################
#
# Module for Lagrange polynomials routines needed for interpolation
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################


import numpy as np
from constants import pio4
from cs_transform import inverse_equidistant_gnomonic_map, inverse_equiangular_gnomonic_map

####################################################################################
#Compute the jth Lagrange polynomial of degree N
####################################################################################
def lagrange_basis(x, nodes, N, j):
    Lj = 1.0
    for i in range(0,N+1):
        if i != j:
            Lj = Lj*(x-nodes[i])/(nodes[j]-nodes[i])
    return Lj

####################################################################################
#Compute the Lagrange polynomial basis at the ghost cells
####################################################################################
def lagrange_poly_ghostcells(cs_grid, simulation, transformation):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.nghost   # Number o ghost cells
    ngl = cs_grid.nghost_left
    ngr = cs_grid.nghost_right

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    if transformation == "gnomonic_equiangular":
        inverse = inverse_equiangular_gnomonic_map
        x_min, x_max = [-pio4, pio4] # Angular coordinates
    elif transformation == "gnomonic_equidistant":
        inverse = inverse_equidistant_gnomonic_map
        a = cs_grid.R/np.sqrt(3.0)  # Half length of the cube
        x_min, x_max = [-a, a]

    dx = cs_grid.dx
    xc = np.linspace(x_min+dx/2.0-ngl*dx, x_max-dx/2.0+ngr*dx, N+ng) # Centers
    degree = simulation.degree
    order = degree+1

    p = 0
    north = 4
    south = 5
    east  = 1
    west  = 3

    # Ghost cells at east
    X_ghost = cs_grid.centers.X[iend:iend+ngr,:,p]
    Y_ghost = cs_grid.centers.Y[iend:iend+ngr,:,p]
    Z_ghost = cs_grid.centers.Z[iend:iend+ngr,:,p]
    x_ghost, y_ghost = inverse(X_ghost, Y_ghost, Z_ghost, east)

    # Support points
    X = cs_grid.centers.X[i0:i0+ngr,:,east]
    Y = cs_grid.centers.Y[i0:i0+ngr,:,east]
    Z = cs_grid.centers.Z[i0:i0+ngr,:,east]
    x, y = inverse(X, Y, Z, east)

    halo_ghost_points_east  = np.zeros((ngl, N+ng))
    halo_ghost_points_east[:,:] = y_ghost

    # Interpolation indexes
    K = (np.floor((halo_ghost_points_east-xc[0])/dx)).astype(int)
    Kmax = np.minimum(K + order, N+ng).astype(int)
    Kmin = np.maximum(Kmax-order, 0).astype(int)

    K[:,i0:iend] =  (np.floor((halo_ghost_points_east[:,i0:iend]-xc[0])/dx)).astype(int)
    Kmax[:,i0:iend] = np.minimum(K[:,i0:iend] + order, N+ngl).astype(int)
    Kmin[:,i0:iend] = np.maximum(Kmax[:,i0:iend]-order, ngl).astype(int)

    halo_lagrange_nodes = np.zeros((ngr, N+ng, order))
    lagrange_poly = np.zeros((ngr, N+ng, order))

    # Compute the Lagrange nodes at halo region
    for g in range(0, ngl):
        for j in range(0, N+ng):
            halo_lagrange_nodes[g,j,:] = y[g,Kmin[g,j]:Kmax[g,j]]

   # Compute the Lagrange nodes at halo region
    for g in range(0, ngr):
        for k in range(0, N+ng):
            for l in range(0, order):
                lagrange_poly[g,k,l] = lagrange_basis(halo_ghost_points_east[g,k], halo_lagrange_nodes[g,k,:], degree, l)

    lagrange_poly_east  = lagrange_poly
    lagrange_poly_west  = np.flip(lagrange_poly,axis=0)
    lagrange_poly_north = np.transpose(lagrange_poly_east,(1,0,2))
    lagrange_poly_south = np.flip(lagrange_poly_north,axis=1)

    Kmin_east , Kmax_east  = Kmin, Kmax
    Kmin_west , Kmax_west  = np.flip(Kmin,axis=0), np.flip(Kmax,axis=0)
    Kmin_north, Kmax_north = np.transpose(Kmin_east,(1,0)), np.transpose(Kmax_east,(1,0))
    Kmin_south, Kmax_south = np.flip(Kmin_north,axis=1), np.flip(Kmax_north,axis=1)

    Kmin = [Kmin_east, Kmin_west, Kmin_north, Kmin_south]
    Kmax = [Kmax_east, Kmax_west, Kmax_north, Kmax_south]
    lagrange_poly = [lagrange_poly_east, lagrange_poly_west, lagrange_poly_north, lagrange_poly_south]
    return lagrange_poly, Kmin, Kmax
