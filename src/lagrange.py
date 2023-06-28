####################################################################################
#
# Module for Lagrange polynomials routines needed for interpolation
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################


import numpy as np
from math import floor, ceil
from constants import pio4, nbfaces, rad2deg
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
#Compute the Lagrange polynomial basis at the ghost cells centers
####################################################################################
def lagrange_poly_ghostcell_pc(cs_grid, simulation):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.ng   # Number o ghost cells
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    if cs_grid.projection == "gnomonic_equiangular":
        inverse = inverse_equiangular_gnomonic_map
        y_min, y_max = [-pio4, pio4] # Angular coordinates
    else:
        print('ERROR in lagrange_poly_ghostcells: grid is not gnomonic_equiangular.')
        exit()

    dy = cs_grid.dy
    yc = np.linspace(y_min+dy/2.0-ngl*dy, y_max-dy/2.0+ngr*dy, N+ng) # Centers
    ye = np.linspace(y_min-ngl*dy, y_max+ngr*dy, N+ng+1) # edges 
    degree = simulation.degree
    order = degree+1

    p = 0
    north = 4
    south = 5
    east  = 1
    west  = 3

    # Ghost cells at east
    X_ghost = cs_grid.pc.X[iend:iend+ngr,:,p]
    Y_ghost = cs_grid.pc.Y[iend:iend+ngr,:,p]
    Z_ghost = cs_grid.pc.Z[iend:iend+ngr,:,p]
    x_ghost, y_ghost = inverse(X_ghost, Y_ghost, Z_ghost, east)

    # Support points
    X = cs_grid.pc.X[i0:i0+ngr,:,east]
    Y = cs_grid.pc.Y[i0:i0+ngr,:,east]
    Z = cs_grid.pc.Z[i0:i0+ngr,:,east]
    x, y = inverse(X, Y, Z, east)

    halo_ghost_points_east  = np.zeros((ngl, N+ng))
    halo_ghost_points_east[:,:] = y_ghost

    rad2deg = 180.0/np.pi

    # Interpolation indexes
    K = np.zeros((ngl, N+ng))
    Kmin = np.zeros((ngl, N+ng))
    Kmax = np.zeros((ngl, N+ng))

    for g in range(0, ngl):
        for i in range(0, N+ng):
            K[g,i] =((halo_ghost_points_east[g,i]-y[g,0])/dy)
            Kmax[g,i] = K[g,i]+ceil(order/2)
            Kmin[g,i] = Kmax[g,i]-order+1
 
            # Shift stencils if needed
            if i in range(i0, iend):
                if Kmax[g,i]>=iend:
                    Kmax[g,i] = iend-1
                    Kmin[g,i] = Kmax[g,i]-order+1
                elif Kmin[g,i]<i0:
                    Kmin[g,i] = i0
                    Kmax[g,i] = Kmin[g,i]+order-1
            
            elif i>=iend:
                if Kmax[g,i]>=N+ng:
                    Kmax[g,i] = N+ng-1
                    Kmin[g,i] = Kmax[g,i]-order+1
            else: #i<i0
                if Kmin[g,i]<0:
                    Kmin[g,i] = 0
                    Kmax[g,i] = Kmin[g,i]+order-1

    K = K.astype(int)
    Kmin = Kmin.astype(int)
    Kmax = Kmax.astype(int)

    # Debugging
    for g in range(0, ngl):
        for i in range(i0, iend):
        #for i in range(0,N+ng):
            #print(i,'K=',K[g,i],', stencil: ',Kmin[g,i],Kmax[g,i])
            #print(i,Kmin[g,i],Kmax[g,i], Kmax[g,i]-Kmin[g,i])
            #print('g = ',g,' ', y_ghost[g,i]*rad2deg, y[g,Kmin[g,i]]*rad2deg,y[g,Kmax[g,i]]*rad2deg)

            if (Kmax[g,i]-Kmin[g,i] != degree):
                print('ERROR in lagrange_poly_ghostcells', degree)
                exit()
            if Kmin[g,i]<i0 or Kmax[g,i]>iend:
                print('Error in lagrange_poly_ghostcells')
                exit()
            if order>1:
                if not ((y_ghost[g,i]>=y[g,Kmin[g,i]]) and (y_ghost[g,i]<=y[g,Kmax[g,i]]) ):
                    print('ERROR in lagrange_poly_ghostcells')
                    exit()
        #print()
        #print('\n',g, iend, ye[i0]*rad2deg, ye[iend]*rad2deg) 
    #exit()
    halo_lagrange_nodes = np.zeros((ngr, N+ng, order))
    lagrange_poly = np.zeros((ngr, N+ng, order))

    # Compute the Lagrange nodes at halo region
    for g in range(0, ngl):
        for j in range(0, N+ng):
            halo_lagrange_nodes[g,j,:] = y[g,Kmin[g,j]:Kmax[g,j]+1]

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

    stencil = [Kmin, Kmax]
    lagrange_poly = [lagrange_poly_east, lagrange_poly_west, lagrange_poly_north, lagrange_poly_south]

    simulation.stencil_ghost_pc = stencil
    simulation.lagrange_poly_ghost_pc = lagrange_poly

    return

####################################################################################
# Compute the Lagrange polynomial basis needed to interpolate a vector field given in
# a C grid (contravariant) to the center
####################################################################################
def wind_edges2center_lagrange_poly(cs_grid, simulation):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr
    dx = cs_grid.dx

    # Order
    degree = simulation.degree
    order  = degree + 1

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    if cs_grid.projection == "gnomonic_equiangular":
        x_min, x_max = [-pio4, pio4] # Angular coordinates
    elif cs_grid.projection == "overlapped":
        x_min, x_max = [-1.0,1.0]
    else:
        print('ERROR in lagrange_poly_ghostcells: grid is not valid, ',cs_grid.projection)
        exit()

    # Positions
    x_pu = np.linspace(x_min-ngl*dx, x_max+ngr*dx, N+1+ng) # vertices
    x_pc = np.linspace(x_min+dx/2.0-ngl*dx, x_max-dx/2.0+ngr*dx, N+ng) # Centers
    dx = cs_grid.dx

    # Define the stencil
    K    = np.zeros(N+ng, dtype=int)
    Kmax = np.zeros(N+ng, dtype=int)
    Kmin = np.zeros(N+ng, dtype=int)
 
    for i in range(i0, iend):
        K[i] = i
        Kmax[i] = K[i]+ceil(order/2) 
        Kmin[i] = Kmax[i]-order+1

        # Shift stencils if needed
        if Kmin[i]<i0:
            Kmin[i] = i0
            Kmax[i] = Kmin[i]+order-1

        if Kmax[i]>iend:
            Kmax[i] = iend
            Kmin[i] = Kmax[i]-order+1

        if Kmax[i]-Kmin[i] != order-1:
            print('Error in wind_edges2center_lagrange_poly.')
            exit()
        #print(Kmin[i], Kmax[i], i)

    lagrange_poly_x = np.zeros((N+ng, N+ng, nbfaces, order))
    lagrange_poly_y = np.zeros((N+ng, N+ng, nbfaces, order))
    nodes = np.zeros((N+ng, order))

    # Compute the Lagrange nodes for each x_pc 
    for i in range(i0, iend):
        nodes[i,:] = x_pu[Kmin[i]:Kmax[i]+1]
        #print(nodes[i,:]*rad2deg, x_pc[i]*rad2deg, Kmin[i],Kmax[i])
        # Debugging
        if nodes[i,0]>x_pc[i] or nodes[i,order-1]<x_pc[i]:
            if degree>=1:
                print('Error in wind_edges2center_lagrange_poly.')
                exit()

    # Compute the Lagrange polynomial basis at pc
    for i in range(i0, iend):
        for k in range(0, order):
            lagrange_poly_x[i,:,:,k] = lagrange_basis(x_pc[i], nodes[i,:], degree, k)
            lagrange_poly_y[:,i,:,k] = lagrange_poly_x[i,:,:,k]

    stencil = [Kmin, Kmax]
    lagrange_poly = [lagrange_poly_x, lagrange_poly_y]

    simulation.lagrange_poly_edge = lagrange_poly
    simulation.stencil_edge  = stencil

    return

####################################################################################
# Compute the Lagrange polynomial basis needed to interpolate a latlon vector field 
# given at the cell center (including ghost cells) to the ghost cell edges
####################################################################################
def wind_center2ghostedges_lagrange_poly_ghost(cs_grid, simulation):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr
    dx = cs_grid.dx

    # Order
    degree = simulation.degree
    order  = degree + 1

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Positions
    x_pu = np.linspace(-pio4-ngl*dx, pio4+ngr*dx, N+1+ng) # vertices
    x_pc = np.linspace(-pio4+dx/2.0-ngl*dx, pio4-dx/2.0+ngr*dx, N+ng) # Centers
    dx = cs_grid.dx

    # Define the stencil
    K = np.zeros(N+ng, dtype=int)
    Kmin = np.zeros(N+ng, dtype=int)
    Kmax = np.zeros(N+ng, dtype=int)

    for i in range(i0, iend+1):
        #Kmin[i] = i-1 # Cell where the center point is located
        K[i] = i-1
        Kmax[i] = K[i]+ceil(order/2) 
        Kmin[i] = Kmax[i]-order+1

        # Shift stencils if needed
        if Kmin[i]<0:
            Kmin[i] = 0
            Kmax[i] = Kmin[i]+order-1

        if Kmax[i]>=N+ng:
            Kmax[i] = N+ng-1
            Kmin[i] = Kmax[i]-order+1

        #print(Kmin[i], Kmax[i], i)
        if Kmax[i]-Kmin[i] != order-1:
            print('Error in wind_center2ghostedges_lagrange_poly_ghost')
            exit()
        #print(Kmin[i],Kmax[i],i)

    lagrange_poly_east = np.zeros((ngl, N+ng+1, nbfaces, order))
    lagrange_poly_west = np.zeros((ngl, N+ng+1, nbfaces, order))
    lagrange_poly_north = np.zeros((N+ng+1, ngl, nbfaces, order))
    lagrange_poly_south = np.zeros((N+ng+1, ngl, nbfaces, order))
    nodes = np.zeros((N+ng, order))

    # Compute the Lagrange nodes for each x_pc 
    for i in range(i0, iend+1):
        nodes[i,:] = x_pc[Kmin[i]:Kmax[i]+1]
        #print(nodes[i,:]*rad2deg, x_pu[i]*rad2deg, Kmin[i],Kmax[i])
        # Debugging
        if nodes[i,0]>x_pu[i] or nodes[i,order-1]<x_pu[i]:
            if degree>=1:
                print('Error in wind_center2ghostedges_lagrange_poly_ghost.')
                exit()

    # Compute the Lagrange polynomial basis at pc
    for j in range(j0, jend+1):
        for k in range(0, order):
            lagrange_poly_east[:,j,:,k] = lagrange_basis(x_pu[j], nodes[j,:], degree, k)

    for i in range(0, ngl):
        lagrange_poly_west[i,:,:,:] = lagrange_poly_east[ngl-1-i,:,:,:]
        lagrange_poly_north[:,i,:,:] = lagrange_poly_east[i,:,:,:]
        lagrange_poly_south[:,i,:,:] = lagrange_poly_east[ngl-1-i,:,:,:]
    #print(lagrange_poly[i0:iend+1,0,0,:])

    stencil = [Kmin, Kmax]
    lagrange_poly = [lagrange_poly_east, lagrange_poly_west, lagrange_poly_north, lagrange_poly_south] 

    simulation.stencil_ghost_edge = stencil
    simulation.lagrange_poly_ghost_edge = lagrange_poly

    return
