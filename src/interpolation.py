####################################################################################
#
# Module for cubed-sphere mesh interpolation routines
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
import os.path
import time
import netCDF4 as nc
import math

from constants import*
from sphgeo import*
from cs_transform import inverse_equiangular_gnomonic_map, inverse_equidistant_gnomonic_map
from plot import ll2cs_netcdf
from advection_ic  import qexact_adv

####################################################################################
# This routine receives cubeb-sphere and lat-lon grids and convert each lat-lon
# point to a cubed-sphere point searching the nearest point in the cubed sphere.
#
# Outputs:
# - i, j: arrays of dimension [Nlon, Nlat]. Stores the indexes (i,j) of each latlon
#   point in their respective panel, after being converted to a cubed sphere point
#
# - panel_list: array of dimension [Nlon, Nlat], that stores the panel of each
#   point in the latlon grid
#
# Reference: Lauritzen, P. H., Bacmeister, J. T., Callaghan, P. F., and Taylor,
#   M. A.: NCAR_Topo (v1.0): NCAR global model topography generation software
#   for unstructured grids, Geosci. Model Dev., 8, 3975–3986,
#   https://doi.org/10.5194/gmd-8-3975-2015, 2015.
####################################################################################
def ll2cs(cs_grid, latlon_grid):
    # Cartesian coordinates of the latlon grid points.
    Xll = latlon_grid.X
    Yll = latlon_grid.Y
    Zll = latlon_grid.Z

    # Indexes lists
    i = np.zeros(np.shape(Xll), dtype=np.uint32)
    j = np.zeros(np.shape(Xll), dtype=np.uint32)
    panel_list = np.zeros(np.shape(Xll), dtype=np.uint32)

    filename = griddir+cs_grid.name+'_ll2cs'+'.nc'
    #print(filename)
    if (os.path.isfile(filename)):
        #print("--------------------------------------------------------")
        #print("Loading lat-lon grid points to cubed-sphere points (for plotting) ...")
        # Open grid data
        data = nc.Dataset(filename,'r')
        i[:,:] = data['i'][:,:]
        j[:,:] = data['j'][:,:]
        panel_list[:,:] = data['panel'][:,:]
        #print("--------------------------------------------------------\n")
    else:
        # Define a grid for each transformation
        if cs_grid.projection == 'gnomonic_equiangular':
            a  = pio4
        elif cs_grid.projection == 'gnomonic_equidistant':
            a  = 1.0/np.sqrt(3.0) # Half length of the cube
        elif cs_grid.projection == 'conformal':
            a  = 1.0/np.sqrt(3.0)

        # Grid spacing
        dx = 2*a/cs_grid.N
        dy = 2*a/cs_grid.N

        # Create the grid
        [xmin, xmax, ymin, ymax] = [-a, a, -a, a]
        x = np.linspace(xmin+dx/2.0, xmax-dx/2.0, cs_grid.N)
        y = np.linspace(ymin+dy/2.0, ymax-dy/2.0, cs_grid.N)

        # This routine receives an array xx (yy) and for each
        # point in xx (yy), returns in i (j) the index of the
        # closest point in the array xx (yy).
        def find_closest_index(xx,yy):
            i = (np.floor((xx-xmin)/dx))
            j = (np.floor((yy-ymin)/dy))
            i = np.array(i, dtype=np.uint32)
            j = np.array(j, dtype=np.uint32)
            return i, j

        # Start time counting
        #print("--------------------------------------------------------")
        #print("Converting lat-lon grid points to cubed-sphere points (for plotting) ...")
        start_time = time.time()

        # Find panel - Following Lauritzen et al 2015.
        P = np.zeros( (np.shape(Xll)[0], np.shape(Xll)[1], 3) )
        P[:,:,0] = abs(Xll)
        P[:,:,1] = abs(Yll)
        P[:,:,2] = abs(Zll)
        PM = P.max(axis=2)

        # Panel 0
        mask = np.logical_and(PM == abs(Xll), Xll>0)
        panel_list[mask] = 0

        # Panel 1
        mask = np.logical_and(PM == abs(Yll), Yll>0)
        panel_list[mask] = 1

        # Panel 2
        mask = np.logical_and(PM == abs(Xll), Xll<0)
        panel_list[mask] = 2

        # Panel 3
        mask = np.logical_and(PM == abs(Yll), Yll<0)
        panel_list[mask] = 3

        # Panel 4
        mask = np.logical_and(PM == abs(Zll), Zll>0)
        panel_list[mask] = 4

        # Panel 5
        mask = np.logical_and(PM == abs(Zll), Zll<=0)
        panel_list[mask] = 5

        # Compute inverse transformation (sphere to cube) for each panel p
        for p in range(0, nbfaces):
            mask = (panel_list == p)
            if cs_grid.projection == 'gnomonic_equiangular':
                x, y = inverse_equiangular_gnomonic_map(Xll[mask], Yll[mask], Zll[mask], p)
                i[mask], j[mask] = find_closest_index(x, y)
            elif cs_grid.projection == 'gnomonic_equidistant':
                x, y = inverse_equidistant_gnomonic_map(Xll[mask], Yll[mask], Zll[mask], p)
                i[mask], j[mask] = find_closest_index(x, y)
            #elif cs_grid.projection == 'conformal':

        # Finish time counting
        elapsed_time = time.time() - start_time

        #print("Done in ","{:.2e}".format(elapsed_time),"seconds.")
        #print("--------------------------------------------------------\n")

        # Save indexes
        ll2cs_netcdf(i, j, panel_list, cs_grid)
    return i, j, panel_list

####################################################################################
# Interpolate values of Q from latlon to cubed-sphere using nearest neighbour
####################################################################################
def nearest_neighbour(Q, cs_grid, latlon_grid):
    Q_ll = np.zeros((latlon_grid.Nlon, latlon_grid.Nlat))
    for p in range(0, nbfaces):
        mask = (latlon_grid.mask==p)
        Q_ll[mask] = Q.f[latlon_grid.ix[mask], latlon_grid.jy[mask], p]
    return Q_ll

####################################################################################
# Ghost cells interpolation using Lagrange polynomials
####################################################################################
def ghost_cells_lagrange_interpolation(Q, cs_grid, transformation, simulation, interpol_method,\
                                       lagrange_poly, lagrange_poly_flipped, Kmin, Kmax):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.nghost   # Number o ghost cells
    ngl = cs_grid.nghost_left
    ngr = cs_grid.nghost_right

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    if interpol_method>=1 or interpol_method<=4:
        degree = interpol_method
        order = degree + 1
        data_values = np.zeros((ngr,N,order))

        # --------------------- Panel 0 ----------------------------
        p = 0
        north = 4
        south = 5
        east  = 1
        west  = 3

        # Interpolate ghost cells of panel 0 at east
        support_values = Q[i0:i0+ngr,j0:jend,east] # Panel 1
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[iend:iend+ngr,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[iend:iend+ngr,j0:jend,p] = Q[iend:iend+ngr,j0:jend,p] + lagrange_poly[0:ngr,0:N,l]*data_values[0:ngr,0:N,l]

        # Interpolate ghost cells of panel 0 at west
        support_values = Q[iend-ngl:iend,j0:jend, west] # Panel 3
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[0:ngl,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[0:ngl,j0:jend,p] = Q[0:ngl,j0:jend,p] + lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngl,0:N,l]

        # Interpolate ghost cells of panel 0 at north
        support_values =  np.transpose(Q[i0:iend,j0:j0+ngr, north]) # Panel 4
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[i0:iend,jend:jend+ngr,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,jend:jend+ngr,p] = Q[i0:iend,jend:jend+ngr,p] + np.transpose(lagrange_poly[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # Interpolate ghost cells of panel 0 at south
        support_values =  np.transpose(Q[i0:iend,jend-ngl:jend, south]) # Panel 5
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]
        Q[i0:iend,0:ngl,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,0:ngl,p]  = Q[i0:iend,0:ngl,p] + np.transpose(lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # --------------------- Panel 1 ----------------------------
        p = 1
        north = 4
        south = 5
        east  = 2
        west  = 0

        # Interpolate ghost cells of panel 1 at east
        support_values = Q[i0:i0+ngr,j0:jend,east] # Panel 2
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[iend:iend+ngr,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[iend:iend+ngr,j0:jend,p] = Q[iend:iend+ngr,j0:jend,p] + lagrange_poly[0:ngr,0:N,l]*data_values[0:ngr,0:N,l]

        # Interpolate ghost cells of panel 1 at west
        support_values = Q[iend-ngl:iend,j0:jend, west] # Panel 0
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[0:ngl,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[0:ngl,j0:jend,p] = Q[0:ngl,j0:jend,p] + lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngl,0:N,l]

        # Interpolate ghost cells of panel 1 at north
        support_values = Q[iend-ngl:iend,j0:jend, north] # Panel 4
        support_values = np.flip(support_values, axis=0)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[i0:iend,jend:jend+ngr,p] = 0.0
        for l in range(0, order):
            Q[i0:iend,jend:jend+ngr,p] = Q[i0:iend,jend:jend+ngr,p] + np.transpose(lagrange_poly[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # Interpolate ghost cells of panel 1 at south
        support_values = np.flip(Q[iend-ngl:iend,j0:jend, south],axis=1) # Panel 5

        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]
        Q[i0:iend,0:ngl,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,0:ngl,p]  = Q[i0:iend,0:ngl,p] + np.transpose(lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # --------------------- Panel 2 ----------------------------
        p = 2
        north = 4
        south = 5
        east  = 3
        west  = 1

        # Interpolate ghost cells of panel 2 at east
        support_values = Q[i0:i0+ngr,j0:jend,east] # Panel 3
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[iend:iend+ngr,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[iend:iend+ngr,j0:jend,p] = Q[iend:iend+ngr,j0:jend,p] + lagrange_poly[0:ngr,0:N,l]*data_values[0:ngr,0:N,l]

        # Interpolate ghost cells of panel 2 at west
        support_values = Q[iend-ngl:iend,j0:jend, west] # Panel 1
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[0:ngl,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[0:ngl,j0:jend,p] = Q[0:ngl,j0:jend,p] + lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngl,0:N,l]

        # Interpolate ghost cells of panel 2 at north
        support_values = Q[i0:iend,jend-ngr:jend, north] # Panel 4
        support_values = np.flip(support_values)
        support_values = np.transpose(support_values)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[i0:iend,jend:jend+ngr,p] = 0.0
        for l in range(0, order):
            Q[i0:iend,jend:jend+ngr,p] = Q[i0:iend,jend:jend+ngr,p] + np.transpose(lagrange_poly[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # Interpolate ghost cells of panel 2 at south
        support_values = np.transpose(Q[i0:iend,j0:j0+ngr,south]) # Panel 5
        support_values = np.flip(support_values)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]
        Q[i0:iend,0:ngl,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,0:ngl,p]  = Q[i0:iend,0:ngl,p] + np.transpose(lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # --------------------- Panel 3 ----------------------------
        p = 3
        north = 4
        south = 5
        east  = 0
        west  = 2

        # Interpolate ghost cells of panel 3 at east
        support_values = Q[i0:i0+ngr,j0:jend,east] # Panel 3
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[iend:iend+ngr,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[iend:iend+ngr,j0:jend,p] = Q[iend:iend+ngr,j0:jend,p] + lagrange_poly[0:ngr,0:N,l]*data_values[0:ngr,0:N,l]

        # Interpolate ghost cells of panel 3 at west
        support_values = Q[iend-ngl:iend,j0:jend, west] # Panel 1
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[0:ngl,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[0:ngl,j0:jend,p] = Q[0:ngl,j0:jend,p] + lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngl,0:N,l]

        # Interpolate ghost cells of panel 3 at north
        support_values = np.flip(Q[i0:i0+ngr,j0:jend, north],axis=1) # Panel 4
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[i0:iend,jend:jend+ngr,p] = 0.0
        for l in range(0, order):
            Q[i0:iend,jend:jend+ngr,p] = Q[i0:iend,jend:jend+ngr,p] + np.transpose(lagrange_poly[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # Interpolate ghost cells of panel 3 at south
        support_values = (Q[i0:i0+ngr,j0:jend,south]) # Panel 5
        support_values = np.flip(support_values,axis=0)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]
        Q[i0:iend,0:ngl,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,0:ngl,p]  = Q[i0:iend,0:ngl,p] + np.transpose(lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # --------------------- Panel 4 ----------------------------
        p = 4
        north = 2
        south = 0
        east  = 1
        west  = 3

        # Interpolate ghost cells of panel 4 at east
        support_values = np.transpose(Q[i0:iend,jend-ngl:jend,east]) # Panel 3
        support_values = np.flip(support_values,axis=0)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[iend:iend+ngr,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[iend:iend+ngr,j0:jend,p] = Q[iend:iend+ngr,j0:jend,p] + lagrange_poly[0:ngr,0:N,l]*data_values[0:ngr,0:N,l]

        # Interpolate ghost cells of panel 4 at west
        support_values = np.transpose(Q[i0:iend,jend-ngr:jend, west]) # Panel 1
        support_values = np.flip(support_values,axis=1)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[0:ngl,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[0:ngl,j0:jend,p] = Q[0:ngl,j0:jend,p] + lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngl,0:N,l]

        # Interpolate ghost cells of panel 4 at north
        support_values = Q[i0:iend,jend-ngr:jend, north] # Panel 2
        support_values = np.flip(support_values)
        support_values = np.transpose(support_values)

        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[i0:iend,jend:jend+ngr,p] = 0.0
        for l in range(0, order):
            Q[i0:iend,jend:jend+ngr,p] = Q[i0:iend,jend:jend+ngr,p] + np.transpose(lagrange_poly[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # Interpolate ghost cells of panel 4 at south
        support_values = np.transpose(Q[i0:iend,jend-ngl:jend,south]) # Panel 0
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]
        Q[i0:iend,0:ngl,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,0:ngl,p]  = Q[i0:iend,0:ngl,p] + np.transpose(lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # --------------------- Panel 5 ----------------------------
        p = 5
        north = 0
        south = 2
        east  = 1
        west  = 3

        # Interpolate ghost cells of panel 5 at east
        support_values = np.transpose(Q[i0:iend,j0:j0+ngr,east]) # Panel 1
        support_values = np.flip(support_values,axis=1)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[iend:iend+ngr,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[iend:iend+ngr,j0:jend,p] = Q[iend:iend+ngr,j0:jend,p] + lagrange_poly[0:ngr,0:N,l]*data_values[0:ngr,0:N,l]

        # Interpolate ghost cells of panel 5 at west
        support_values = np.transpose(Q[i0:iend,j0:j0+ngr, west]) # Panel 3
        support_values = np.flip(support_values,axis=0)
        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[0:ngl,j0:jend,p] = 0.0
        for l in range(0, order):
            Q[0:ngl,j0:jend,p] = Q[0:ngl,j0:jend,p] + lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngl,0:N,l]

        # Interpolate ghost cells of panel 5 at north
        support_values = np.transpose(Q[i0:iend,j0:j0+ngr, north]) # Panel 0

        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]

        Q[i0:iend,jend:jend+ngr,p] = 0.0
        for l in range(0, order):
            Q[i0:iend,jend:jend+ngr,p] = Q[i0:iend,jend:jend+ngr,p] + np.transpose(lagrange_poly[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

        # Interpolate ghost cells of panel 4 at south
        support_values = Q[i0:iend,j0:j0+ngr,south] # Panel 2
        support_values = np.flip(support_values)
        support_values = np.transpose(support_values)

        for k in range(0, N):
            data_values[:,k,:] = support_values[:,Kmin[0,k]:Kmax[0,k]]
        Q[i0:iend,0:ngl,p] = 0.0

        for l in range(0, order):
            Q[i0:iend,0:ngl,p]  = Q[i0:iend,0:ngl,p] + np.transpose(lagrange_poly_flipped[0:ngl,0:N,l]*data_values[0:ngr,0:N,l])

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
def lagrange_poly_ghostcells(cs_grid, transformation, degree):
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
    xc = np.linspace(x_min+dx/2.0, x_max-dx/2.0, N) # Centers
    order = degree+1

    p = 0
    north = 4
    south = 5
    east  = 1
    west  = 3

    # Ghost cells at east
    X_ghost = cs_grid.centers.X[iend:iend+ngr,j0:jend,p]
    Y_ghost = cs_grid.centers.Y[iend:iend+ngr,j0:jend,p]
    Z_ghost = cs_grid.centers.Z[iend:iend+ngr,j0:jend,p]
    x_ghost, y_ghost = inverse(X_ghost, Y_ghost, Z_ghost, east)

    # Support points
    X = cs_grid.centers.X[i0:i0+ngr,j0:jend,east]
    Y = cs_grid.centers.Y[i0:i0+ngr,j0:jend,east]
    Z = cs_grid.centers.Z[i0:i0+ngr,j0:jend,east]
    x, y = inverse(X, Y, Z, east)

    # Interpolation
    K = (np.floor((y_ghost-xc[0])/dx)).astype(int)
    Kmax = np.minimum(K + order, N-1).astype(int)
    Kmin = np.maximum(Kmax-order, 0).astype(int)
    support_points = y
    data_nodes  = np.zeros((ngr,N,order))
    lagrange_poly = np.zeros((ngr, N, order))
    lagrange_poly_flipped = np.zeros((ngr, N, order))

    # Compute the Lagrange basis at ghost cells points
    for k in range(0, N):
        data_nodes[:,k,:]  = support_points[:,Kmin[0,k]:Kmax[0,k]]
        for g in range(0, ngr):
            for l in range(0, order):
                lagrange_poly[g,k,l] = lagrange_basis(y_ghost[g,k], data_nodes[g,k,:], degree, l)
    lagrange_poly_flipped[0:ngl,0:N,:] = np.flip(lagrange_poly[0:ngl,0:N,:],axis=0)

    return lagrange_poly, lagrange_poly_flipped, Kmin, Kmax
