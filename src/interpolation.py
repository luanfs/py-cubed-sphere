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
from cs_transform   import inverse_equiangular_gnomonic_map, inverse_equidistant_gnomonic_map, inverse_conformal_map
from plot           import ll2cs_netcdf
from advection_ic   import qexact_adv
from halo_data      import get_halo_data_interpolation, get_halo_data_interpolation_NS, get_halo_data_interpolation_WE

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
        a = cs_grid.a

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

        i0, iend = cs_grid.i0, cs_grid.iend
        # Compute inverse transformation (sphere to cube) for each panel p
        for p in range(0, nbfaces):
            mask = (panel_list == p)
            if cs_grid.projection == 'gnomonic_equiangular':
                x, y = inverse_equiangular_gnomonic_map(Xll[mask], Yll[mask], Zll[mask], p)
                i[mask], j[mask] = find_closest_index(x, y)
            elif cs_grid.projection == 'gnomonic_equidistant':
                x, y = inverse_equidistant_gnomonic_map(Xll[mask], Yll[mask], Zll[mask], p)
                i[mask], j[mask] = find_closest_index(x, y)

        #exit()
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
# Ghost cell center interpolation using Lagrange polynomials
####################################################################################
def ghost_cell_pc_lagrange_interpolation(Q, cs_grid, simulation):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Lagrange polynomials and its stencils
    lagrange_poly, stencil = simulation.lagrange_poly_ghost_pc, simulation.stencil_ghost_pc
    Kmin, Kmax = stencil[0], stencil[1]

    # Order
    interpol_degree = simulation.degree

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Get halo data
    halodata = get_halo_data_interpolation(Q, cs_grid)
    halo_data_east  = halodata[0]
    halo_data_west  = halodata[1]
    halo_data_north = halodata[2]
    halo_data_south = halodata[3]

    # Interpolation indexes
    Kmin_east , Kmax_east  = Kmin[0], Kmax[0]
    Kmin_west , Kmax_west  = Kmin[1], Kmax[1]
    Kmin_north, Kmax_north = Kmin[2], Kmax[2]
    Kmin_south, Kmax_south = Kmin[3], Kmax[3]

    # Get Lagrange polynomials
    lagrange_poly_east  = lagrange_poly[0]
    lagrange_poly_west  = lagrange_poly[1]
    lagrange_poly_north = lagrange_poly[2]
    lagrange_poly_south = lagrange_poly[3]

    if interpol_degree>=0:
        degree = interpol_degree
        order = degree + 1
        halo_data_x = np.zeros((ngr, N+ng, order)) # Data used in the interpolation
        halo_data_ghost_x = np.zeros((ngr, N+ng))  # Interpolated data at ghost cells
        halo_data_y = np.zeros((N+ng, ngr,order)) # Data used in the interpolation
        halo_data_ghost_y = np.zeros((N+ng, ngr, N+ng))  # Interpolated data at ghost cells

        for p in range(0, nbfaces):
            # Interpolate ghost cells of panel p at east
            support_values = halo_data_east[:,:,p]
            for g in range(0, ngl):
                for k in range(0, N+ng):
                    halo_data_x[g,k,:] = support_values[g,Kmin_east[g,k]:Kmax_east[g,k]+1]

            interpolation_data = halo_data_x*lagrange_poly_east

            halo_data_ghost_x = np.sum(interpolation_data[:,:,:], axis=2)

            Q[iend:iend+ngr,j0:jend,p] = halo_data_ghost_x[:,j0:jend]

            # Interpolate ghost cells of panel p at west
            support_values = halo_data_west[:,:,p]

            for g in range(0, ngl):
                for k in range(0, N+ng):
                    halo_data_x[g,k,:] = support_values[g,Kmin_west[g,k]:Kmax_west[g,k]+1]

            interpolation_data = halo_data_x*lagrange_poly_west

            halo_data_ghost_x = np.sum(interpolation_data[:,:,:], axis=2)

            Q[0:i0,j0:jend,p] = halo_data_ghost_x[:,j0:jend]

            # Interpolate ghost cells of panel p at north
            support_values = halo_data_north[:,:,p]

            for g in range(0, ngl):
                for k in range(0, N+ng):
                    halo_data_y[k,g,:] = support_values[Kmin_north[k,g]:Kmax_north[k,g]+1,g]
            interpolation_data = halo_data_y*lagrange_poly_north

            halo_data_ghost_y = np.sum(interpolation_data[:,:,:], axis=2)

            Q[i0:iend,jend:jend+ngr,p] = halo_data_ghost_y[i0:iend,:]

            # Interpolate ghost cells of panel p at south
            support_values = halo_data_south[:,:,p]

            for g in range(0, ngl):
                for k in range(0, N+ng):
                    halo_data_y[k,g,:] = support_values[Kmin_south[k,g]:Kmax_south[k,g]+1,g]
            interpolation_data = halo_data_y*lagrange_poly_south

            halo_data_ghost_y = np.sum(interpolation_data[:,:,:], axis=2)

            Q[i0:iend,0:i0,p] = halo_data_ghost_y[i0:iend,:]

        # Now let us interpolate the remaing ghost cell data
        # Get halo data
        halodata = get_halo_data_interpolation(Q, cs_grid)
        halo_data_east  = halodata[0]
        halo_data_west  = halodata[1]
        halo_data_north = halodata[2]
        halo_data_south = halodata[3]

        halo_data = np.zeros((ngr, ngr, order)) # Data used in the interpolation
        halo_data_ghost = np.zeros((ngr, ngr))  # Interpolated data at ghost cells


        for p in range(0, nbfaces):
            # Interpolate ghost cells of panel p at east - cells in [iend:iend+ngr,jend:jend+ngr,p]
            support_values = halo_data_east[:,:,p]

            for g in range(0, ngl):
                for j in range(0, ngl):
                    halo_data[g,j,:] = support_values[g,Kmin_east[g,j+jend]:Kmax_east[g,j+jend]+1]

            interpolation_data = halo_data*lagrange_poly_east[:,jend:jend+ngr,:]

            halo_data_ghost = np.sum(interpolation_data[:,:,:], axis=2)

            Q[iend:iend+ngr,jend:jend+ngr,p] = halo_data_ghost[:,:]

            # Interpolate ghost cells of panel p at east - cells in [iend:iend+ngr,0:j0,p]
            support_values = halo_data_east[:,:,p]

            for g in range(0, ngl):
                for j in range(0, ngl):
                    halo_data[g,j,:] = support_values[g,Kmin_east[g,j]:Kmax_east[g,j]+1]

            interpolation_data = halo_data*lagrange_poly_east[:,0:j0,:]

            halo_data_ghost = np.sum(interpolation_data[:,:,:], axis=2)

            Q[iend:iend+ngr,0:j0,p] = halo_data_ghost[:,:]

            # Interpolate ghost cells of panel p at west - cells in [0:i0,jend:jend+ngr,p]
            support_values = halo_data_west[:,:,p]

            for g in range(0, ngl):
                for j in range(0, ngr):
                    halo_data[g,j,:] = support_values[g,Kmin_west[g,j+jend]:Kmax_west[g,j+jend]+1]

            interpolation_data = halo_data*lagrange_poly_west[:,jend:jend+ngr,:]

            halo_data_ghost = np.sum(interpolation_data[:,:,:], axis=2)

            Q[0:i0,jend:jend+ngr,p] = halo_data_ghost[:,:]

            # Interpolate ghost cells of panel p at west - cells in [0:i0,0:j0,p]
            support_values = halo_data_west[:,:,p]
            halo_data_ghost = np.zeros((ngl, ngr))  # Interpolated data at ghost cells

            for g in range(0, ngl):
                for j in range(0, ngr):
                    halo_data[g,j,:] = support_values[g,Kmin_west[g,j]:Kmax_west[g,j]+1]

            interpolation_data = halo_data*lagrange_poly_west[:,0:j0,:]

            halo_data_ghost = np.sum(interpolation_data[:,:,:], axis=2)

            Q[0:i0,0:j0,p] = halo_data_ghost[:,:]


####################################################################################
# Ghost cells interpolation using Lagrange polynomials - consider only ghost
# cells at south and north
####################################################################################
def ghost_cells_lagrange_interpolation_NS(Qx, Qy, cs_grid, simulation,\
                                       lagrange_poly, stencil):
    N   = cs_grid.N        # Number of cells in x direction
    ng  = cs_grid.ng   # Number o ghost cells
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    Kmin, Kmax = stencil[0], stencil[1]
    # Order
    interpol_degree = simulation.degree

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Get halo data
    halodata = get_halo_data_interpolation_NS(Qx, Qy, cs_grid)
    halo_data_north = halodata[0]
    halo_data_south = halodata[1]

    # Interpolation stencil
    Kmin_north, Kmax_north = Kmin[2], Kmax[2]
    Kmin_south, Kmax_south = Kmin[3], Kmax[3]

    # Get Lagrange polynomials
    lagrange_poly_north = lagrange_poly[2]
    lagrange_poly_south = lagrange_poly[3]

    if interpol_degree>=0:
        degree = interpol_degree
        order = degree + 1
        halo_data_x = np.zeros((N+ng, ngr, order)) # Data used in the interpolation
        halo_data_ghost_x = np.zeros((N+ng, ngr))  # Interpolated data at ghost cells

        for p in range(0, nbfaces):
            # Interpolate ghost cells of panel p at north
            support_values = halo_data_north[:,:,p]
            for g in range(0, ngl):
                for k in range(i0, iend):
                    halo_data_x[k,g,:] = support_values[Kmin_north[k,g]:Kmax_north[k,g]+1,g]

            interpolation_data = halo_data_x[i0:iend,:,:]*lagrange_poly_north[i0:iend,:,:]
            halo_data_ghost_x[i0:iend,:] = np.sum(interpolation_data[:,:,:], axis=2)
            Qx[i0:iend,jend:jend+ngr,p] = halo_data_ghost_x[i0:iend,:]

            # Interpolate ghost cells of panel p at south
            support_values = halo_data_south[:,:,p]
            for g in range(0, ngl):
                for k in range(i0, iend):
                    halo_data_x[k,g,:] = support_values[Kmin_south[k,g]:Kmax_south[k,g]+1,g]

            interpolation_data = halo_data_x[i0:iend,:,:]*lagrange_poly_south[i0:iend,:,:]
            halo_data_ghost_x[i0:iend,:] = np.sum(interpolation_data[:,:,:], axis=2)
            Qx[i0:iend,0:j0,p] = halo_data_ghost_x[i0:iend,:]

####################################################################################
# Ghost cells interpolation using Lagrange polynomials - consider only ghost
# cells at west and east adjacent panels
####################################################################################
def ghost_cells_lagrange_interpolation_WE(Qy, Qx, cs_grid, simulation,\
                                       lagrange_poly, stencil):
    N   = cs_grid.N   # Number of cells in x direction
    ng  = cs_grid.ng  # Number o ghost cells
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    Kmin, Kmax = stencil[0], stencil[1]

    # Order
    interpol_degree = simulation.degree

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Get halo data
    halodata = get_halo_data_interpolation_WE(Qy, Qx, cs_grid)
    halo_data_east  = halodata[0]
    halo_data_west  = halodata[1]

    # Interpolation stencil
    Kmin_east , Kmax_east  = Kmin[0], Kmax[0]
    Kmin_west , Kmax_west  = Kmin[1], Kmax[1]

    # Get Lagrange polynomials
    lagrange_poly_east  = lagrange_poly[0]
    lagrange_poly_west  = lagrange_poly[1]

    if interpol_degree>=0:
        degree = interpol_degree
        order = degree + 1
        halo_data_y = np.zeros((ngr, N+ng, order)) # Data used in the interpolation
        halo_data_ghost_y = np.zeros((ngr, N+ng))  # Interpolated data at ghost cells

        for p in range(0, nbfaces):
            # Interpolate ghost cells of panel p at east
            support_values = halo_data_east[:,:,p]
            for g in range(0, ngl):
                for k in range(j0, jend):
                    halo_data_y[g,k,:] = support_values[g,Kmin_east[g,k]:Kmax_east[g,k]+1]

            interpolation_data = halo_data_y[:,j0:jend,:]*lagrange_poly_east[:,j0:jend,:]
            halo_data_ghost_y[:,j0:jend] = np.sum(interpolation_data[:,:,:], axis=2)
            Qy[iend:iend+ngr,j0:jend,p] = halo_data_ghost_y[:,j0:jend]

            # Interpolate ghost cells of panel p at west
            support_values = halo_data_west[:,:,p]
            for g in range(0, ngl):
                for k in range(j0, jend):
                    halo_data_y[g,k,:] = support_values[g,Kmin_west[g,k]:Kmax_west[g,k]+1]

            interpolation_data = halo_data_y[:,j0:jend,:]*lagrange_poly_west[:,j0:jend,:]
            halo_data_ghost_y[:,j0:jend] = np.sum(interpolation_data[:,:,:], axis=2)
            Qy[0:i0,j0:jend,p] = halo_data_ghost_y[:,j0:jend]

####################################################################################
# This routine set the ghost cell values equal to the adjacent panel values
# ignoring the coordinate system discontinuity
####################################################################################
def ghost_cells_adjacent_panels(Qx, Qy, cs_grid, simulation):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend
    ngl = cs_grid.ngl

    # Get halo data
    halodata = get_halo_data_interpolation_WE(Qx, Qy, cs_grid)
    halo_data_east  = halodata[0]
    halo_data_west  = halodata[1]
    halodata = get_halo_data_interpolation_NS(Qx, Qy, cs_grid)
    halo_data_north = halodata[0]
    halo_data_south = halodata[1]

    # Set the values of the adjacent panel for ghost cells
    Qy[0:i0,:,:]  = halo_data_west[:,:,:]
    Qy[iend:,:,:] = halo_data_east[:,:,:]
    Qx[:,0:j0,:]  = halo_data_south[:,:,:]
    Qx[:,jend:,:] = halo_data_north[:,:,:]

####################################################################################
# This routine interpolates the wind given in a C grid edges
# to the cell pc (including ghost cell centers)
# using Lagrange polynomials
####################################################################################
def wind_edges2center_lagrange_interpolation(U_pc, U_pu, U_pv, cs_grid, simulation):
    # Parameters of the grid
    N   = cs_grid.N
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr
    dx  = cs_grid.dx

    lagrange_poly_U = simulation.lagrange_poly_edge
    stencil_U = simulation.stencil_edge
    lagrange_poly_ghost_pc = simulation.stencil_ghost_pc
    stencil_ghost_pc =  simulation.lagrange_poly_ghost_pc

    # Order
    degree = simulation.degree
    order  = degree + 1

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Stencil indexes
    Kmin, Kmax = stencil_U[0], stencil_U[1]

    # Lagrange polynomials for the wind interpolation
    lagrange_poly_u, lagrange_poly_v = lagrange_poly_U[0], lagrange_poly_U[1]

    # First, we interpolate the wind at pc only at the interior of each panel
    interpd_u = np.zeros((N+ng, N+ng, nbfaces, order)) # Data used in the interpolation
    interpd_v = np.zeros((N+ng, N+ng, nbfaces, order)) # Data used in the interpolation

    # Get the contravariant wind data from the C grid for interpolation 
    for i in range(i0, iend):
        for j in range(j0, jend):
            for p in range(0,nbfaces):
                interpd_u[i,j,p,:] = U_pu.ucontra[Kmin[i]:Kmax[i]+1,j,p]
                interpd_v[j,i,p,:] = U_pv.vcontra[j,Kmin[i]:Kmax[i]+1,p]

    # Lagrange interpolation
    U_pc.ucontra[i0:iend,j0:jend,:] = np.sum(interpd_u[i0:iend,j0:jend,:,:]*lagrange_poly_u[i0:iend,j0:jend,:,:], axis = 3)  
    U_pc.vcontra[i0:iend,j0:jend,:] = np.sum(interpd_v[i0:iend,j0:jend,:,:]*lagrange_poly_v[i0:iend,j0:jend,:,:], axis = 3)  

    # Convert from contravariant to latlon
    U_pc.ulon[i0:iend,j0:jend,:], U_pc.vlat[i0:iend,j0:jend,:] = contravariant_to_latlon\
    (U_pc.ucontra[i0:iend,j0:jend,:], U_pc.vcontra[i0:iend,j0:jend,:],\
     cs_grid.prod_ex_elon_pc[i0:iend,j0:jend,:], cs_grid.prod_ex_elat_pc[i0:iend,j0:jend,:],\
     cs_grid.prod_ey_elon_pc[i0:iend,j0:jend,:], cs_grid.prod_ey_elat_pc[i0:iend,j0:jend,:])

    if cs_grid.projection == 'gnomonic_equiangular':
        # Now let us interpolate the latlon wind to the center of ghost cells
        ghost_cell_pc_lagrange_interpolation(U_pc.ulon, cs_grid, simulation)
        ghost_cell_pc_lagrange_interpolation(U_pc.vlat, cs_grid, simulation)

####################################################################################
# This routine interpolates the wind (latlon) from cell centers
# to ghost cell edges
####################################################################################
def wind_center2ghostedge_lagrange_interpolation(U_pc, U_pu, U_pv, cs_grid, simulation):
    # Parameters of the grid
    N   = cs_grid.N
    ng  = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr
    dx  = cs_grid.dx

    lagrange_poly_ghost_edge = simulation.lagrange_poly_ghost_edge 
    stencil_ghost_edge = simulation.stencil_ghost_edge 

    # Order
    degree = simulation.degree
    order  = degree + 1

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Stencil indexes
    Kmin, Kmax = stencil_ghost_edge[0], stencil_ghost_edge[1]
    lagrange_poly_east = lagrange_poly_ghost_edge[0]
    lagrange_poly_west = lagrange_poly_ghost_edge[1]
    lagrange_poly_north = lagrange_poly_ghost_edge[2]
    lagrange_poly_south = lagrange_poly_ghost_edge[3]

    # Get interpolation data 
    interpd_u_east = np.zeros((ngl, N+ng, nbfaces, order))
    interpd_v_east = np.zeros((ngl, N+ng, nbfaces, order))
    interpd_u_west = np.zeros((ngl, N+ng, nbfaces, order))
    interpd_v_west = np.zeros((ngl, N+ng, nbfaces, order))
    interpd_u_north = np.zeros((N+ng, ngl, nbfaces, order))
    interpd_v_north = np.zeros((N+ng, ngl, nbfaces, order))
    interpd_u_south = np.zeros((N+ng, ngl, nbfaces, order))
    interpd_v_south = np.zeros((N+ng, ngl, nbfaces, order))

    for j in range(j0, jend+1):
        for i in range(0, ngl):
            for p in range(0, nbfaces):
                interpd_u_east[i,j,p,:] = U_pc.ulon[iend+i,Kmin[j]:Kmax[j]+1,p]
                interpd_v_east[i,j,p,:] = U_pc.vlat[iend+i,Kmin[j]:Kmax[j]+1,p]
                interpd_u_west[i,j,p,:] = U_pc.ulon[i,Kmin[j]:Kmax[j]+1,p]
                interpd_v_west[i,j,p,:] = U_pc.vlat[i,Kmin[j]:Kmax[j]+1,p]
                interpd_u_north[j,i,p,:] = U_pc.ulon[Kmin[j]:Kmax[j]+1,iend+i,p]
                interpd_v_north[j,i,p,:] = U_pc.vlat[Kmin[j]:Kmax[j]+1,iend+i,p]
                interpd_u_south[j,i,p,:] = U_pc.ulon[Kmin[j]:Kmax[j]+1,i,p]
                interpd_v_south[j,i,p,:] = U_pc.vlat[Kmin[j]:Kmax[j]+1,i,p]

    # Interpolate from ghost cell center to ghost cell edges in east layer
    U_pv.ulon[iend:,j0:jend+1,:] = np.sum(interpd_u_east[0:,j0:jend+1,:,:]*lagrange_poly_east[0:,j0:jend+1,:,:],axis=3)
    U_pv.vlat[iend:,j0:jend+1,:] = np.sum(interpd_v_east[0:,j0:jend+1,:,:]*lagrange_poly_east[0:,j0:jend+1,:,:],axis=3)

    # Interpolate from ghost cell center to ghost cell edges in west layer
    U_pv.ulon[:i0,j0:jend+1,:] = np.sum(interpd_u_west[0:,j0:jend+1,:,:]*lagrange_poly_west[0:,j0:jend+1,:,:],axis=3)
    U_pv.vlat[:i0,j0:jend+1,:] = np.sum(interpd_v_west[0:,j0:jend+1,:,:]*lagrange_poly_west[0:,j0:jend+1,:,:],axis=3)

    # Interpolate from ghost cell center to ghost cell edges in north layer
    U_pu.ulon[i0:iend+1,jend:,:] = np.sum(interpd_u_north[i0:iend+1,0:,:]*lagrange_poly_north[i0:iend+1,0:,:],axis=3)
    U_pu.vlat[i0:iend+1,jend:,:] = np.sum(interpd_v_north[i0:iend+1,0:,:]*lagrange_poly_north[i0:iend+1,0:,:],axis=3)

    # Interpolate from ghost cell center to ghost cell edges in south layer
    U_pu.ulon[i0:iend+1,:j0,:] = np.sum(interpd_u_south[i0:iend+1,0:,:]*lagrange_poly_south[i0:iend+1,0:,:],axis=3)
    U_pu.vlat[i0:iend+1,:j0,:] = np.sum(interpd_v_south[i0:iend+1,0:,:]*lagrange_poly_south[i0:iend+1,0:,:],axis=3)

    # Convert latlon to contravariant
    U_pv.ucontra[iend:,j0:jend+1,:], U_pv.vcontra[iend:,j0:jend+1,:] =\
    latlon_to_contravariant(U_pv.ulon[iend:,j0:jend+1,:], U_pv.vlat[iend:,j0:jend+1,:],\
                            cs_grid.prod_ex_elon_pv[iend:,j0:jend+1,:], cs_grid.prod_ex_elat_pv[iend:,j0:jend+1,:],\
                            cs_grid.prod_ey_elon_pv[iend:,j0:jend+1,:], cs_grid.prod_ey_elat_pv[iend:,j0:jend+1,:],\
                            cs_grid.determinant_ll2contra_pv[iend:,j0:jend+1,:])

    U_pv.ucontra[:i0,j0:jend+1,:], U_pv.vcontra[:i0,j0:jend+1,:] =\
    latlon_to_contravariant(U_pv.ulon[:i0,j0:jend+1,:], U_pv.vlat[:i0,j0:jend+1,:],\
                            cs_grid.prod_ex_elon_pv[:i0,j0:jend+1,:], cs_grid.prod_ex_elat_pv[:i0,j0:jend+1,:],\
                            cs_grid.prod_ey_elon_pv[:i0,j0:jend+1,:], cs_grid.prod_ey_elat_pv[:i0,j0:jend+1,:],\
                            cs_grid.determinant_ll2contra_pv[:i0,j0:jend+1,:])

    U_pu.ucontra[i0:iend+1,jend:,:], U_pu.vcontra[i0:iend+1,jend:,:] =\
    latlon_to_contravariant(U_pu.ulon[i0:iend+1,jend:,:], U_pu.vlat[i0:iend+1,jend:,:],\
                            cs_grid.prod_ex_elon_pu[i0:iend+1,jend:,:], cs_grid.prod_ex_elat_pu[i0:iend+1,jend:,:],\
                            cs_grid.prod_ey_elon_pu[i0:iend+1,jend:,:], cs_grid.prod_ey_elat_pu[i0:iend+1,jend:,:],\
                            cs_grid.determinant_ll2contra_pu[i0:iend+1,jend:,:])

    U_pu.ucontra[i0:iend+1,:j0,:], U_pu.vcontra[i0:iend+1,:j0,:] =\
    latlon_to_contravariant(U_pu.ulon[i0:iend+1,:j0,:], U_pu.vlat[i0:iend+1,:j0,:],\
                            cs_grid.prod_ex_elon_pu[i0:iend+1,:j0,:], cs_grid.prod_ex_elat_pu[i0:iend+1,:j0,:],\
                            cs_grid.prod_ey_elon_pu[i0:iend+1,:j0,:], cs_grid.prod_ey_elat_pu[i0:iend+1,:j0,:],\
                            cs_grid.determinant_ll2contra_pu[i0:iend+1,:j0,:])


