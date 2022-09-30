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

from constants import*
from sphgeo import*
from cs_transform import inverse_equiangular_gnomonic_map, inverse_equidistant_gnomonic_map, inverse_conformal_map
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
        print("--------------------------------------------------------")
        print("Converting lat-lon grid points to cubed-sphere points (for plotting) ...")
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

        print("Done in ","{:.2e}".format(elapsed_time),"seconds.")
        print("--------------------------------------------------------\n")

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
# Ghost cells interpolation
####################################################################################
def ghost_cells_interpolation(Q, cs_grid, t, simulation):
    N  = cs_grid.N       # Number of cells in x direction
    nghost = cs_grid.nghost   # Number o ghost cells

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    #print(np.shape(Q[i0:iend,0:j0,:]))
    #print(np.shape(Q[i0:iend,jend:N+1+nghost,:]))

    # Get center position in lat/lon system
    center_lon = cs_grid.centers.lon
    center_lat = cs_grid.centers.lat

    Q[0:i0,:,:] = qexact_adv(center_lon[0:i0,:,:], center_lat[0:i0,:,:], t, simulation)

    Q[iend:N+1+nghost,:,:] = qexact_adv(center_lon[iend:N+1+nghost,:,:], center_lat[iend:N+1+nghost,:,:], t, simulation)

    Q[:,0:j0,:] = qexact_adv(center_lon[:,0:j0,:], center_lat[:,0:j0,:], t, simulation)

    Q[:,jend:N+1+nghost,:] = qexact_adv(center_lon[:,jend:N+1+nghost,:], center_lat[:,jend:N+1+nghost,:], t, simulation)
