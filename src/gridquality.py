####################################################################################
#
# This module contains all the routines needed to compute measures
# of grid quality
#
# Luan da Fonseca Santos - February 2022
####################################################################################

# Imports
import numpy as np
from cs_datastruct import scalar_field, latlon_grid
from plot          import plot_scalar_field, open_netcdf4
from interpolation import ll2cs, nearest_neighbour
from sphgeo import*
from constants import*

####################################################################################
# This routine compute grid quality measures and save it as a netcdf file
####################################################################################
def grid_quality(cs_grid, ll_grid, map_projection):
    print("--------------------------------------------------------")
    print("Grid quality testing\n")

    # Computation of some distance that shall be needed
    N = cs_grid.N
    nghost = cs_grid.nghost
    # Compute the geodesic distance of cell edges in x direction
    # Given points
    p1 = [cs_grid.vertices.X[0:N+nghost,:,:], cs_grid.vertices.Y[0:N+nghost  ,:,:], cs_grid.vertices.Z[0:N+nghost  ,:,:]]
    p2 = [cs_grid.vertices.X[1:N+nghost+1,:,:], cs_grid.vertices.Y[1:N+nghost+1,:,:], cs_grid.vertices.Z[1:N+nghost+1,:,:]]

    # Reshape
    p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost+1)*nbfaces))
    p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost+1)*nbfaces))

    # Compute arclen
    d = arclen(p1, p2)
    d = np.reshape(d,(N+nghost,N+nghost+1,nbfaces))
    length_x = d

    # Compute the geodesic distance of cell edges in y direction
    # Given points
    p1 = [cs_grid.vertices.X[:,0:N+nghost  ,:], cs_grid.vertices.Y[:,0:N+nghost  ,:], cs_grid.vertices.Z[:,0:N+nghost  ,:]]
    p2 = [cs_grid.vertices.X[:,1:N+nghost+1,:], cs_grid.vertices.Y[:,1:N+nghost+1,:], cs_grid.vertices.Z[:,1:N+nghost+1,:]]

    # Reshape
    p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost+1)*nbfaces))
    p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost+1)*nbfaces))

    # Compute arclen
    d = arclen(p1,p2)
    d = np.reshape(d,(N+nghost+1,N+nghost,nbfaces))
    length_y = d

    # Compute the quality measures
    area          = areas(cs_grid)
    mean_length   = mean_lengths(cs_grid, length_x, length_y)
    distortion    = distortions (cs_grid, length_x, length_y)
    alignment     = alignment_index(cs_grid, length_x, length_y)
    metric_tensor = metrictensor(cs_grid)

    # Interpolate to the latlon grid
    area_ll          = nearest_neighbour(area, cs_grid, ll_grid)
    mean_length_ll   = nearest_neighbour(mean_length, cs_grid, ll_grid)
    distortion_ll    = nearest_neighbour(distortion, cs_grid, ll_grid)
    alignment_ll     = nearest_neighbour(alignment, cs_grid, ll_grid)
    metric_tensor_ll = nearest_neighbour(metric_tensor, cs_grid, ll_grid)

    # Create list of fields
    fields_ll = [area_ll, mean_length_ll, distortion_ll, alignment_ll, metric_tensor_ll]
    fields_cs = [area   , mean_length   , distortion   , alignment   , metric_tensor   ]

    # Netcdf file
    netcdf_name = cs_grid.name+'_grid_quality'
    netcdf_data = open_netcdf4(fields_cs, [0], netcdf_name)

    # Plot the fields
    for l in range(0,len(fields_ll)):
        plot_scalar_field(fields_ll[l], fields_cs[l].name, cs_grid, ll_grid, map_projection)
        netcdf_data[fields_cs[l].name][:,:,0] = fields_ll[l][:,:]
        exit()
    #Close netcdf file
    netcdf_data.close()
    print("A netcdf file has been created in ", datadir+netcdf_name+'.nc')
    print("--------------------------------------------------------\n")

####################################################################################
# Compute the cell areas considering the earth radius
####################################################################################
def areas(grid):
    # Interior cells index (we are ignoring ghost cells)
    i0   = grid.i0
    iend = grid.iend
    j0   = grid.j0
    jend = grid.jend
    areas = scalar_field(grid, 'areas', 'center')
    areas.f = grid.areas[i0:iend,j0:jend,:]*erad*erad/10**6
    print(np.amin(areas.f), np.amax(areas.f))
    #areas.f = np.sqrt(areas.f)
    #print((np.sum(grid.areas)-4*np.pi)/4*np.pi)
    return areas

####################################################################################
# Compute the metric tensor considering the unit sphere
####################################################################################
def metrictensor(grid):
    i0   = grid.i0
    iend = grid.iend
    j0   = grid.j0
    jend = grid.jend
    metrictensor = scalar_field(grid, 'metric_tensor', 'center')
    metrictensor.f = grid.metric_tensor_centers[i0:iend,j0:jend,:]
    return metrictensor

####################################################################################
# Compute the mean lenght for each cell in a cube sphere grid
####################################################################################
def mean_lengths(grid, length_x, length_y):
    # Interior cells index (we are ignoring ghost cells)
    i0   = grid.i0
    iend = grid.iend
    j0   = grid.j0
    jend = grid.jend
    length = scalar_field(grid, 'mean_length', 'center')
    # Mean edge lengths in x direction
    length_x = length_x[i0:iend,j0:jend,:]+length_x[i0:iend,j0+1:jend+1,:]
    length_x = 0.5*length_x

    # Mean edge lengths in y direction
    length_y = length_y[i0:iend,j0:jend,:]+length_y[i0+1:iend+1,j0:jend,:]
    length_y = 0.5*length_y

    # Mean length
    length.f = ((length_x + length_y)*0.5)*erad/10**3
    #length.f = erad*grid.length_x[:,0:grid.N,:]
    return length

####################################################################################
# Distortion of cell routine . Given a cell with edge lenghts l1, l2, l3, l4 we compute:
# lmean = sqrt{ (l1^2 + l2^2 + l3^2 + l4^2)/4 }
# The distortion is then given by:
# sqrt{ (l1-lmean)^2 +(l2-lmean)^2 + (l3-lmean)^2+(l4-lmean)^2)/4 }/lmean
#
# Based on Hirofumi Tomita, Motohiko Tsugawa, Masaki Satoh, Koji Goto, Shallow Water
# Model on a Modified Icosahedral Geodesic Grid by Using Spring Dynamics,
# Journal of Computational Physics, https://doi.org/10.1006/jcph.2001.6897.
####################################################################################
def distortions(grid, length_x, length_y):
    # Interior cells index (we are ignoring ghost cells)
    i0   = grid.i0
    iend = grid.iend
    j0   = grid.j0
    jend = grid.jend

    # Compute lmean = sqrt{ (l1^2 + l2^2 + l3^2 + l4^2)/4 }
    mean_length_l2 = length_y[i0:iend,j0:jend,:]**2
    mean_length_l2 = mean_length_l2 + length_y[i0+1:iend+1,j0:jend,:]**2
    mean_length_l2 = mean_length_l2 + length_x[i0:iend,j0:jend,:]**2
    mean_length_l2 = mean_length_l2 + length_x[i0:iend,j0+1:jend+1,:] **2
    mean_length_l2 = mean_length_l2/4.0
    mean_length_l2 = np.sqrt(mean_length_l2)

    # Compute the distortion
    distortion   =  scalar_field(grid, 'distortion', 'center')
    distortion.f = (length_y[i0:iend,j0:jend,:]-mean_length_l2)**2
    distortion.f = distortion.f + (length_y[i0+1:iend+1,j0:jend,:] - mean_length_l2)**2
    distortion.f = distortion.f + (length_x[i0:iend,j0:jend,:]   - mean_length_l2)**2
    distortion.f = distortion.f + (length_x[i0:iend,j0+1:jend+1,:] - mean_length_l2)**2
    distortion.f = distortion.f/4.0
    distortion.f = np.sqrt(distortion.f)/mean_length_l2
    return distortion

####################################################################################
# Compute the alignment index for each cell
# The alignement index has been introduced and define in the following paper:
# Pedro S. Peixoto, Saulo R.M. Barros, Analysis of grid imprinting on geodesic
# spherical icosahedral grids, Journal of Computational Physics, 2013,
# https://doi.org/10.1016/j.jcp.2012.11.041.
####################################################################################
def alignment_index(grid, length_x, length_y):
    # Interior cells index (we are ignoring ghost cells)
    i0   = grid.i0
    iend = grid.iend
    j0   = grid.j0
    jend = grid.jend
    # Compute the mean alignment
    mean_align = length_x[i0:iend,j0:jend,:] + length_x[i0:iend,j0+1:jend+1,:]
    mean_align = mean_align + length_y[i0:iend,j0:jend,:] + length_y[i0+1:iend+1,j0:jend,:]
    mean_align = mean_align*0.25

    # Each cell vertex is identified as below
    #    4---------3
    #    |         |
    #    |         |
    #    |         |
    #    |         |
    #    1---------2

    alignment   = scalar_field(grid, 'alignment', 'center')
    # |d_21 - d_43|
    alignment.f = alignment.f + abs(length_x[i0:iend,j0:jend,:]-length_x[i0:iend,j0+1:jend+1,:])
    # |d_32 - d_14|
    alignment.f = alignment.f + abs(length_y[i0:iend,j0:jend,:]-length_y[i0+1:iend+1,j0:jend,:])
    # |d_21 - d_43|
    alignment.f = alignment.f + abs(length_x[i0:iend,j0:jend,:]-length_x[i0:iend,j0+1:jend+1,:])
    # |d_32 - d_14|
    alignment.f = alignment.f + abs(length_y[i0:iend,j0:jend,:]-length_y[i0+1:iend+1,j0:jend,:])
    alignment.f = alignment.f*0.25
    alignment.f = alignment.f/mean_align
    return alignment
