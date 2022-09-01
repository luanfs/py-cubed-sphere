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
from constants     import pio2, rad2deg, datadir

####################################################################################
# This routine compute grid quality measures and save it as a netcdf file
####################################################################################
def grid_quality(cs_grid, ll_grid, map_projection):
    print("--------------------------------------------------------")
    print("Grid quality testing\n")

    # Compute the quality measures
    angle         = angles_deviation(cs_grid)
    area          = areas(cs_grid)   
    mean_length   = mean_lengths(cs_grid)
    distortion    = distortions (cs_grid)
    alignment     = alignment_index(cs_grid)
    metric_tensor = metrictensor(cs_grid)
 
    # Interpolate to the latlon grid
    angle_ll         = nearest_neighbour(angle, cs_grid, ll_grid)
    area_ll          = nearest_neighbour(area, cs_grid, ll_grid)
    mean_length_ll   = nearest_neighbour(mean_length, cs_grid, ll_grid)
    distortion_ll    = nearest_neighbour(distortion, cs_grid, ll_grid)
    alignment_ll     = nearest_neighbour(alignment, cs_grid, ll_grid)
    metric_tensor_ll = nearest_neighbour(metric_tensor, cs_grid, ll_grid)

    # Create list of fields
    fields_ll = [angle_ll , area_ll, mean_length_ll, distortion_ll, alignment_ll, metric_tensor_ll]
    fields_cs = [angle    , area   , mean_length   , distortion   , alignment   , metric_tensor   ]
  
    # Netcdf file
    netcdf_name = cs_grid.name+'_grid_quality'
    netcdf_data = open_netcdf4(fields_cs, [0], netcdf_name)
 
    # Plot the fields
    for l in range(0,len(fields_ll)):
        plot_scalar_field(fields_ll[l], fields_cs[l].name, cs_grid, ll_grid, map_projection)
        netcdf_data[fields_cs[l].name][:,:,0] = fields_ll[l][:,:]

    #Close netcdf file
    netcdf_data.close()
    print("A netcdf file has been created in ", datadir+netcdf_name+'.nc')
    print("--------------------------------------------------------\n")

####################################################################################
# Compute the cell areas considering the earth radius
####################################################################################
def areas(grid):
    areas = scalar_field(grid, 'areas', 'center')
    areas.f = grid.areas
    #areas.f = np.sqrt(areas.f)
    #print((np.sum(grid.areas)-4*np.pi)/4*np.pi)
    return areas

####################################################################################
# Compute the metric tensor considering the unit sphere
####################################################################################
def metrictensor(grid):
    metrictensor = scalar_field(grid, 'metric_tensor', 'center')
    metrictensor.f = grid.metric_tensor_centers
    return metrictensor

####################################################################################
# Compute the angle deviation from orthogonality for each cell in a cube sphere grid
####################################################################################
def angles_deviation(grid):
    angles = scalar_field(grid, 'angle', 'center')
    for k in range(0,4):
        angles.f = angles.f + (grid.angles[:,:,:,k]*rad2deg-90.0)**2
    angles.f = np.sqrt(angles.f)/2.0
    return angles

####################################################################################
# Compute the mean lenght for each cell in a cube sphere grid
####################################################################################
def mean_lengths(grid):
    length = scalar_field(grid, 'mean_length', 'center')   
    # Mean edge lengths in x direction   
    length_x = grid.length_x[:,0:grid.N,:]+grid.length_x[:,1:grid.N+1,:]   
    length_x = 0.5*length_x

    # Mean edge lengths in y direction
    length_y = grid.length_y[0:grid.N,:,:]+grid.length_y[1:grid.N+1,:,:]
    length_y = 0.5*length_y
   
    # Mean length
    length.f = (length_x + length_y)*0.5
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
def distortions(grid):
    # Compute lmean = sqrt{ (l1^2 + l2^2 + l3^2 + l4^2)/4 }
    mean_length_l2 = grid.length_y[0:grid.N,:,:]**2
    mean_length_l2 = mean_length_l2 + grid.length_y[1:grid.N+1,:,:]**2
    mean_length_l2 = mean_length_l2 + grid.length_x[:,0:grid.N,:]**2
    mean_length_l2 = mean_length_l2 + grid.length_x[:,1:grid.N+1,:] **2      
    mean_length_l2 = mean_length_l2/4.0
    mean_length_l2 = np.sqrt(mean_length_l2)
   
    # Compute the distortion
    distortion   =  scalar_field(grid, 'distortion', 'center')
    distortion.f = (grid.length_y[0:grid.N,:,:]-mean_length_l2)**2
    distortion.f = distortion.f + (grid.length_y[1:grid.N+1,:,:] - mean_length_l2)**2
    distortion.f = distortion.f + (grid.length_x[:,0:grid.N,:]   - mean_length_l2)**2
    distortion.f = distortion.f + (grid.length_x[:,1:grid.N+1,:] - mean_length_l2)**2
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
def alignment_index(grid):
    # Compute the mean alignment
    mean_align = grid.length_x[:,0:grid.N,:] + grid.length_x[:,1:grid.N+1,:]
    mean_align = mean_align + grid.length_y[0:grid.N,:,:] + grid.length_y[1:grid.N+1,:,:] 
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
    alignment.f = alignment.f + abs(grid.length_x[:,0:grid.N,:] -grid.length_x[:,1:grid.N+1,:])
    # |d_32 - d_14|
    alignment.f = alignment.f + abs(grid.length_y[0:grid.N,:,:] -grid.length_y[1:grid.N+1,:,:])
    # |d_21 - d_43|
    alignment.f = alignment.f + abs(grid.length_x[:,0:grid.N,:] -grid.length_x[:,1:grid.N+1,:])
    # |d_32 - d_14|   
    alignment.f = alignment.f + abs(grid.length_y[0:grid.N,:,:] -grid.length_y[1:grid.N+1,:,:])
    alignment.f = alignment.f*0.25
    alignment.f = alignment.f/mean_align
    return alignment
