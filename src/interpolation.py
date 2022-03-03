####################################################################################
# 
# Module for cubed-sphere mesh interpolation routines
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from constants import*
from sphgeo import*
import time
from cs_transform import inverse_equiangular_gnomonic_map, inverse_equidistant_gnomonic_map, linear_search, inverse_conformal_map

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
   xll = latlon_grid.x
   yll = latlon_grid.y
   zll = latlon_grid.z

   # Define a grid for each transformation
   if cs_grid.projection == 'gnomonic_equiangular':
      a  = pio4      
   elif cs_grid.projection == 'gnomonic_equidistant':
      a  = 1.0/np.sqrt(3.0) # Half length of the cube
   elif cs_grid.projection == 'conformal':
      a  = 1.0
   
   # Grid spacing
   Δx = 2*a/cs_grid.N
   Δy = 2*a/cs_grid.N

   # Create the grid
   [xmin, xmax, ymin, ymax] = [-a, a, -a, a]      
   x = np.linspace(xmin+Δx/2.0, xmax-Δx/2.0, cs_grid.N)
   y = np.linspace(ymin+Δy/2.0, ymax-Δy/2.0, cs_grid.N)

   # This routine receives an array xx (yy) and for each 
   # point in xx (yy), returns in i (j) the index of the 
   # closest point in the array xx (yy).
   def find_closest_index(xx,yy):
      i = (np.floor((xx-xmin)/Δx))
      j = (np.floor((yy-ymin)/Δy))
      i = np.array(i, dtype=np.uint32)
      j = np.array(j, dtype=np.uint32)  
      return i, j

   # Start time counting
   print("--------------------------------------------------------")      
   print("Converting lat-lon grid points to cubed-sphere points (for plotting) ...")
   start_time = time.time()  

   # Find panel - Following Lauritzen et al 2015.
   P = np.zeros( (np.shape(xll)[0], np.shape(xll)[1], 3) )
   P[:,:,0] = abs(xll)
   P[:,:,1] = abs(yll)
   P[:,:,2] = abs(zll)
   PM = P.max(axis=2)

   # Indexes lists
   i = np.zeros(np.shape(xll), dtype=np.uint32)
   j = np.zeros(np.shape(xll), dtype=np.uint32)   
   panel_list = np.zeros(np.shape(xll), dtype=np.uint32)

   # Panel 0 
   mask = np.logical_and(PM == abs(xll), xll>0)
   panel_list[mask] = 0

   # Panel 1
   mask = np.logical_and(PM == abs(yll), yll>0)
   panel_list[mask] = 1
   
   # Panel 2
   mask = np.logical_and(PM == abs(xll), xll<0)
   panel_list[mask] = 2

   # Panel 3
   mask = np.logical_and(PM == abs(yll), yll<0)
   panel_list[mask] = 3
   
   # Panel 4
   mask = np.logical_and(PM == abs(zll), zll>0)
   panel_list[mask] = 4
   
   # Panel 5
   mask = np.logical_and(PM == abs(zll), zll<=0)
   panel_list[mask] = 5

   # Compute inverse transformation (sphere to cube) for each panel p
   for p in range(0, nbfaces):
      mask = (panel_list == p)
      if cs_grid.projection == 'gnomonic_equiangular':
         ξ, η = inverse_equiangular_gnomonic_map(xll[mask], yll[mask], zll[mask], p)
         i[mask], j[mask] = find_closest_index(ξ, η)
      elif cs_grid.projection == 'gnomonic_equidistant':
         ξ, η = inverse_equidistant_gnomonic_map(xll[mask], yll[mask], zll[mask], p)
         i[mask], j[mask] = find_closest_index(ξ, η)
      elif cs_grid.projection == 'conformal':
         print(p)
         i[mask], j[mask] = linear_search(xll[mask], yll[mask], zll[mask], cs_grid, p)
         #i[mask], j[mask] = inverse_conformal_map(xll[mask], yll[mask], zll[mask], cs_grid, p)
         #exit()
   #exit()
   # Finish time counting
   elapsed_time = time.time() - start_time

   print("Done in ","{:.2e}".format(elapsed_time),"seconds.")
   print("--------------------------------------------------------\n")
   return i, j, panel_list
   
####################################################################################
# Interpolate values of ψ from latlon to cubed-sphere using nearest neighbour
####################################################################################
def nearest_neighbour(ψ, cs_grid, latlon_grid):
   ψ_ll = np.zeros((latlon_grid.Nlon, latlon_grid.Nlat))
   for p in range(0, nbfaces):
      mask = (latlon_grid.mask==p)
      ψ_ll[mask] = ψ.f[latlon_grid.ix[mask], latlon_grid.jy[mask], p]
   return ψ_ll
