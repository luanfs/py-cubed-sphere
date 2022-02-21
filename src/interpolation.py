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

####################################################################################
# This routine receives cubeb-sphere and lat-lon grids and convert each lat-lon 
# point to a cubed-sphere point seacrhing the nearest point in the cubed sphere.
#
# Outputs:
#
# - ix, jy: list of size 6 containing all the indexes (i,j) of latlon grid points that
#   are located in each panel 0,...,5]. For instance, ix[0], jy[0] contains all the 
#   values of such that (lon_ix[[0][:]],lat[jy[0][:]]) is located in panel 0.
#
# - mask: list of dimension [6, Nlon, Nlat], where mask[p,:,:] is a Boolean array such
#   that mask[p,ilon,jlat] = True if (lon(ilon),lat(jlat)) is located in panel p and
#   False otherwise.
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
      Δx = pio2/cs_grid.N
      Δy = pio2/cs_grid.N

   elif cs_grid.projection == 'gnomonic_equidistant':
      a  = 1.0/np.sqrt(3.0) # Half length of the cube
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

   # Empty lists
   ix = [None]*nbfaces
   jy = [None]*nbfaces
   mask = [None]*nbfaces

   # Start time counting
   print("--------------------------------------------------------")      
   print("Converting lat-lon grid points to cubed-sphere points (for plotting) ...")
   start_time = time.time()  

   # Find panel
   
   # Panel 0 - compute inverse transformation (sphere to cube)
   if cs_grid.projection == 'gnomonic_equiangular':
      X = yll/xll
      Y = zll/xll
      ξ, η = np.arctan(X), np.arctan(Y) 
   elif cs_grid.projection == 'gnomonic_equidistant':
      r = a/xll
      ξ = yll*r
      η = zll*r

   # Find the points in panel 0
   mask[0] = ((abs(ξ) <= a) & (abs(η) <= a) & (xll>=0))
   ix[0], jy[0] = find_closest_index(ξ[mask[0]],η[mask[0]])

   # Panel 2
   η = -η
   # Find the points in panel 2 
   mask[2] = ((abs(ξ) <= a) & (abs(η) <= a) & (xll<=0))  
   ix[2], jy[2] = find_closest_index(ξ[mask[2]],η[mask[2]])
   
   # Panel 1 - compute inverse transformation (sphere to cube)
   if cs_grid.projection == 'gnomonic_equiangular':
      X = -xll/yll
      Y =  zll/yll
      ξ, η = np.arctan(X), np.arctan(Y)
   elif cs_grid.projection == 'gnomonic_equidistant':
      r =  a/yll
      ξ = -xll*r
      η =  zll*r

   # Find the points in panel 1
   mask[1] = ((abs(ξ) <= a) & (abs(η) <= a) & (yll>=0)) 
   ix[1], jy[1] = find_closest_index(ξ[mask[1]],η[mask[1]])
         
   # Panel 3
   η = -η   
   # Find the points in panel 3   
   mask[3] = ((abs(ξ) <= a) & (abs(η) <= a) & (yll<=0))
   ix[3], jy[3] = find_closest_index(ξ[mask[3]],η[mask[3]])   
      
   # Panel 5 - compute inverse transformation (sphere to cube)
   if cs_grid.projection == 'gnomonic_equiangular':   
      X = -yll/zll
      Y = -xll/zll
      ξ, η = np.arctan(X), np.arctan(Y)
   elif cs_grid.projection == 'gnomonic_equidistant':
      r = -a/zll
      ξ =  yll*r
      η =  xll*r

   # Find the points in panel 5
   mask[5] = ((abs(ξ) <= a) & (abs(η) <= a) & (zll<=0))
   ix[5], jy[5] = find_closest_index(ξ[mask[5]],η[mask[5]])

   # Panel 4 
   # Find the points in panel 4
   ξ = -ξ
   mask[4] = ((abs(ξ) <= a) & (abs(η) <= a) & (zll>=0))   
   ix[4], jy[4] = find_closest_index(ξ[mask[4]],η[mask[4]])
   
   # Finish time counting
   elapsed_time = time.time() - start_time
   print("Done in ","{:.2e}".format(elapsed_time),"seconds.")
   print("--------------------------------------------------------\n")
   return ix, jy, mask
 
####################################################################################
# Interpolate values of ψ from latlon to cubed-sphere using nearest neighbour
####################################################################################
def nearest_neighbour(ψ, cs_grid, latlon_grid):
   ψ_ll = np.zeros((latlon_grid.Nlon, latlon_grid.Nlat))
   for p in range(0, nbfaces):
      ψ_ll[latlon_grid.mask[p]] = ψ.f[latlon_grid.ix[p], latlon_grid.jy[p],p]
   return ψ_ll
