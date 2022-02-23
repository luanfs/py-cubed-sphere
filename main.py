##########################################################
#
# Cubed-sphere generation code
#
# Luan da Fonseca Santos (luan.santos@usp.br)
#
##########################################################

# Source code directory
srcdir = "src/"

import sys
import os.path
sys.path.append(srcdir)

# Imports
from miscellaneous import createDirs
from configuration import get_parameters
from cs_datastruct import cubed_sphere, latlon_grid
from interpolation import ll2cs
from gridquality   import grid_quality
from plot          import plot_grid, save_grid_netcdf4
from constants     import Nlat, Nlon
#from divergence    import divergence_test

def main():
   # Create the directories
   createDirs()

   # Get the parameters
   N, transformation, showonscreen, gridload, test_case, map_projection = get_parameters()

   # Select test case
   if test_case == 1:    
      print("Test case 1: cubed-sphere generation and plotting.\n")
      # Create CS mesh
      cs_grid = cubed_sphere(N, transformation, showonscreen, gridload)

      # Mesh plot      
      plot_grid(cs_grid, map_projection)
      
      # Save grid in netcdf format
      if not(os.path.isfile(cs_grid.netcdfdata_filename)) or (os.path.isfile(cs_grid.netcdfdata_filename) and gridload==False):
         save_grid_netcdf4(cs_grid)

   else:
      # Create the CS mesh
      cs_grid = cubed_sphere(N, transformation, showonscreen, gridload)

      # Save grid in netcdf format
      if not(os.path.isfile(cs_grid.netcdfdata_filename)) or (os.path.isfile(cs_grid.netcdfdata_filename) and gridload==False):
         save_grid_netcdf4(cs_grid)

      # Create the latlon mesh (for plotting)  
      ll_grid = latlon_grid(Nlat, Nlon)
      ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)      

      if test_case == 2:
         print("Test case 2: grid quality test.\n")
         # Call grid quality test
         grid_quality(cs_grid, ll_grid, map_projection)

      #elif test_case == 3:
      #   # Call divergence test
      #   print("Test case 3: Divergence test.\n")
      #   divergence_test(cs_grid, ll_grid, map_projection)

      else:
         print("ERROR: invalid testcase.")
         exit()
main()
