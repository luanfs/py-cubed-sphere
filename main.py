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
sys.path.append(srcdir)

# Imports
from miscellaneous import createDirs
from configuration import get_parameters
from cs_datastruct import cubed_sphere, latlon_grid
from interpolation import ll2cs
from gridquality   import grid_quality
from plot          import plot_grid
from constants     import Nlat, Nlon

def main():
   # Create the directories
   createDirs()

   # Get the parameters
   N, transformation, showonscreen, test_case, map_projection = get_parameters()

   # Select test case
   if test_case == 1:    
      print("Test case 1: cubed-sphere generation and plotting.\n")
      # Create CS mesh
      cs_grid = cubed_sphere(N, transformation, showonscreen)

      # Mesh plot      
      plot_grid(cs_grid, map_projection)

   else:
      # Create the CS mesh
      cs_grid = cubed_sphere(N, transformation, showonscreen)

      # Create the latlon mesh (for plotting)  
      ll_grid = latlon_grid(Nlat, Nlon)
      ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)      

      if test_case == 2:
         print("Test case 2: grid quality test.\n")
         # Call grid quality test
         grid_quality(cs_grid, ll_grid, map_projection)
      else:
         print("ERROR: invalid testcase.")
         exit()
main()
