#########################################################
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
from miscellaneous      import createDirs
from configuration      import get_parameters, get_advection_parameters
from cs_datastruct      import cubed_sphere, latlon_grid
from interpolation      import ll2cs
from gridquality        import grid_quality
from plot               import plot_grid, save_grid_netcdf4
from constants          import Nlat, Nlon
from advection_ic       import adv_simulation_par
from advection_sphere   import adv_sphere

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

        elif test_case == 3:
            # Call advection test case
            print("Test case 3: Advection test case.\n")
            dt, Tf, tc, ic, mono = get_advection_parameters()
            simulation = adv_simulation_par(dt, Tf, ic, tc, mono)

            if simulation.tc == 1: # Advection on the sphere simalution
                plot = True
                adv_sphere(cs_grid, ll_grid, simulation, map_projection, plot)

            elif simulation.tc == 2: # Convergence analysis
                print('Not implemented yet =(')

            else:
                print('Invalid advection testcase.\n')
                exit()
        else:
            print("ERROR: invalid testcase.")
            exit()
main()
