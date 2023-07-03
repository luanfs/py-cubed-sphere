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
from advection_error    import error_analysis_adv
from operator_accuracy  import error_analysis_div
from interpolation_test import interpolation_test 

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
        if test_case == 2 or test_case == 5:
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
            # Call interpolation test case
            interpolation_test(map_projection, transformation, showonscreen, True)

        elif test_case == 4:
            # Call divergence test case
            print("Test case 4: Divergence test case.\n")
            dt, Tf, tc, ic, vf, recon, dp, opsplit, et, mt = get_advection_parameters()
            plot = True
            error_analysis_div(vf, map_projection, plot, transformation, showonscreen, gridload)

        elif test_case == 5:
            # Call advection test case
            print("Test case 5: Advection test case.\n")
            dt, Tf, tc, ic, vf, recon, dp, opsplit, et, mt = get_advection_parameters()
            simulation = adv_simulation_par(cs_grid, dt, Tf, ic, vf, tc, recon, dp, opsplit, et, mt)

            if simulation.tc == 1: # Advection on the sphere simulation
                plot = True
                adv_sphere(cs_grid, ll_grid, simulation, map_projection, plot, False)

            elif simulation.tc == 2: # Convergence analysis
                plot = False
                error_analysis_adv(simulation, map_projection, plot, transformation, \
                            showonscreen, gridload)

            else:
                print('Invalid advection testcase.\n')
                exit()
        else:
            print("ERROR: invalid testcase.")
            exit()
main()
