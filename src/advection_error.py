###################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the advection equation
# Luan da Fonseca Santos - September 2022
#
####################################################################################
import numpy as np
from constants import*
from advection_sphere import adv_sphere
from advection_ic     import adv_simulation_par
from errors           import compute_errors, print_errors_simul, plot_convergence_rate, plot_errors_loglog
from configuration    import get_advection_parameters
from cs_datastruct    import cubed_sphere, latlon_grid
from interpolation    import ll2cs

###################################################################################
# Module to compute the advection error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_adv(simulation, map_projection, plot, transformation, showonscreen, gridload):
    # Initial condition
    ic = simulation.ic

    # Monotonization method
    mono = simulation.mono

    # Test case
    tc = simulation.tc

    # Number of tests
    Ntest = 4

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 20

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Let us test and compute the error!
    dt, Tf, tc, ic, mono = get_advection_parameters()

    if ic >= 1 and ic <= 4:
        Tf = 12.0*day2sec   # Period
        u0 = 2.0*erad*pi/(12.0*day2sec) # maximum velocity

    # Time step
    dts = np.zeros(Ntest)

    # CFL number
    CFL = 0.5

    for i in range(0, Ntest):
        simulation = adv_simulation_par(dt, Tf, ic, tc, mono)
        N = int(Nc[i])

        # Create CS mesh
        cs_grid = cubed_sphere(N, transformation, False, gridload)
        minlen = np.amin(cs_grid.length_x)

        dts[i] = CFL*minlen/u0
        simulation.dt = dts[i]

        # Create the latlon mesh (for plotting)
        ll_grid = latlon_grid(Nlat, Nlon)
        ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

        # Get advection error
        error_linf[i], error_l1[i], error_l2[i] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, False)

        # Print errors
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Outputs
    # Convergence rate
    title = "Convergence rate - advection equation"
    filename = graphdir+"adv_tc"+str(tc)+"_ic"+str(ic)+"_cr_rate_"+transformation+"_mono_"+simulation.monot+"_"+simulation.fvmethod
    plot_convergence_rate(Nc, error_linf, error_l1, error_l2, filename, title)

    # Error convergence
    title = "Convergence of error  - advection equation"
    filename = graphdir+"adv_tc"+str(tc)+"_ic"+str(ic)+"_error_convergence_"+transformation+"_mono_"+simulation.monot+"_"+simulation.fvmethod
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)
