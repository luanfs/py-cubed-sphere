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

    # Velocity field
    vf = simulation.vf

    # Flux method
    flux_method = simulation.flux_method

    # Test case
    tc = simulation.tc

    # Number of tests
    Ntest = 5

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
    dt, Tf, tc, ic, vf, flux_method, degree = get_advection_parameters()

    Tf = 5.0   # Period
    if vf <= 2:
        u0 = 2.0*pi/5.0 # maximum velocity
    else:
        u0 = 4.0
    # Time step
    dts = np.zeros(Ntest)

    # CFL number
    CFL = 0.5

    for i in range(0, Ntest):
        simulation = adv_simulation_par(dt, Tf, ic, vf, tc, flux_method, degree)
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
        error_linf[i], error_l1[i], error_l2[i] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, False)

        # Print errors
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Outputs
    # Convergence rate
    title = "Convergence rate - advection equation, "+ simulation.flux_method_name
    filename = graphdir+"adv_tc"+str(tc)+"_ic"+str(ic)+"_vf"+str(vf)+"_cr_rate_"+transformation+"_"+simulation.flux_method_name
    plot_convergence_rate(Nc, error_linf, error_l1, error_l2, filename, title)

    # Error convergence
    title = "Convergence of error  - advection equation, "+ simulation.flux_method_name
    filename = graphdir+"adv_tc"+str(tc)+"_ic"+str(ic)+"_vf"+str(vf)+"_error_convergence_"+transformation+"_"+simulation.flux_method_name
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)
