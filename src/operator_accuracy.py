####################################################################################
#
# Module for FV operators accuaracy testing
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
#
#
####################################################################################

import numpy as np
from constants import*
from cs_datastruct          import cubed_sphere, latlon_grid
from interpolation          import ll2cs
from errors                 import print_errors_simul, plot_errors_loglog, plot_convergence_rate
from configuration          import get_advection_parameters
from advection_ic           import adv_simulation_par
from advection_timestep     import adv_time_step
from advection_sphere       import adv_sphere

###################################################################################
# Routine to compute the divergence error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_div(simulation, map_projection, plot, transformation, showonscreen, gridload):
    # Initial condition
    vf = simulation.vf

    # Flux method
    recon = simulation.recon

    # Test case
    tc = simulation.tc

    # Number of tests
    Ntest = 6

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    dts = np.zeros(Ntest)
    Nc[0] = 16

    if simulation.vf==1 or simulation.vf==2:
        dts[0] = 0.05
    elif simulation.vf==3:
        dts[0] = 0.025
    elif simulation.vf==4:
        dts[0] = 0.0125
    elif simulation.vf==5:
        dts[0] = 0.0125
    elif simulation.vf==6:
        dts[0] = 0.0125
    else:
        print('ERROR: invalid vector field, ',simulation.vf)
        exit()

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i]  = Nc[i-1]*2
        dts[i] = dts[i-1]*0.5

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Let us test and compute the error!
    dt, Tf, tc, ic, vf, recon, opsplit, degree = get_advection_parameters()
    ic = 1
    for i in range(0, Ntest):
        dt = dts[i]
        simulation = adv_simulation_par(dt, Tf, ic, vf, tc, recon, opsplit, degree)
        N = int(Nc[i])

        # Create CS mesh
        cs_grid = cubed_sphere(N, transformation, False, gridload)

        # Create the latlon mesh (for plotting)
        ll_grid = latlon_grid(Nlat, Nlon)
        ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

        # Get divergence error
        error_linf[i], error_l1[i], error_l2[i] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, False, True)

        # Print errors
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Outputs
    # Convergence rate
    title = "Convergence rate - divergence operator, "+simulation.opsplit_name+', '+simulation.recon_name
    filename = graphdir+"div_tc"+str(tc)+"_vf"+str(vf)+"_cr_rate_"+transformation+'_'+simulation.opsplit_name+'_'+simulation.recon_name
    plot_convergence_rate(Nc, error_linf, error_l1, error_l2, filename, title)

    # Error convergence
    title = "Convergence of error  - divergence operator, "+simulation.opsplit_name+', '+simulation.recon_name
    filename = graphdir+"div_tc"+str(tc)+"_vf"+str(vf)+"_error_convergence_"+transformation+'_'+simulation.opsplit_name+'_'+simulation.recon_name
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)
