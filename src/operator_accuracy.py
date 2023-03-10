####################################################################################
#
# Module for FV operators accuracy testing
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
#
#
####################################################################################

import numpy as np
from constants import*
from cs_datastruct          import cubed_sphere, latlon_grid, ppm_parabola, scalar_field
from errors                 import print_errors_simul, plot_errors_loglog, plot_convergence_rate
from configuration          import get_advection_parameters
from advection_ic           import adv_simulation_par
from advection_timestep     import adv_time_step
from advection_sphere       import adv_sphere
from sphgeo                 import sph2cart, cart2sph
from scipy.special          import sph_harm
from interpolation          import ll2cs, nearest_neighbour, ghost_cells_lagrange_interpolation
from lagrange               import lagrange_poly_ghostcells
from plot                   import plot_scalar_field
###################################################################################
# Routine to compute the divergence error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_div(simulation, map_projection, plot, transformation, showonscreen, gridload):
    # Initial condition
    vf = simulation.vf

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
    recons = (3,4)
    #deps = (1,)
    split = (1,3)

    #recons = (simulation.recon,)
    deps = (simulation.dp,)
    #split = (simulation.opsplit,)

    recon_names = ['PPM', 'PPM-CW84','PPM-PL07','PPM-L04']
    dp_names = ['RK1', 'RK3']
    sp_names = ['SP-AVLT', 'SP-L04', 'SP-PL07']
    error_linf = np.zeros((Ntest, len(recons), len(split), len(deps)))
    error_l1   = np.zeros((Ntest, len(recons), len(split), len(deps)))
    error_l2   = np.zeros((Ntest, len(recons), len(split), len(deps)))

    # Let us test and compute the error!
    dt, Tf, tc, ic, vf, recon, dp, opsplit = get_advection_parameters()

    # For divergence testing, we consider a constant field
    ic = 1

    # Let us test and compute the error
    d = 0
    for dp in deps:
        sp = 0
        for opsplit in split:
            rec = 0
            for recon in recons:
                for i in range(0, Ntest):
                    dt = dts[i]
                    simulation = adv_simulation_par(dt, Tf, ic, vf, tc, recon, dp, opsplit)
                    N = int(Nc[i])

                    # Create CS mesh
                    cs_grid = cubed_sphere(N, transformation, False, gridload)

                    # Create the latlon mesh (for plotting)
                    ll_grid = latlon_grid(Nlat, Nlon)
                    ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

                    # Get divergence error
                    error_linf[i,rec,sp,d], error_l1[i,rec,sp,d], error_l2[i,rec,sp,d] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, False, True)
                    print('\nParameters: N = '+str(int(Nc[i]))+', dt = '+str(dts[i]),', recon = ', recon,', split = ',opsplit, ', dp = ', dp)

                    # Print errors
                    print_errors_simul(error_linf[:,rec,sp,d], error_l1[:,rec,sp,d], error_l2[:,rec,sp,d], i)
                rec = rec+1
            sp = sp+1
        d = d+1

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']

    for d in range(0, len(deps)):
        e = 0
        for error in error_list:
            emin, emax = np.amin(error[:]), np.amax(error[:])

            # convergence rate min/max
            n = len(error)
            CR = np.abs(np.log(error[1:n])-np.log(error[0:n-1]))/np.log(2.0)
            CRmin, CRmax = np.amin(CR), np.amax(CR)
            errors = []
            dep_name = []
            for sp in range(0, len(split)):
                for r in range(0, len(recons)):
                    errors.append(error[:,r,sp,d])
                    dep_name.append(sp_names[split[sp]-1]+'/'+recon_names[recons[r]-1])

            title = 'Divergence error, vf='+ str(simulation.vf)+\
            ', dp='+dp_names[deps[d]-1]+', norm='+norm_title[e]
            filename = graphdir+'cs_div_vf'+str(vf)+'_dp'+dp_names[deps[d]-1]\
            +'_norm'+norm_list[e]+'_parabola_errors.pdf'
            plot_errors_loglog(Nc, errors, dep_name, filename, title, emin, emax)

            # Plot the convergence rate
            title = 'Divergence convergence rate, vf=' + str(simulation.vf)+\
            ', dp='+dp_names[deps[d]-1]+', norm='+norm_title[e]
            filename = graphdir+'cs_div_vf'+str(vf)+'_dp'+dp_names[deps[d]-1]\
            +'_norm'+norm_list[e]+'_convergence_rate.pdf'
            plot_convergence_rate(Nc, errors, dep_name, filename, title, CRmin, CRmax)
            e = e+1
