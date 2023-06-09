###################################################################################
#
# Module for FV operators accuracy testing
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
#
#
####################################################################################

import numpy as np
import os.path
from constants import*
from cs_datastruct          import cubed_sphere, latlon_grid, ppm_parabola, scalar_field
from errors                 import print_errors_simul, plot_errors_loglog, plot_convergence_rate
from configuration          import get_advection_parameters
from advection_ic           import adv_simulation_par
from advection_timestep     import adv_time_step
from advection_sphere       import adv_sphere
from sphgeo                 import sph2cart, cart2sph
from scipy.special          import sph_harm
from interpolation          import ll2cs, nearest_neighbour 
from lagrange               import lagrange_poly_ghostcell_pc
from plot                   import plot_scalar_field, save_grid_netcdf4


###################################################################################
# Routine to compute the divergence error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_div(vf, map_projection, plot, transformation, showonscreen,\
                       gridload):
    # Number of tests
    Ntest = 7

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    dts = np.zeros(Ntest)
    Nc[0] = 16

    if vf==1 or vf==2:
        dts[0] = 0.025
    elif vf==3:
        dts[0] = 0.025
    elif vf==4:
        dts[0] = 0.0125
    elif vf==5:
        dts[0] = 0.0125
    elif vf==6:
        dts[0] = 0.0125
    else:
        print('ERROR: invalid vector field, ',simulation.vf)
        exit()

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i]  = Nc[i-1]*2
        dts[i] = dts[i-1]*0.5

    # Errors array
    recons = (3,)
    split = (1,1,1)
    ets   = (3,4,5)
    deps  = (1,1,1)

    recon_names = ['PPM-0', 'PPM-CW84','PPM-PL07','PPM-L04']
    dp_names = ['RK1', 'RK2']
    sp_names = ['SP-AVLT', 'SP-L04', 'SP-PL07']
    et_names = ['ET-S72', 'ET-PL07', 'ET-Z21', 'ET-Z21-AF', 'ET-Z21-PR']
    error_linf = np.zeros((Ntest, len(recons), len(split)))
    error_l1   = np.zeros((Ntest, len(recons), len(split)))
    error_l2   = np.zeros((Ntest, len(recons), len(split)))

    # Let us test and compute the error!
    dt, Tf, tc, ic, vf, recon, dp, opsplit, et = get_advection_parameters()

    # For divergence testing, we consider a constant field
    ic = 1

    # Let us test and compute the error
    for d in range(0,len(deps)):
        dp = deps[d]
        opsplit = split[d]
        ET = ets[d]
        rec = 0
        for recon in recons:
            for i in range(0, Ntest):
                dt = dts[i]
                N = int(Nc[i])

                # Create CS mesh
                cs_grid = cubed_sphere(N, transformation, False, gridload)

                # simulation class 
                simulation = adv_simulation_par(cs_grid, dt, Tf, ic, vf, tc, recon, dp, opsplit, ET)

                # Save the grid
                if not(os.path.isfile(cs_grid.netcdfdata_filename)):
                    save_grid_netcdf4(cs_grid)

                # Create the latlon mesh (for plotting)
                ll_grid = latlon_grid(Nlat, Nlon)
                ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

                # Print parameters
                print('\nParameters: N='+str(int(Nc[i]))+', dt='+str(dts[i]),', recon=', simulation.recon_name,', split=', simulation.opsplit_name, ', dp=', simulation.dp_name, ', et=', simulation.et_name)

                # Get divergence error
                error_linf[i,rec,d], error_l1[i,rec,d], error_l2[i,rec,d] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, False, True)

                # Print errors
                print_errors_simul(error_linf[:,rec,d], error_l1[:,rec,d], error_l2[:,rec,d], i)
            rec= rec+1

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']

    e = 0
    for error in error_list:
        emin, emax = np.amin(error[:]), np.amax(error[:])

        # convergence rate min/max
        n = len(error)
        CR = np.abs(np.log(error[1:n])-np.log(error[0:n-1]))/np.log(2.0)
        CRmin, CRmax = np.amin(CR), np.amax(CR)
        errors = []
        dep_name = []
        for r in range(0, len(recons)):
            for d in range(0, len(deps)):
                errors.append(error[:,r,d])
                dep_name.append(sp_names[split[d]-1]+'/'+recon_names[recons[r]-1]+'/'+et_names[ets[d]-1]+'/'+str(dp_names[deps[d]-1]))

        title = 'Divergence error, vf='+ str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_div_vf'+str(vf)+'_norm'+norm_list[e]+'_parabola_errors.pdf'
        plot_errors_loglog(Nc, errors, dep_name, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Divergence convergence rate, vf=' + str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_div_vf'+str(vf)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, dep_name, filename, title, CRmin, CRmax)
        e = e+1
