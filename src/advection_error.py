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
# Routine to compute the advection error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_adv(simulation, map_projection, plot, transformation, showonscreen, gridload):
    # Initial condition
    vf = simulation.vf

    # Flux method
    recon = simulation.recon

    # Test case
    tc = simulation.tc

    # Number of tests
    Ntest = 4

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
    recons = (3,)
    deps = (1,2)
    split = (1,3)
    ets = (1,2,5)

    #recons = (simulation.recon,)
    #deps = (simulation.dp,)
    #split = (simulation.opsplit,)

    recon_names = ['PPM-0', 'PPM-CW84','PPM-PL07','PPM-L04']
    dp_names = ['RK1', 'RK2']
    sp_names = ['SP-AVLT', 'SP-L04', 'SP-PL07']
    et_names = ['ET-S72', 'ET-PL07', 'ET-R96', 'ET-R96-AF','ET-Z21', 'ET-Z21-AF']
    error_linf = np.zeros((Ntest, len(recons), len(split), len(ets), len(deps)))
    error_l1   = np.zeros((Ntest, len(recons), len(split), len(ets), len(deps)))
    error_l2   = np.zeros((Ntest, len(recons), len(split), len(ets), len(deps)))

    # Let us test and compute the error!
    dt, Tf, tc, ic, vf, recon, dp, opsplit, et = get_advection_parameters()

    # Period for all tests
    Tf = 5

    # Let us test and compute the error
    d = 0
    for dp in deps:
        et = 0
        for ET in ets:
            sp = 0
            for opsplit in split:
                rec = 0
                for recon in recons:
                    for i in range(0, Ntest):
                        dt = dts[i]
                        simulation = adv_simulation_par(dt, Tf, ic, vf, tc, recon, dp, opsplit, ET)
                        N = int(Nc[i])

                        # Create CS mesh
                        cs_grid = cubed_sphere(N, transformation, False, gridload)

                        # Create the latlon mesh (for plotting)
                        ll_grid = latlon_grid(Nlat, Nlon)
                        ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

                        # Get advection error
                        error_linf[i,rec,sp,et,d], error_l1[i,rec,sp,et,d], error_l2[i,rec,sp,et,d] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, False, False)

                        # Print errors
                        print_errors_simul(error_linf[:,rec,sp,et,d], error_l1[:,rec,sp,et,d], error_l2[:,rec,sp,et,d], i)
                    rec = rec+1
                sp = sp+1
            et = et+1
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
                    for et in range(0, len(ets)):
                        errors.append(error[:,r,sp,et,d])
                        dep_name.append(sp_names[split[sp]-1]+'/'+recon_names[recons[r]-1]+'/'+et_names[ets[et]-1])

            title = simulation.title + '- ic=' + str(simulation.ic)+', vf='+ str(simulation.vf)+\
            ', dp='+dp_names[deps[d]-1]+', norm='+norm_title[e]
            filename = graphdir+'cs_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_dp'+dp_names[deps[d]-1]\
            +'_norm'+norm_list[e]+'_parabola_errors.pdf'

            plot_errors_loglog(Nc, errors, dep_name, filename, title, emin, emax)

            # Plot the convergence rate
            title = 'Convergence rate - ic=' + str(simulation.ic) +', vf=' + str(simulation.vf)+\
            ', dp='+dp_names[deps[d]-1]+', norm='+norm_title[e]
            filename = graphdir+'cs_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_dp'+dp_names[deps[d]-1]\
            +'_norm'+norm_list[e]+'_convergence_rate.pdf'
            plot_convergence_rate(Nc, errors, dep_name, filename, title, CRmin, CRmax)
            e = e+1
