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
from plot             import plot_scalar_field

###################################################################################
# Routine to compute the advection error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_adv(simulation, map_projection, plot, transformation, \
                      showonscreen, gridload):
    # Initial condition
    vf = simulation.vf

    # Flux method
    recon = simulation.recon

    # Test case
    tc = simulation.tc

    # Number of tests
    Ntest  = 4

    # run simulation? if false, load the error file from data (must already exist)
    run = True

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    dts = np.zeros(Ntest)
    Nc[0] = 16

    if simulation.vf==1:
        dts[0] = 0.025
    elif simulation.vf==2:
        dts[0] = 0.0125
    elif simulation.vf==3:
        dts[0] = 0.00625
    elif simulation.vf==4:
        dts[0] = 0.0125
    else:
        print('ERROR: invalid vector field, ',simulation.vf)
        exit()

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i]  = Nc[i-1]*2
        dts[i] = dts[i-1]*0.5

    # Errors array
    recons = (3,3,3,3)
    split  = (3,3,1,1)
    ets    = (2,3,3,3)
    mts    = (2,2,1,1)
    deps   = (1,1,2,2)
    mfs    = (1,3,2,3)

    recon_names = ['PPM-0', 'PPM-CW84','PPM-PL07','PPM-L04']
    dp_names = ['RK1', 'RK2']
    sp_names = ['SP-AVLT', 'SP-L04', 'SP-PL07']
    et_names = ['ET-S72', 'ET-PL07', 'ET-DG']
    mt_names = ['MT-0', 'MT-PL07']
    mf_names = ['MF-0', 'MF-AF', 'MF-PR']

    error_linf = np.zeros((Ntest, len(recons)))
    error_l1   = np.zeros((Ntest, len(recons)))
    error_l2   = np.zeros((Ntest, len(recons)))

    # Let us test and compute the error!
    dt, Tf, tc, ic, vf, recon, dp, opsplit, et, mt, mf = get_advection_parameters()

    # Period for all tests
    Tf = 5

    # Let us test and compute the error
    for d in range(0,len(deps)):
        dp = deps[d]
        opsplit = split[d]
        ET = ets[d]
        MT = mts[d]
        MF = mfs[d]
        recon = recons[d]
        for i in range(0, Ntest):
            dt = dts[i]
            N = int(Nc[i])

            # Create CS mesh
            cs_grid = cubed_sphere(N, transformation, False, gridload)

            simulation = adv_simulation_par(cs_grid, dt, Tf, ic, vf, tc, recon, dp, opsplit, ET, MT, MF)

            # Create the latlon mesh (for plotting)
            ll_grid = latlon_grid(Nlat, Nlon)
            ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

            # Print parameters
            print('\nParameters: N='+str(int(Nc[i]))+', dt='+str(dts[i]),', recon=', simulation.recon_name,', split=', \
            simulation.opsplit_name, ', dp=', simulation.dp_name, ', et=', simulation.et_name)

            if run:
                # Get advection error
                error_linf[i,d], error_l1[i,d], error_l2[i,d] = adv_sphere(cs_grid, ll_grid, simulation, map_projection, False, False)

            else:
                # Open data
                # Filename
                filename = 'adv_Q_error_ic'+str(simulation.ic)+'_vf'+\
                str(simulation.vf)+'_'+simulation.opsplit_name+'_'+\
                simulation.recon_name+"_"+simulation.dp_name+'_'\
                +simulation.et_name+'_'+simulation.mt_name+'_'+simulation.mf_name+'_interp'+str(simulation.degree)
                # open txt files and get errors, CFL, mass change
                datatxt = np.loadtxt(datadir+cs_grid.name+'_'+filename)
                error_linf[i,d] = datatxt[0]
                error_l1[i,d]   = datatxt[1]
                error_l2[i,d]   = datatxt[2]
                simulation.CFL  = datatxt[3]
                simulation.mass_change = datatxt[4]
                simulation.dt  = datatxt[5]
                simulation.Tf  = datatxt[6]

                # plot the data
                print(datadir+cs_grid.name+'_'+filename)

                Q_error_ll = np.load(datadir+cs_grid.name+'_'+filename+'.npy',allow_pickle=True)

                colormap = 'seismic'
                qmax_abs = np.amax(abs(Q_error_ll))
                qmin = -qmax_abs
                qmax = qmax_abs
                cfl = str("{:.2e}".format(simulation.CFL))
                time = str("{:.2e}".format(simulation.Tf))
                title = "Error - Time="+time+', N='+str(cs_grid.N)+\
                ", ic="+str(simulation.ic)+", vf="+str(simulation.vf)+", CFL="+cfl+'\n '\
                +simulation.opsplit_name+', '+simulation.recon_name +', '+\
                simulation.dp_name+', '+simulation.et_name+', '+simulation.mt_name+', '+simulation.mf_name
 
                plot_scalar_field(Q_error_ll, filename, cs_grid, ll_grid, map_projection, \
                                      colormap, qmin, qmax, title)
 


            # Print errors
            print_errors_simul(error_linf[:,d], error_l1[:,d], error_l2[:,d], i)

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
        fname = []
        for d in range(0, len(deps)):
            errors.append(error[:,d])
            #fname.append(sp_names[split[d]-1]+'/'+recon_names[recons[d]-1]+'/'+et_names[ets[d]-1]+'/'+mt_names[mts[d]-1]+'/'\
            #+dp_names[deps[d]-1]+'/'+mf_names[mts[d]-1])
            fname.append('Scheme '+str(d+1))

        title = simulation.title + '- ic=' + str(simulation.ic)+', vf='+ str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_norm'+norm_list[e]+'_parabola_errors.pdf'

        plot_errors_loglog(Nc, errors, fname, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Convergence rate - ic=' + str(simulation.ic) +', vf=' + str(simulation.vf)+\
        ', dp='+dp_names[deps[d]-1]+', norm='+norm_title[e]
        filename = graphdir+'cs_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, fname, filename, title, CRmin, CRmax)
        e = e+1
