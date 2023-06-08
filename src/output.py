import numpy as np
from errors        import compute_errors
from advection_ic  import qexact_adv, div_exact
from interpolation import ll2cs, nearest_neighbour
from plot          import plot_scalar_field, plot_scalar_and_vector_field
from sphgeo        import contravariant_to_latlon
from diagnostics   import mass_computation
from cs_datastruct import scalar_field

###################################################################################
# Print the advectioc diagnostics variables on the screen
####################################################################################
def print_diagnostics_adv(error_linf, error_l1, error_l2, mass_change, t, Nsteps):
    print('\nStep', t, 'from', Nsteps)
    print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf), "{:.2e}".format(error_l1), "{:.2e}".format(error_l2))
    print('Total mass variation:', "{:.2e}".format(mass_change))

####################################################################################
# Output and plot for advection model
####################################################################################
def output_adv(cs_grid, ll_grid, simulation,\
               plot, k, t, Nsteps,\
               plotstep, map_projection, divtest_flag):

    if plot or k==Nsteps:
        # Interior cells index (ignoring ghost cells)
        i0   = cs_grid.i0
        iend = cs_grid.iend
        j0   = cs_grid.j0
        jend = cs_grid.jend

        # Exact field
        q_exact = scalar_field(cs_grid, 'q_exact', 'center')
        q = scalar_field(cs_grid, 'q', 'center')
        q_exact.f[:,:,:] = qexact_adv(cs_grid.pc.lon[i0:iend,j0:jend,:], cs_grid.pc.lat[i0:iend,j0:jend,:], t, simulation)
        #Q[i0:iend,j0:jend,:] = q_exact.f[:,:,:]
        q.f[:,:,:] = simulation.Q[i0:iend,j0:jend,:]

        # Relative errors in different metrics
        simulation.error_linf[k], simulation.error_l1[k], simulation.error_l2[k] =\
        compute_errors(q.f, q_exact.f)

        if(simulation.error_linf[k]>100.0):
            print('Stopping due to large errors.')
            print('The CFL number is:', simulation.CFL)
            exit()

        # Diagnostic - mass
        simulation.total_mass, simulation.mass_change = mass_computation(simulation.Q, cs_grid, simulation.total_mass0)

        if k>0 and (not divtest_flag):
            # Print diagnostics on the screen
            print_diagnostics_adv(simulation.error_linf[k], simulation.error_l1[k], simulation.error_l2[k], simulation.mass_change, k, Nsteps)

        # Plot the solution
        if k%plotstep==0 or k==0 or k==Nsteps:
            # Convert contravariant to latlon at ed_x
            ex_elon_pu = cs_grid.prod_ex_elon_pu
            ex_elat_pu = cs_grid.prod_ex_elat_pu
            ey_elon_pu = cs_grid.prod_ey_elon_pu
            ey_elat_pu = cs_grid.prod_ey_elat_pu
            simulation.U_pu.ulon, simulation.U_pu.vlat = contravariant_to_latlon(\
            simulation.U_pu.ucontra, simulation.U_pu.vcontra, \
            ex_elon_pu, ex_elat_pu, ey_elon_pu, ey_elat_pu)

            # Convert latlon to contravariant at ed_y
            ex_elon_pv = cs_grid.prod_ex_elon_pv
            ex_elat_pv = cs_grid.prod_ex_elat_pv
            ey_elon_pv = cs_grid.prod_ey_elon_pv
            ey_elat_pv = cs_grid.prod_ey_elat_pv
            simulation.U_pv.ulon, simulation.U_pv.vlat = contravariant_to_latlon(\
            simulation.U_pv.ucontra, simulation.U_pv.vcontra, \
            ex_elon_pv, ex_elat_pv, ey_elon_pv, ey_elat_pv)

            # Plot scalar field
            if (not divtest_flag):
                # Interpolate to the latlon grid and plot
                Q_exact_ll = nearest_neighbour(q_exact, cs_grid, ll_grid)
                Q_ll = nearest_neighbour(q, cs_grid, ll_grid)

                if simulation.vf != 5:
                    qmin = -0.3
                    qmax =  1.3
                else:
                    qmin = -0.2
                    qmax =  3.5
                #if simulation.ic == 1:
                #    qmin = 0.99
                #    qmax = 1.01

                colormap = 'jet'
                q_min = str("{:.2e}".format(np.amin(q.f)))
                q_max = str("{:.2e}".format(np.amax(q.f)))
                time = str("{:.2e}".format(t))
                cfl = str("{:.2e}".format(simulation.CFL))

                # filename
                filename = 'adv_Q_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+\
                            "_"+simulation.opsplit_name+"_"+simulation.recon_name+"_"+simulation.dp_name+'_'+simulation.et_name+\
                            "_interp"+str(simulation.degree)+"_t"+str(k)

                # Title
                title = "Min="+q_min+", Max="+q_max+", Time="+time+', N='+str(cs_grid.N)+\
                ", ic="+str(simulation.ic)+", vf="+str(simulation.vf)+", CFL="+cfl+'\n '\
                +simulation.opsplit_name+', '+simulation.recon_name +', '+simulation.dp_name+', '+simulation.et_name+\
                ', Interpolation degree='+str(simulation.degree)+'\n' 

                plot_scalar_field(Q_ll, filename, cs_grid, ll_grid, map_projection, \
                                  colormap, qmin, qmax, title)

                #plot_scalar_and_vector_field(Q_ll, U_pu.ulon, U_pu.vlat, U_pv.ulon, U_pv.vlat,\
                #                             filename, title, cs_grid, ll_grid, map_projection, \
                #                             colormap, qmin, qmax)

                # Q error - only for t>0
                if t>0:
                    Q_error_ll =  Q_exact_ll - Q_ll
                    colormap = 'seismic'
                    qmax_abs = np.amax(abs(Q_error_ll))
                    qmin = -qmax_abs
                    qmax = qmax_abs
                    time = str("{:.2e}".format(t))
                    # filename
                    filename = 'adv_Q_error_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+\
                                "_"+simulation.opsplit_name+"_"+simulation.recon_name+"_"+simulation.dp_name+'_'+simulation.et_name+\
                                "_interp"+str(simulation.degree)+"_t"+str(k)
                    title = "Error - Time="+time+', N='+str(cs_grid.N)+\
                    ", ic="+str(simulation.ic)+", vf="+str(simulation.vf)+", CFL="+cfl+'\n '\
                    +simulation.opsplit_name+', '+simulation.recon_name +', '+simulation.dp_name+', '+simulation.et_name+\
                    ', Interpolation degree='+str(simulation.degree)+'\n'

                    plot_scalar_field(Q_error_ll, filename, cs_grid, ll_grid, map_projection, \
                                      colormap, qmin, qmax, title)
                    #plot_scalar_and_vector_field(Q_error_ll, U_pu.ulon, U_pu.vlat, U_pv.ulon, U_pv.vlat, \
                    #                             filename, title, cs_grid, ll_grid, map_projection, \
                    #                             colormap, qmin, qmax)

            else:
                # Divergence plotting
                d = scalar_field(cs_grid, 'div', 'center')
                dex = scalar_field(cs_grid, 'div_exact', 'center')
                d.f[:,:,:] = simulation.div[i0:iend,j0:jend,:]
                dex.f[:,:,:] = div_exact(cs_grid.pc.lon[i0:iend,j0:jend,:], cs_grid.pc.lat[i0:iend,j0:jend,:], simulation)
                d_ll = nearest_neighbour(d, cs_grid, ll_grid)
                dex_ll = nearest_neighbour(dex, cs_grid, ll_grid)
                error_d = dex_ll - d_ll

                # Plot parameters
                colormap = 'seismic'
                dmax_abs = np.amax(abs(error_d))
                dmin, dmax = -dmax_abs, dmax_abs
                cfl = str("{:.2e}".format(simulation.CFL))

                # Relative errors in different metrics
                simulation.error_linf[k], simulation.error_l1[k], simulation.error_l2[k] = compute_errors(d.f, dex.f)

                filename = 'div_error'+'_vf'+str(simulation.vf)+'_'+simulation.opsplit_name+'_'+simulation.recon_name+'_'+simulation.dp_name+'_'+simulation.et_name
                title = "Divergence error"+', N='+str(cs_grid.N)+", vf="+str(simulation.vf)+", CFL="+cfl+', '\
                +simulation.opsplit_name+', '+simulation.recon_name+', '+simulation.dp_name+', '+simulation.et_name+'\n'

                plot_scalar_field(error_d, filename, cs_grid, ll_grid, map_projection, \
                colormap, dmin, dmax, title)

                if simulation.vf >= 5: # Plot the divergence
                    colormap = 'jet'
                    dmin, dmax = np.amin(d_ll), np.amax(d_ll)
                    filename = 'div_vf'+str(simulation.vf)+'_'+simulation.opsplit_name+'_'+simulation.recon_name+'_'+simulation.dp_name+'_'+simulation.et_name
                    title = "Divergence"+', N='+str(cs_grid.N)+", vf = "+str(simulation.vf)+", CFL = "+cfl+', '\
                    +simulation.opsplit_name+', '+simulation.recon_name+', '+simulation.dp_name+', '+simulation.et_name+'\n'
                    plot_scalar_field(d_ll, filename, cs_grid, ll_grid, map_projection, \
                                      colormap, dmin, dmax, title)


