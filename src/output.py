import numpy as np
from errors        import compute_errors
from advection_ic  import qexact_adv, div_exact
from interpolation import ll2cs, nearest_neighbour
from plot          import plot_scalar_and_vector_field
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
def output_adv(cs_grid, ll_grid, simulation, Q, div, \
               ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy,\
               error_linf, error_l1, error_l2, plot, k, t, Nsteps,\
               plotstep, total_mass0, map_projection, CFL, divtest_flag):
    if plot or k==Nsteps:
        # Interior cells index (ignoring ghost cells)
        i0   = cs_grid.i0
        iend = cs_grid.iend
        j0   = cs_grid.j0
        jend = cs_grid.jend

        # Exact field
        q_exact = scalar_field(cs_grid, 'q_exact', 'center')
        q = scalar_field(cs_grid, 'q', 'center')
        q_exact.f[:,:,:] = qexact_adv(cs_grid.centers.lon[i0:iend,j0:jend,:], cs_grid.centers.lat[i0:iend,j0:jend,:], t, simulation)
        #Q[i0:iend,j0:jend,:] = q_exact.f[:,:,:]
        q.f[:,:,:] = Q[i0:iend,j0:jend,:]

        # Relative errors in different metrics
        error_linf[k], error_l1[k], error_l2[k] = compute_errors(q.f, q_exact.f)

        if(error_linf[k]>100.0):
            print('Stopping due to large errors.')
            print('The CFL number is:', CFL)
            exit()

        # Diagnostic - mass
        total_mass, mass_change = mass_computation(Q, cs_grid, total_mass0)

        if k>0 and (not divtest_flag):
            # Print diagnostics on the screen
            print_diagnostics_adv(error_linf[k], error_l1[k], error_l2[k], mass_change, k, Nsteps)

        # Plot the solution
        if k%plotstep==0 or k==0 or k==Nsteps:
            # Convert contravariant to latlon at ed_x
            ex_elon_edx = cs_grid.prod_ex_elon_edx
            ex_elat_edx = cs_grid.prod_ex_elat_edx
            ey_elon_edx = cs_grid.prod_ey_elon_edx
            ey_elat_edx = cs_grid.prod_ey_elat_edx
            ulon_edx, vlat_edx = contravariant_to_latlon(ucontra_edx, vcontra_edx, ex_elon_edx, ex_elat_edx, ey_elon_edx, ey_elat_edx)

            # Convert latlon to contravariant at ed_y
            ex_elon_edy = cs_grid.prod_ex_elon_edy
            ex_elat_edy = cs_grid.prod_ex_elat_edy
            ey_elon_edy = cs_grid.prod_ey_elon_edy
            ey_elat_edy = cs_grid.prod_ey_elat_edy
            ulon_edy, vlat_edy = contravariant_to_latlon(ucontra_edy, vcontra_edy, ex_elon_edy, ex_elat_edy, ey_elon_edy, ey_elat_edy)


            # Plot scalar field
            if (not divtest_flag):
                # Interpolate to the latlon grid and plot
                Q_exact_ll = nearest_neighbour(q_exact, cs_grid, ll_grid)
                Q_ll = nearest_neighbour(q, cs_grid, ll_grid)

                if simulation.vf != 5:
                    qmin = -0.2
                    qmax =  1.2
                else:
                    qmin = -0.2
                    qmax =  3.5
                if simulation.ic == 1:
                    qmin = 0.99
                    qmax = 1.01

                colormap = 'jet'
                q_min = str("{:.2e}".format(np.amin(q.f)))
                q_max = str("{:.2e}".format(np.amax(q.f)))
                time = str("{:.2e}".format(t))
                cfl = str("{:.2e}".format(CFL))

                # filename
                filename = 'adv_Q_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+\
                            "_"+simulation.opsplit_name+"_"+simulation.recon_name+\
                            "_interp"+str(simulation.degree)+"_t"+str(k)

                # Title
                title = "Min = "+q_min+", Max = "+q_max+", Time = "+time+', N='+str(cs_grid.N)+\
                ", ic ="+str(simulation.ic)+", vf = "+str(simulation.vf)+", CFL = "+cfl+'\n '\
                +simulation.opsplit_name+', '+simulation.recon_name +\
                ', Interpolation degree = '+str(simulation.degree)

                plot_scalar_and_vector_field(Q_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy,\
                                             filename, title, cs_grid, ll_grid, map_projection, \
                                             colormap, qmin, qmax)

                # Q error - only for t>0
                if t>0:
                    Q_error_ll =  Q_exact_ll - Q_ll
                    colormap = 'seismic'
                    qmax_abs = np.amax(abs(Q_error_ll))
                    qmin = -qmax_abs
                    qmax =  qmax_abs
                    time = str("{:.2e}".format(t))
                    filename = 'adv_Q_error'+'_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+\
                               "_interpol"+str(simulation.degree)+"_"+str(simulation.recon_name)+"_t"+str(k)
                    title = "Error - Time = "+time+', N='+str(cs_grid.N)+\
                    ", ic ="+str(simulation.ic)+", vf = "+str(simulation.vf)+", CFL = "+cfl+'\n '\
                    +simulation.opsplit_name+', '+simulation.recon_name +\
                    ', Interpolation degree = '+str(simulation.degree)
                    plot_scalar_and_vector_field(Q_error_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, \
                                                 filename, title, cs_grid, ll_grid, map_projection, \
                                                 colormap, qmin, qmax)

            else:
                # Divergence plotting
                d = scalar_field(cs_grid, 'div', 'center')
                dex = scalar_field(cs_grid, 'div_exact', 'center')
                d.f[:,:,:] = div[i0:iend,j0:jend,:]
                dex.f[:,:,:] = div_exact(cs_grid.centers.lon[i0:iend,j0:jend,:], cs_grid.centers.lat[i0:iend,j0:jend,:], simulation)
                d_ll = nearest_neighbour(d, cs_grid, ll_grid)
                dex_ll = nearest_neighbour(dex, cs_grid, ll_grid)
                error_d = dex_ll - d_ll

                # Plot parameters
                colormap = 'seismic'
                dmin, dmax = np.amin(error_d), np.amax(error_d)
                cfl = str("{:.2e}".format(CFL))

                # Relative errors in different metrics
                error_linf[k], error_l1[k], error_l2[k] = compute_errors(d.f, dex.f)

                filename = 'div_error'+'_vf'+str(simulation.vf)+'_'+simulation.opsplit_name+'_'+simulation.recon_name
                title = "Divergence error"+', N='+str(cs_grid.N)+", vf = "+str(simulation.vf)+", CFL = "+cfl+', '\
                +simulation.opsplit_name+', '+simulation.recon_name
                plot_scalar_and_vector_field(error_d, ulon_edx, vlat_edx, ulon_edy, vlat_edy, \
                                             filename, title, cs_grid, ll_grid, map_projection, \
                                             colormap, dmin, dmax)

                if simulation.vf >= 5: # Plot the divergence
                    colormap = 'jet'
                    dmin, dmax = np.amin(d_ll), np.amax(d_ll)
                    filename = 'div_vf'+str(simulation.vf)+'_'+simulation.opsplit_name+'_'+simulation.recon_name
                    title = "Divergence"+', N='+str(cs_grid.N)+", vf = "+str(simulation.vf)+", CFL = "+cfl+', '\
                    +simulation.opsplit_name+', '+simulation.recon_name
                    plot_scalar_and_vector_field(d_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, \
                                                 filename, title, cs_grid, ll_grid, map_projection, \
                                                 colormap, dmin, dmax)


