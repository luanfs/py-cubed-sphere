####################################################################################
#
# Module for diagnostic computation and output routines
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from errors        import compute_errors
from advection_ic  import qexact_adv
from interpolation import ll2cs, nearest_neighbour
from plot          import plot_scalar_and_vector_field
from sphgeo        import contravariant_to_latlon

####################################################################################
# Total mass computation
####################################################################################
def mass_computation(Q, cs_grid, total_mass0):
    # Interior cells index (we are ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend
    total_mass =  np.sum(Q*cs_grid.areas[i0:iend,j0:jend,:])  # Compute new mass
    if abs(total_mass0)>10**(-10):
        mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
    else:
        mass_change = abs(total_mass0-total_mass)
    return total_mass, mass_change

####################################################################################
# Print the advectioc diagnostics variables on the screen
####################################################################################
def print_diagnostics_adv(error_linf, error_l1, error_l2, mass_change, t, Nsteps):
    print('\nStep', t, 'from', Nsteps)
    print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf), "{:.2e}".format(error_l1), "{:.2e}".format(error_l2))
    print('Total mass variation:', "{:.2e}".format(mass_change))

####################################################################################
# Output and plot
####################################################################################
def output_adv(cs_grid, ll_grid, simulation, Q, Q_new, q_exact, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, map_projection):
    if plot or k==Nsteps:
        # Interior cells index (ignoring ghost cells)
        i0   = cs_grid.i0
        iend = cs_grid.iend
        j0   = cs_grid.j0
        jend = cs_grid.jend

        q_exact.f[:,:,:] = qexact_adv(cs_grid.centers.lon[i0:iend,j0:jend,:], cs_grid.centers.lat[i0:iend,j0:jend,:], t, simulation)
        Q.f[:,:,:] = Q_new[i0:iend,j0:jend,:]

        # Relative errors in different metrics
        error_linf[k], error_l1[k], error_l2[k] = compute_errors(Q.f, q_exact.f)

        # Diagnostic - mass
        total_mass, mass_change = mass_computation(Q.f, cs_grid, total_mass0)

        if plot and k>0:
            # Print diagnostics on the screen
            print_diagnostics_adv(error_linf[k], error_l1[k], error_l2[k], mass_change, k, Nsteps)

        # Plot the solution
        if k%plotstep==0 or k==0 or k==Nsteps:
            # Convert contravariant to latlon
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

            # Interpolate to the latlon grid and plot
            # Q exact
            Q_exact_ll = nearest_neighbour(q_exact, cs_grid, ll_grid)
            colormap = 'jet'
            #if simulation.ic>=2:
            #    qmin = -0.2
            #    qmax =  1.2
            #    colormap = 'jet'
            #    plot_scalar_and_vector_field(Q_exact_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, 'adv_'+q_exact.name+'_ic'+str(simulation.ic)+"_t"+str(k), cs_grid, ll_grid, map_projection, colormap, qmin, qmax)

            # Q
            Q_ll = nearest_neighbour(Q, cs_grid, ll_grid)
            if simulation.ic>=2:
                if simulation.vf != 5:
                    qmin = -0.2
                    qmax =  1.2
                else:
                    qmin = -0.2
                    qmax =  3.5
                colormap = 'jet'
                plot_scalar_and_vector_field(Q_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, 'adv_'+Q.name+'_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+"_t"+str(k), cs_grid, ll_grid, map_projection, colormap, qmin, qmax)

            # Q error
            Q_error_ll =  Q_exact_ll - Q_ll
            colormap = 'seismic'
            qmax_abs = np.amax(abs(Q_error_ll))
            qmin = -qmax_abs
            qmax =  qmax_abs
            if t>0:
                plot_scalar_and_vector_field(Q_error_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, 'adv_'+'Q_error'+'_ic'+str(simulation.ic)+'_vf'+str(simulation.vf)+"_t"+str(k), cs_grid, ll_grid, map_projection, colormap, qmin, qmax)
