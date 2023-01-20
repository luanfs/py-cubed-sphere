####################################################################################
# Module for solving the advection equation using dimension splitting with PPM
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from diagnostics            import mass_computation
from output                 import output_adv
from advection_vars         import init_vars_adv
from advection_timestep     import adv_time_step
from advection_ic           import velocity_adv
from cfl                    import cfl_x, cfl_y
from sphgeo                 import latlon_to_contravariant

def adv_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, plot):
    N  = cs_grid.N       # Number of cells in x direction
    nghost = cs_grid.nghost   # Number o ghost cells
    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Plot step
    plotstep = int(Nsteps/5)

    # Initialize the variables
    if simulation.vf <= 2:
        Q_new, Q_old, Q, q_exact, flux_x, flux_y, cx, cy, \
        error_linf, error_l1, error_l2, F_gQ, G_gQ, GF_gQ, FG_gQ, \
        ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
        g_metric, lagrange_poly, Kmin, Kmax\
        = init_vars_adv(cs_grid, simulation, transformation, N, nghost, Nsteps)

    elif simulation.vf >= 3: # Extra variables needed only for time dependent velocity
        Q_new, Q_old, Q, q_exact, flux_x, flux_y, cx, cy, \
        error_linf, error_l1, error_l2, F_gQ, G_gQ, GF_gQ, FG_gQ, \
        ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
        g_metric,\
        lagrange_poly, Kmin, Kmax,\
        ex_elon_edx, ex_elat_edx, ey_elon_edx, ey_elat_edx, det_edx, \
        ex_elon_edy, ex_elat_edy, ey_elon_edy, ey_elat_edy, det_edy, \
        ulon_edx, vlat_edx, edx_lon, edx_lat, \
        ulon_edy, vlat_edy, edy_lon, edy_lat \
        = init_vars_adv(cs_grid, simulation, transformation, N, nghost, Nsteps)

    # Compute initial mass
    total_mass0, _ = mass_computation(Q.f, cs_grid, 1.0)

    # Initial plot
    output_adv(cs_grid, ll_grid, simulation, Q, Q_old, q_exact, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, error_linf, error_l1, error_l2, plot, 0, 0, Nsteps, plotstep, total_mass0, map_projection)

    # Time looping
    for k in range(1, Nsteps+1):
        # Current time
        t = k*dt

        # Compute a timestep
        adv_time_step(cs_grid, simulation, g_metric, Q_old, Q_new, k, dt, t, cx, cy, \
                      flux_x, flux_y, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy,\
                      F_gQ, G_gQ, GF_gQ, FG_gQ, transformation,\
                      lagrange_poly, Kmin, Kmax)

        # Output
        output_adv(cs_grid, ll_grid, simulation, Q, Q_new, q_exact, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, map_projection)

    #-----------------------------End of time loop---------------------
    return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
