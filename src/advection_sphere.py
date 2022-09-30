####################################################################################
# Module for solving the advection equation using dimension splitting with PPM
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from diagnostics            import mass_computation, output_adv
from advection_vars         import init_vars_adv
from advection_timestep     import adv_time_step

def adv_sphere(cs_grid, ll_grid, simulation, map_projection, plot):
    N  = cs_grid.N       # Number of cells in x direction
    nghost = cs_grid.nghost   # Number o ghost cells
    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition

    # Plot step
    #plotstep = int(Nsteps/1000)
    plotstep = 100

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Initialize the variables
    Q_new, Q_old, Q, q_exact, flux_x, flux_y, ax, ay, cx, cy, cx2, cy2, \
    error_linf, error_l1, error_l2, F_gQ, G_gQ, GF_gQ, FG_gQ, \
    ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
    g_metric = init_vars_adv(cs_grid, simulation, N, nghost, Nsteps)

    # Compute initial mass
    total_mass0, _ = mass_computation(Q.f, cs_grid, 1.0)

    # Initial plot
    output_adv(cs_grid, ll_grid, simulation, Q, Q_old, q_exact, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, error_linf, error_l1, error_l2, plot, 0, 0, Nsteps, plotstep, total_mass0, map_projection)

    # Time looping
    for k in range(1, Nsteps+1):
        # Current time
        t = k*dt

        # Compute a timestep
        adv_time_step(cs_grid, simulation, g_metric, Q_old, Q_new, k, dt, t, ax, ay, cx, cx2, cy, cy2, \
                      flux_x, flux_y, ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy,\
                      F_gQ, G_gQ, GF_gQ, FG_gQ)

        # Output
        output_adv(cs_grid, ll_grid, simulation, Q, Q_new, q_exact,  ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, map_projection)

        # Update
        Q_old = Q_new
    #-----------------------------End of time loop---------------------
    return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
