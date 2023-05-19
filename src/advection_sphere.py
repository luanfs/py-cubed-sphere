####################################################################################
# Module for solving the advection equation using dimension splitting with PPM
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from diagnostics            import mass_computation
from output                 import output_adv
from advection_vars         import init_vars_adv
from advection_timestep     import adv_time_step, update_adv

def adv_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, plot, divtest_flag):
    # Simulation parameters
    dt = simulation.dt # Time step
    Tf = simulation.Tf # Final time

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Plot step
    plotstep = int(Nsteps/2)

    if (divtest_flag):
        Nsteps = 1
        plotstep = 1

    # Error variables
    error_linf, error_l1, error_l2 = np.zeros(Nsteps+1), np.zeros(Nsteps+1), np.zeros(Nsteps+1)

    # Initialize the variables
    Q, gQ, div, px, py, cx, cy, \
    U_pu, U_pv, lagrange_poly_ghost_pc, stencil_ghost_pc, \
    lagrange_poly_edge, stencil_edge, \
    lagrange_poly_ghost_edge, stencil_ghost_edge, CFL \
     = init_vars_adv(cs_grid, simulation, transformation)

    # Compute initial mass
    total_mass0, _ = mass_computation(Q, cs_grid, 1.0)
    if (not divtest_flag):
        # Initial plot
        output_adv(cs_grid, ll_grid, simulation, Q, div, U_pu, U_pv,\
                   error_linf, error_l1, error_l2, plot, 0, 0.0, Nsteps,\
                   plotstep, total_mass0, map_projection, CFL, divtest_flag)

    # Time looping
    for k in range(1, Nsteps+1):
        # Current time
        t = k*dt

        # Compute a timestep
        adv_time_step(cs_grid, simulation, Q, gQ, div, px, py, cx, cy, \
                      U_pu, U_pv, transformation, lagrange_poly_ghost_pc, stencil_ghost_pc, k, t)
        # Output
        output_adv(cs_grid, ll_grid, simulation, Q, div, U_pu, U_pv,\
                   error_linf, error_l1, error_l2, plot, k, t, Nsteps,\
                   plotstep, total_mass0, map_projection, CFL, divtest_flag)

        # Updates for next time step
        update_adv(cs_grid, simulation, t, cx, cy, U_pu, U_pv)
    #-----------------------------End of time loop---------------------

    return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
