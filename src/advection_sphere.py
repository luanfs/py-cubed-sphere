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

def adv_sphere(cs_grid, ll_grid, simulation, map_projection, plot, divtest_flag):
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
    simulation.error_linf, simulation.error_l1, simulation.error_l2 = \
    np.zeros(Nsteps+1), np.zeros(Nsteps+1), np.zeros(Nsteps+1)

    # Initialize the variables
    init_vars_adv(cs_grid, simulation)

    # Compute initial mass
    simulation.total_mass0, _ = mass_computation(simulation.Q, cs_grid, 1.0)

    if (not divtest_flag):
        # Initial plot
        output_adv(cs_grid, ll_grid, simulation, plot, 0, 0.0, Nsteps,\
                   plotstep, map_projection, divtest_flag)


    # Time looping
    for k in range(1, Nsteps+1):
        # Current time
        t = k*dt

        # Compute a timestep
        adv_time_step(cs_grid, simulation, k, t)

        # Output
        output_adv(cs_grid, ll_grid, simulation, plot, k, t, Nsteps,\
                   plotstep, map_projection, divtest_flag)

        # Updates for next time step
        update_adv(cs_grid, simulation, t)

    #-----------------------------End of time loop---------------------

    return simulation.error_linf[Nsteps], simulation.error_l1[Nsteps], simulation.error_l2[Nsteps]
