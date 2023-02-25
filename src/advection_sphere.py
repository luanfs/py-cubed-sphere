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
    ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
    ulon_edx, vlat_edx, ulon_edy, vlat_edy, \
    lagrange_poly, Kmin, Kmax, CFL \
    = init_vars_adv(cs_grid, simulation, transformation)

    # Compute initial mass
    total_mass0, _ = mass_computation(Q, cs_grid, 1.0)

    # Initial plot
    output_adv(cs_grid, ll_grid, simulation, Q, div,\
                   ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy,\
                   error_linf, error_l1, error_l2, plot, 0, 0.0, Nsteps,\
                   plotstep, total_mass0, map_projection, CFL, divtest_flag)


    # Time looping
    for k in range(1, Nsteps+1):
        # Current time
        t = k*dt

        # Compute a timestep
        adv_time_step(cs_grid, simulation, Q, gQ, div, px, py, cx, cy, \
                  ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
                  transformation, lagrange_poly, Kmin, Kmax, k, t)

        # Output
        output_adv(cs_grid, ll_grid, simulation, Q, div,\
                   ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy,\
                   error_linf, error_l1, error_l2, plot, k, t, Nsteps,\
                   plotstep, total_mass0, map_projection, CFL, divtest_flag)

    #-----------------------------End of time loop---------------------
    return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
