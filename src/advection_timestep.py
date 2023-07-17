####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from flux                   import compute_fluxes
from advection_ic           import velocity_adv
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from discrete_operators     import divergence
from averaged_velocity      import time_averaged_velocity
from edges_treatment        import edges_ghost_cell_treatment_scalar, edges_ghost_cell_treatment_vector

####################################################################################
# This routine computes one advection timestep
####################################################################################
def adv_time_step(cs_grid, simulation, k, t):

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Fill ghost cell values - scalar field
    edges_ghost_cell_treatment_scalar(simulation.Q, simulation.Q, cs_grid, simulation)

    # Updates in velocity - only for time dependent velocity
    if simulation.vf >= 2:   
        # Fill ghost cell values - velocity field
        edges_ghost_cell_treatment_vector(simulation.U_pu, simulation.U_pv, \
        simulation.U_pc, cs_grid, simulation)

        #Compute the velocity need for the departure point
        time_averaged_velocity(cs_grid, simulation)

    # Compute the divergence
    divergence(cs_grid, simulation)

    # Q update
    simulation.Q[i0:iend,j0:jend,:] = simulation.Q[i0:iend,j0:jend,:] - simulation.dt*simulation.div[i0:iend,j0:jend,:]

####################################################################################
# This routine computes advection updates for the next timestep
####################################################################################
def update_adv(cs_grid, simulation, t):
    # Updates for next time step - only for time dependent velocity
    if simulation.vf >= 2:
        i0, iend = cs_grid.i0, cs_grid.iend
        j0, jend = cs_grid.j0, cs_grid.jend

        # Velocity
        simulation.U_pu.ulon[i0:iend+1,j0:jend,:], simulation.U_pu.vlat[i0:iend+1,j0:jend,:] = \
        velocity_adv(cs_grid.pu.lon[i0:iend+1,j0:jend,:],cs_grid.pu.lat[i0:iend+1,j0:jend,:], t, simulation)
        simulation.U_pv.ulon[i0:iend,j0:jend+1,:], simulation.U_pv.vlat[i0:iend,j0:jend+1,:] = \
        velocity_adv(cs_grid.pv.lon[i0:iend,j0:jend+1,:],cs_grid.pv.lat[i0:iend,j0:jend+1,:], t, simulation)

        # Store old velocity
        simulation.U_pu.ucontra_old[:,:,:] = simulation.U_pu.ucontra[:,:,:]
        simulation.U_pv.vcontra_old[:,:,:] = simulation.U_pv.vcontra[:,:,:]

        # Latlon to contravariant
        simulation.U_pu.ucontra[:,:,:], simulation.U_pu.vcontra[:,:,:] = \
        latlon_to_contravariant(simulation.U_pu.ulon[:,:,:], simulation.U_pu.vlat[:,:,:],\
        cs_grid.prod_ex_elon_pu[:,:,:], cs_grid.prod_ex_elat_pu[:,:,:], \
        cs_grid.prod_ey_elon_pu[:,:,:], cs_grid.prod_ey_elat_pu[:,:,:], \
        cs_grid.determinant_ll2contra_pu[:,:,:])

        simulation.U_pv.ucontra[:,:,:], simulation.U_pv.vcontra[:,:,:] = \
        latlon_to_contravariant(simulation.U_pv.ulon[:,:,:], simulation.U_pv.vlat[:,:,:],\
        cs_grid.prod_ex_elon_pv[:,:,:], cs_grid.prod_ex_elat_pv[:,:,:], \
        cs_grid.prod_ey_elon_pv[:,:,:], cs_grid.prod_ey_elat_pv[:,:,:], \
        cs_grid.determinant_ll2contra_pv[:,:,:])

