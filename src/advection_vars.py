####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - September 2022
####################################################################################

import numpy as np
from constants import*
from advection_ic           import velocity_adv, q0_adv
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid, ppm_parabola, velocity
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cfl                    import cfl_x, cfl_y
from lagrange               import lagrange_poly_ghostcell_pc, wind_edges2center_lagrange_poly, wind_center2ghostedges_lagrange_poly_ghost
from edges_treatment        import edges_ghost_cell_treatment_vector
from averaged_velocity      import time_averaged_velocity

####################################################################################
# This routine initializates the advection routine variables
####################################################################################
def init_vars_adv(cs_grid, simulation):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # other grid vars
    N = cs_grid.N
    ng = cs_grid.ng
    ngl = cs_grid.ngl
    ngr = cs_grid.ngr

    # Velocity at edges
    simulation.U_pu = velocity(cs_grid, 'pu')
    simulation.U_pv = velocity(cs_grid, 'pv')
    simulation.U_pc = velocity(cs_grid, 'pc')
        
    # Get velocities
    simulation.U_pu.ulon[i0:iend+1,j0:jend,:], simulation.U_pu.vlat[i0:iend+1,j0:jend,:] = velocity_adv\
    (cs_grid.pu.lon[i0:iend+1,j0:jend,:], cs_grid.pu.lat[i0:iend+1,j0:jend,:], 0.0, simulation)
    simulation.U_pv.ulon[i0:iend,j0:jend+1,:], simulation.U_pv.vlat[i0:iend,j0:jend+1,:] = velocity_adv\
    (cs_grid.pv.lon[i0:iend,j0:jend+1,:], cs_grid.pv.lat[i0:iend,j0:jend+1,:], 0.0, simulation)

    # Convert latlon to contravariant at pu
    simulation.U_pu.ucontra[i0:iend+1,j0:jend,:], simulation.U_pu.vcontra[i0:iend+1,j0:jend,:] = latlon_to_contravariant(\
    simulation.U_pu.ulon[i0:iend+1,j0:jend,:], simulation.U_pu.vlat[i0:iend+1,j0:jend,:], \
    cs_grid.prod_ex_elon_pu[i0:iend+1,j0:jend,:], cs_grid.prod_ex_elat_pu[i0:iend+1,j0:jend,:],\
    cs_grid.prod_ey_elon_pu[i0:iend+1,j0:jend,:], cs_grid.prod_ey_elat_pu[i0:iend+1,j0:jend,:], cs_grid.determinant_ll2contra_pu[i0:iend+1,j0:jend,:])
    
    # Convert latlon to contravariant at pv
    simulation.U_pv.ucontra[i0:iend,j0:jend+1,:], simulation.U_pv.vcontra[i0:iend,j0:jend+1,:] = latlon_to_contravariant(\
    simulation.U_pv.ulon[i0:iend,j0:jend+1,:], simulation.U_pv.vlat[i0:iend,j0:jend+1,:], 
    cs_grid.prod_ex_elon_pv[i0:iend,j0:jend+1,:], cs_grid.prod_ex_elat_pv[i0:iend,j0:jend+1,:],\
    cs_grid.prod_ey_elon_pv[i0:iend,j0:jend+1,:], cs_grid.prod_ey_elat_pv[i0:iend,j0:jend+1,:], cs_grid.determinant_ll2contra_pv[i0:iend,j0:jend+1,:])
    '''
    # Get velocities
    simulation.U_pu.ulon[:,:,:], simulation.U_pu.vlat[:,:,:] = velocity_adv(cs_grid.pu.lon, cs_grid.pu.lat, 0.0, simulation)
    simulation.U_pv.ulon[:,:,:], simulation.U_pv.vlat[:,:,:] = velocity_adv(cs_grid.pv.lon, cs_grid.pv.lat, 0.0, simulation)

    # Convert latlon to contravariant at pu
    simulation.U_pu.ucontra[:,:,:], simulation.U_pu.vcontra[:,:,:] = latlon_to_contravariant(simulation.U_pu.ulon, simulation.U_pu.vlat, cs_grid.prod_ex_elon_pu, cs_grid.prod_ex_elat_pu,\
                                                       cs_grid.prod_ey_elon_pu, cs_grid.prod_ey_elat_pu, cs_grid.determinant_ll2contra_pu)

    # Convert latlon to contravariant at pv
    simulation.U_pv.ucontra[:,:,:], simulation.U_pv.vcontra[:,:,:] = latlon_to_contravariant(simulation.U_pv.ulon, simulation.U_pv.vlat, cs_grid.prod_ex_elon_pv, cs_grid.prod_ex_elat_pv,\
                                                       cs_grid.prod_ey_elon_pv, cs_grid.prod_ey_elat_pv, cs_grid.determinant_ll2contra_pv)
    '''
    # Compute the Lagrange polynomials
    if cs_grid.projection=="gnomonic_equiangular":
        wind_edges2center_lagrange_poly(cs_grid, simulation)
        lagrange_poly_ghostcell_pc(cs_grid, simulation)
        wind_center2ghostedges_lagrange_poly_ghost(cs_grid, simulation)

    # Fill ghost cell - velocity field
    edges_ghost_cell_treatment_vector(simulation.U_pu, simulation.U_pv, \
        simulation.U_pc, cs_grid, simulation)

    simulation.U_pu.ucontra_old[:,:,:] = simulation.U_pu.ucontra[:,:,:]
    simulation.U_pv.vcontra_old[:,:,:] = simulation.U_pv.vcontra[:,:,:]

    # Initial departure points
    time_averaged_velocity(cs_grid, simulation)

    # CFL at edges - x direction
    simulation.cx = cfl_x(simulation.U_pu.ucontra, cs_grid, simulation)

    # CFL at edges - y direction
    simulation.cy = cfl_y(simulation.U_pv.vcontra, cs_grid, simulation)

    # CFL number
    CFL_x = np.amax(simulation.cx)
    CFL_y = np.amax(simulation.cy)
    simulation.CFL = max(abs(CFL_x),abs(CFL_y))

    # PPM parabolas
    simulation.px = ppm_parabola(cs_grid,simulation,'x')
    simulation.py = ppm_parabola(cs_grid,simulation,'y')

    # Compute average values of Q (initial condition) at cell pc
    simulation.Q[i0:iend,j0:jend,:] = q0_adv(cs_grid.pc.lon[i0:iend,j0:jend,:], cs_grid.pc.lat[i0:iend,j0:jend,:], simulation)
    
    return
