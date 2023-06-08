####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

from advection_ic import velocity_adv
from sphgeo       import latlon_to_contravariant
import numexpr as ne
import numpy   as np
from edges_treatment import edges_ghost_cell_treatment_vector

def time_averaged_velocity(cs_grid, simulation):

    i0, iend = cs_grid.i0, cs_grid.iend
    j0, jend = cs_grid.j0, cs_grid.jend
    N = cs_grid.N
    ng = cs_grid.ng

    # Compute the velocity needed for the departure point
    if simulation.dp_name == 'RK1':
        simulation.U_pu.ucontra_averaged[:,:,:] = simulation.U_pu.ucontra[:,:,:]
        simulation.U_pv.vcontra_averaged[:,:,:] = simulation.U_pv.vcontra[:,:,:]

    elif simulation.dp_name == 'RK2':
        # Fill ghost cells - velocity field
        #edges_ghost_cell_treatment_vector(U_pu, U_pv, U_pc, cs_grid, simulation,\
        #lagrange_poly_edge, stencil_edge, lagrange_poly_ghost_pc, stencil_ghost_pc, \
        #lagrange_poly_ghost_edge, stencil_ghost_edge)
        dt = simulation.dt
        dto2 = simulation.dto2

        #----------------------------------------------------
        # Velocity data at edges used for interpolation
        u_interp = 1.5*simulation.U_pu.ucontra[:,:,:] - 0.5*simulation.U_pu.ucontra_old[:,:,:] # extrapolation for time at n+1/2
        # Linear interpolation
        a = simulation.U_pu.ucontra[i0:iend+1,:,:]*dto2/cs_grid.dx
        upos = simulation.U_pu.ucontra[i0:iend+1,:,:]>=0
        uneg = ~upos
        simulation.U_pu.ucontra_averaged[i0:iend+1,:,:][upos] = (1.0-a[upos])*u_interp[i0:iend+1,:,:][upos] + a[upos]*u_interp[i0-1:iend,:,:][upos]
        simulation.U_pu.ucontra_averaged[i0:iend+1,:,:][uneg] = -a[uneg]*u_interp[i0+1:iend+2,:,:][uneg] + (1.0+a[uneg])*u_interp[i0:iend+1,:,:][uneg]

        #----------------------------------------------------
        # Velocity data at edges used for interpolation
        v_interp = 1.5*simulation.U_pv.vcontra[:,:,:] - 0.5*simulation.U_pv.vcontra_old[:,:,:] # extrapolation for time at n+1/2
        # Linear interpolation
        a = simulation.U_pv.vcontra[:,j0:jend+1,:]*dto2/cs_grid.dy
        vpos = simulation.U_pv.vcontra[:,j0:jend+1,:]>=0
        vneg = ~vpos
        simulation.U_pv.vcontra_averaged[:,j0:jend+1,:][vpos] = (1.0-a[vpos])*v_interp[:,j0:jend+1,:][vpos] + a[vpos]*v_interp[:,j0-1:jend,:][vpos]
        simulation.U_pv.vcontra_averaged[:,j0:jend+1,:][vneg] = -a[vneg]*v_interp[:,j0+1:jend+2,:][vneg] + (1.0+a[vneg])*v_interp[:,j0:jend+1,:][vneg]
