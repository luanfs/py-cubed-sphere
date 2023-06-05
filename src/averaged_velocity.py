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
from edges_treatment    import edges_ghost_cell_treatment_vector

def time_averaged_velocity(U_pu, U_pv, cs_grid, simulation):
    i0, iend = cs_grid.i0, cs_grid.iend
    j0, jend = cs_grid.j0, cs_grid.jend
    N = cs_grid.N
    ng = cs_grid.ng

    # Fill ghost cell - velocity field
    edges_ghost_cell_treatment_vector(U_pu.ucontra, U_pv.vcontra, cs_grid, simulation)

    # Compute the velocity needed for the departure point
    if simulation.dp_name == 'RK1':
        U_pu.ucontra_averaged[:,:,:] = U_pu.ucontra[:,:,:]
        U_pv.vcontra_averaged[:,:,:] = U_pv.vcontra[:,:,:]

    elif simulation.dp_name == 'RK2':
        dt = simulation.dt
        dto2 = simulation.dto2

        #----------------------------------------------------
        # Velocity data at edges used for interpolation
        u_interp = 1.5*U_pu.ucontra[:,:,:] - 0.5*U_pu.ucontra_old[:,:,:] # extrapolation for time at n+1/2
        # Linear interpolation
        a = U_pu.ucontra[i0:iend+1,:,:]*dto2/cs_grid.dx
        upos = U_pu.ucontra[i0:iend+1,:,:]>=0
        uneg = ~upos
        U_pu.ucontra_averaged[i0:iend+1,:,:][upos] = (1.0-a[upos])*u_interp[i0:iend+1,:,:][upos] + a[upos]*u_interp[i0-1:iend,:,:][upos]
        U_pu.ucontra_averaged[i0:iend+1,:,:][uneg] = -a[uneg]*u_interp[i0+1:iend+2,:,:][uneg] + (1.0+a[uneg])*u_interp[i0:iend+1,:,:][uneg]

        #----------------------------------------------------
        # Velocity data at edges used for interpolation
        v_interp = 1.5*U_pv.vcontra[:,:,:] - 0.5*U_pv.vcontra_old[:,:,:] # extrapolation for time at n+1/2
        # Linear interpolation
        a = U_pv.vcontra[:,j0:jend+1,:]*dto2/cs_grid.dy
        vpos = U_pv.vcontra[:,j0:jend+1,:]>=0
        vneg = ~vpos
        U_pv.vcontra_averaged[:,j0:jend+1,:][vpos] = (1.0-a[vpos])*v_interp[:,j0:jend+1,:][vpos] + a[vpos]*v_interp[:,j0-1:jend,:][vpos]
        U_pv.vcontra_averaged[:,j0:jend+1,:][vneg] = -a[vneg]*v_interp[:,j0+1:jend+2,:][vneg] + (1.0+a[vneg])*v_interp[:,j0:jend+1,:][vneg]
