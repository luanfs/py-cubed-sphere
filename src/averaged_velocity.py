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

    u = simulation.U_pu.ucontra[i0:iend+1,:,:]
    simulation.U_pu.upos = ne.evaluate('u>=0')
    simulation.U_pu.uneg = ~simulation.U_pu.upos

    v = simulation.U_pv.vcontra[:,j0:jend+1,:]
    simulation.U_pv.vpos = ne.evaluate('v>=0')
    simulation.U_pv.vneg = ~simulation.U_pv.vpos

    # Compute the velocity needed for the departure point
    if simulation.dp_name == 'RK1':
        simulation.U_pu.ucontra_averaged[:,:,:] = simulation.U_pu.ucontra[:,:,:]
        simulation.U_pv.vcontra_averaged[:,:,:] = simulation.U_pv.vcontra[:,:,:]

    elif simulation.dp_name == 'RK2':
        dt = simulation.dt
        dto2 = simulation.dto2

        #----------------------------------------------------
        # x direction
        # Velocity data at edges used for interpolation
        upos, uneg = simulation.U_pu.upos, simulation.U_pu.uneg
        u_interp = ne.evaluate('1.5*ucontra - 0.5*ucontra_old', local_dict=vars(simulation.U_pu)) # extrapolation for time at n+1/2
        # Linear interpolation
        a = simulation.U_pu.ucontra[i0:iend+1,:,:]*dto2/cs_grid.dx
        ap, an = a[upos], a[uneg]
        u1, u2 = u_interp[i0-1:iend,:,:][upos], u_interp[i0:iend+1,:,:][upos] 
        u3, u4 = u_interp[i0:iend+1:,:,:][uneg], u_interp[i0+1:iend+2,:,:][uneg]
        simulation.U_pu.ucontra_averaged[i0:iend+1,:,:][upos] = ne.evaluate('(1.0-ap)*u2 + ap*u1')
        simulation.U_pu.ucontra_averaged[i0:iend+1,:,:][uneg] = ne.evaluate('-an*u4 + (1.0+an)*u3')

        #----------------------------------------------------
        # y direction
        # Velocity data at edges used for interpolation
        vpos, vneg = simulation.U_pv.vpos, simulation.U_pv.vneg
        v_interp = ne.evaluate('1.5*vcontra - 0.5*vcontra_old', local_dict=vars(simulation.U_pv)) # extrapolation for time at n+1/2
        # Linear interpolation
        a = simulation.U_pv.vcontra[:,j0:jend+1,:]*dto2/cs_grid.dy
        ap, an = a[vpos], a[vneg]
        v1, v2 = v_interp[:,j0-1:jend,:][vpos], v_interp[:,j0:jend+1,:][vpos]
        v3, v4 = v_interp[:,j0:jend+1,:][vneg], v_interp[:,j0+1:jend+2,:][vneg]
        simulation.U_pv.vcontra_averaged[:,j0:jend+1,:][vpos] = ne.evaluate('(1.0-ap)*v2 + ap*v1')
        simulation.U_pv.vcontra_averaged[:,j0:jend+1,:][vneg] = ne.evaluate('-an*v4 + (1.0+an)*v3')

