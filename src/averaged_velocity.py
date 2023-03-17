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

def time_averaged_velocity(U_pu, U_pv, k, t, cs_grid, simulation):
    i0, iend = cs_grid.i0, cs_grid.iend
    j0, jend = cs_grid.j0, cs_grid.jend
    N = cs_grid.N
    ng = cs_grid.nghost

    # Compute the velocity needed for the departure point
    if simulation.dp == 1:
        U_pu.ucontra_averaged[:,:,:] = U_pu.ucontra[:,:,:]
        U_pv.vcontra_averaged[:,:,:] = U_pv.vcontra[:,:,:]
    elif simulation.dp == 2:
        dt = simulation.dt
        dto2 = simulation.dto2

        #----------------------------------------------------
        # First departure point estimate
        Xu = cs_grid.Xu
        xd = Xu[:,:,:]-dto2*U_pu.ucontra[:,:,:]

        # Velocity data at edges used for interpolation
        u_interp = 1.5*U_pu.ucontra[:,:,:] - 0.5*U_pu.ucontra_old[:,:,:] # extrapolation for time at n+1/2

        # Linear interpolation
        #for j in range(0, N+ng):
        #    for p in range(0,6):
        #        U_pu.ucontra_averaged[i0:iend+1,j,p] = np.interp(xd[i0:iend+1,j,p], Xu[:,j,p], u_interp[:,j,p])

        a = (Xu[i0:iend+1,:,:]-xd[i0:iend+1,:,:])/cs_grid.dx
        upos = U_pu.ucontra[i0:iend+1,:,:]>=0
        uneg = ~upos
        U_pu.ucontra_averaged[i0:iend+1,:,:][upos] = (1.0-a[upos])*u_interp[i0:iend+1,:,:][upos] + a[upos]*u_interp[i0-1:iend,:,:][upos]
        U_pu.ucontra_averaged[i0:iend+1,:,:][uneg] = -a[uneg]*u_interp[i0+1:iend+2,:,:][uneg] + (1.0+a[uneg])*u_interp[i0:iend+1,:,:][uneg]


        #----------------------------------------------------
        # First departure point estimate
        Yv = cs_grid.Yv
        yd = Yv[:,:,:]-dto2*U_pv.vcontra[:,:,:]

        # Velocity data at edges used for interpolation
        v_interp = 1.5*U_pv.vcontra[:,:,:] - 0.5*U_pv.vcontra_old[:,:,:] # extrapolation for time at n+1/2

        # Linear interpolation
        #for i in range(0, N+ng):
        #    for p in range(0,6):
        #        U_pv.vcontra_averaged[i,j0:jend+1,p] = np.interp(yd[i,j0:jend+1,p], Yv[i,:,p], v_interp[i,:,p])
        a = (Yv[:,j0:jend+1,:]-yd[:,j0:jend+1,:])/cs_grid.dy
        vpos = U_pv.vcontra[:,j0:jend+1,:]>=0
        vneg = ~vpos
        U_pv.vcontra_averaged[:,j0:jend+1,:][vpos] = (1.0-a[vpos])*v_interp[:,j0:jend+1,:][vpos] + a[vpos]*v_interp[:,j0-1:jend,:][vpos]
        U_pv.vcontra_averaged[:,j0:jend+1,:][vneg] = -a[vneg]*v_interp[:,j0+1:jend+2,:][vneg] + (1.0+a[vneg])*v_interp[:,j0:jend+1,:][vneg]


    #----------------------------------------------------
