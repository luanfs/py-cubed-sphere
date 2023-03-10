####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

from advection_ic import velocity_adv
from sphgeo       import latlon_to_contravariant
import numexpr as ne

def time_averaged_velocity(U_pu, U_pv, k, t, cs_grid, simulation):
    # Compute the velocity needed for the departure point
    if simulation.dp == 1:
        U_pu.ucontra_averaged[:,:,:] = U_pu.ucontra[:,:,:]
        U_pv.vcontra_averaged[:,:,:] = U_pv.vcontra[:,:,:]
    elif simulation.dp == 2:
        dt = simulation.dt
        dto2 = simulation.dto2
        twodt = simulation.twodt

        #----------------------------------------------------
        Xu = cs_grid.edx.lat
        Yu = cs_grid.edx.lon
        K1u, K1v = velocity_adv(Xu          , Yu, t, simulation)
        K2u, K2v = velocity_adv(Xu-dto2*K1u , Yu, t-dto2, simulation)
        K3u, K3v = velocity_adv(Xu-twodt*K2u, Yu, t-dt, simulation)
        U_av = ne.evaluate('(K1u + 4.0*K2u + K3u)/6.0')
        V_av = ne.evaluate('(K1v + 4.0*K2v + K3v)/6.0')
        U_pu.ucontra_averaged[:,:,:], _ = latlon_to_contravariant(U_av, V_av, cs_grid.prod_ex_elon_edx, cs_grid.prod_ex_elat_edx,\
                                                       cs_grid.prod_ey_elon_edx, cs_grid.prod_ey_elat_edx, cs_grid.determinant_ll2contra_edx)

       #----------------------------------------------------
        Xu = cs_grid.edy.lat
        Yu = cs_grid.edy.lon
        K1u, K1v = velocity_adv(Xu, Yu          , t, simulation)
        K2u, K2v = velocity_adv(Xu, Yu-dto2*K1v , t-dto2, simulation)
        K3u, K3v = velocity_adv(Xu, Yu-twodt*K2v, t-dt, simulation)
        U_av = ne.evaluate('(K1u + 4.0*K2u + K3u)/6.0')
        V_av = ne.evaluate('(K1v + 4.0*K2v + K3v)/6.0')
        _, U_pv.vcontra_averaged[:,:,:] =  latlon_to_contravariant(U_av, V_av, cs_grid.prod_ex_elon_edy, cs_grid.prod_ex_elat_edy,\
                                                       cs_grid.prod_ey_elon_edy, cs_grid.prod_ey_elat_edy, cs_grid.determinant_ll2contra_edy)
