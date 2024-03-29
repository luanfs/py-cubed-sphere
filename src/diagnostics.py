####################################################################################
#
# Module for diagnostic computation and output routines
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np

####################################################################################
# Total mass computation
####################################################################################
def mass_computation(Q, cs_grid, total_mass0):
    # Interior cells index (we are ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend
    dx, dy = cs_grid.dx, cs_grid.dy
    total_mass =  np.sum(Q[i0:iend,j0:jend,:]*cs_grid.metric_tensor_pc[i0:iend,j0:jend,:]*dx*dy)  # Compute new mass
    if abs(total_mass0)>10**(-10):
        mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
    else:
        mass_change = abs(total_mass0-total_mass)
    return total_mass, mass_change
