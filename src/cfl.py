####################################################################################
#
# Module for computing the CFL number
####################################################################################
import numpy as np

####################################################################################
# CFL number in x direction
####################################################################################
def cfl_x(u_edges, cs_grid, simulation):
    cx = np.sign(u_edges)*u_edges*simulation.dt/cs_grid.dx
    cx2 = cx*cx
    return cx, cx2

####################################################################################
# CFL number in y direction
####################################################################################
def cfl_y(v_edges, cs_grid, simulation):
    cy = np.sign(v_edges)*v_edges*simulation.dt/cs_grid.dy
    cy2 = cy*cy
    return cy, cy2
