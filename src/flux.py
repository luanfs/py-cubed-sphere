import numpy   as np
import numexpr as ne
from constants         import nbfaces
from reconstruction_1d import ppm_reconstruction

####################################################################################
# Compute the 1d flux operators
####################################################################################
def compute_fluxes(Qx, Qy, px, py, U_pu, U_pv, cx, cy, cs_grid, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    ppm_reconstruction(Qx, Qy, px, py, cs_grid, simulation)

    # Compute the fluxes
    numerical_flux_ppm_x(Qx, px, U_pu, cx, cs_grid, simulation)
    numerical_flux_ppm_y(Qy, py, U_pv, cy, cs_grid, simulation)

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(Qx, px, U_pu, cx, cs_grid, simulation):
    M = cs_grid.N
    ng = cs_grid.ng
    dx = cs_grid.dx
    dt = simulation.dt
    i0, iend = cs_grid.i0, cs_grid.iend

    # multiply values at edges by metric tensors
    if simulation.et_name == 'ET-Z21' or simulation.et_name == 'ET-Z21-AF'\
        or simulation.et_name=='ET-Z21-PR':
        px.q_L[i0-1:iend+1,:,:] = px.q_L[i0-1:iend+1,:,:]*cs_grid.metric_tensor_pu[i0-1:iend+1,:,:]
        px.q_R[i0-1:iend+1,:,:] = px.q_R[i0-1:iend+1,:,:]*cs_grid.metric_tensor_pu[i0:iend+2,:,:]
        q = Qx[i0-1:iend+1,:,:]*cs_grid.metric_tensor_pc[i0-1:iend+1,:,:]

    else: 
        q = Qx[i0-1:iend+1,:,:]

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    q_L =  px.q_L[i0-1:iend+1,:,:]
    q_R =  px.q_R[i0-1:iend+1,:,:]
    px.dq[i0-1:iend+1,:,:] = ne.evaluate('q_R-q_L')
    px.q6[i0-1:iend+1,:,:] = ne.evaluate('6*q- 3*(q_R + q_L)')

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    upos = U_pu.upos
    c   = cx[i0:iend+1,:,:][upos]
    q_R = px.q_R[i0-1:iend,:,:][upos]
    dq  = px.dq[i0-1:iend,:,:][upos]
    q6  = px.q6[i0-1:iend,:,:][upos]
    px.f_L[i0:iend+1,:,:][upos] = ne.evaluate("q_R + c*0.5*(q6-dq) - q6*c*c/3.0")

    # Flux at right edges
    uneg = U_pu.uneg
    c   = cx[i0:iend+1,:,:][uneg]
    q_L = px.q_L[i0:iend+1,:,:][uneg]
    dq  = px.dq[i0:iend+1,:,:][uneg]
    q6  = px.q6[i0:iend+1,:,:][uneg]
    px.f_R[i0:iend+1,:,:][uneg] = ne.evaluate("q_L - c*0.5*(q6+dq) - q6*c*c/3.0")
 
    # F - Formula 1.13 from Collela and Woodward 1984)
    px.f_upw[i0:iend+1,:,:][upos] = px.f_L[i0:iend+1,:,:][upos]
    px.f_upw[i0:iend+1,:,:][uneg] = px.f_R[i0:iend+1,:,:][uneg]
    px.f_upw[i0:iend+1,:,:] = px.f_upw[i0:iend+1,:,:]*U_pu.ucontra_averaged[i0:iend+1,:,:]

    # multiply values at edges by metric tensors
    if simulation.et_name == 'ET-S72' or simulation.et_name == 'ET-PL07':
        px.f_upw[i0:iend+1,:,:] = px.f_upw[i0:iend+1,:,:]*cs_grid.metric_tensor_pu[i0:iend+1,:,:]

###############################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_ppm_y(Qy, py, U_pv, cy, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.ng
    j0 = cs_grid.j0
    jend = cs_grid.jend
    dy = cs_grid.dy
    dt = simulation.dt

    # multiply values at edges by metric tensors
    if simulation.et_name == 'ET-Z21' or simulation.et_name == 'ET-Z21-AF'\
        or simulation.et_name=='ET-Z21-PR':
        py.q_L[:,j0-1:jend+1,:] = py.q_L[:,j0-1:jend+1,:]*cs_grid.metric_tensor_pv[:,j0-1:jend+1,:]
        py.q_R[:,j0-1:jend+1,:] = py.q_R[:,j0-1:jend+1,:]*cs_grid.metric_tensor_pv[:,j0:jend+2,:]
        q = Qy[:,j0-1:jend+1,:]*cs_grid.metric_tensor_pc[:,j0-1:jend+1,:]

    else:
        q = Qy[:,j0-1:jend+1,:]

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    q_L =  py.q_L[:,j0-1:jend+1,:]
    q_R =  py.q_R[:,j0-1:jend+1,:]
    py.dq[:,j0-1:jend+1,:] = ne.evaluate('q_R-q_L')
    py.q6[:,j0-1:jend+1,:] = ne.evaluate('6*q- 3*(q_R + q_L)')

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    vpos = U_pv.vpos
    c   = cy[:,j0:jend+1,:][vpos]
    q_R = py.q_R[:,j0-1:jend,:][vpos]
    dq  = py.dq[:,j0-1:jend,:][vpos]
    q6  = py.q6[:,j0-1:jend,:][vpos]
    py.f_L[:,j0:jend+1,:][vpos] = ne.evaluate("q_R + c*0.5*(q6-dq) - q6*c*c/3.0")

    # Flux at right edges
    vneg = U_pv.vneg
    c   = cy[:,j0:jend+1,:][vneg]
    q_L = py.q_L[:,j0:jend+1,:][vneg]
    dq  = py.dq[:,j0:jend+1,:][vneg]
    q6  = py.q6[:,j0:jend+1,:][vneg]
    py.f_R[:,j0:jend+1,:][vneg] = ne.evaluate("q_L - c*0.5*(q6+dq) - q6*c*c/3.0")

    # G - Formula 1.13 from Collela and Woodward 1984)
    py.f_upw[:,j0:jend+1,:][vpos] = py.f_L[:,j0:jend+1,:][vpos]
    py.f_upw[:,j0:jend+1,:][vneg] = py.f_R[:,j0:jend+1,:][vneg]
    py.f_upw[:,j0:jend+1,:] = py.f_upw[:,j0:jend+1,:]*U_pv.vcontra_averaged[:,j0:jend+1,:]

    # multiply values at edges by metric tensors
    if simulation.et_name == 'ET-S72' or simulation.et_name == 'ET-PL07':
        py.f_upw[:,j0:jend+1,:] = py.f_upw[:,j0:jend+1,:]*cs_grid.metric_tensor_pv[:,j0:jend+1,:]

