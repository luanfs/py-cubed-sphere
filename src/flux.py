import numpy as np
import reconstruction_1d as rec
from constants import nbfaces
import numexpr as ne

####################################################################################
# Compute the 1d flux operators
####################################################################################
def compute_fluxes(Qx, Qy, px, py, cx, cy, cs_grid, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_x(Qx, px, cs_grid, simulation)
    rec.ppm_reconstruction_y(Qy, py, cs_grid, simulation)

    # Compute the fluxes
    numerical_flux_ppm_x(px, cx, cs_grid, simulation)
    numerical_flux_ppm_y(py, cy, cs_grid, simulation)


####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(px, cx, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.nghost
    i0 = cs_grid.i0
    iend = cs_grid.iend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    #px.f_L[i0:iend+1,:,:] = px.q_R[i0-1:iend,:,:] - cx[i0:iend+1,:,:]*0.5*(px.dq[i0-1:iend,:,:] - (1.0-(2.0/3.0)*cx[i0:iend+1,:,:])*px.q6[i0-1:iend,:,:])
    q_R = px.q_R[i0-1:iend,:,:]
    c   = cx[i0:iend+1,:,:]
    dq  = px.dq[i0-1:iend,:,:]
    q6  = px.q6[i0-1:iend,:,:]
    px.f_L[i0:iend+1,:,:] = ne.evaluate("q_R - c*0.5*(dq - (1.0-(2.0/3.0)*c)*q6)")

    # Flux at right edges
    #px.f_R[i0:iend+1,:,:] = px.q_L[i0:iend+1,:,:] - cx[i0:iend+1,:,:]*0.5*(px.dq[i0:iend+1,:,:] + (1.0+(2.0/3.0)*cx[i0:iend+1,:,:])*px.q6[i0:iend+1,:,:])
    q_L = px.q_L[i0:iend+1,:,:]
    c   = cx[i0:iend+1,:,:]
    dq  = px.dq[i0:iend+1,:]
    q6  = px.q6[i0:iend+1,:]
    px.f_R[i0:iend+1,:,:] = ne.evaluate("q_L - c*0.5*(dq + (1.0+(2.0/3.0)*c)*q6)")

    # F - Formula 1.13 from Collela and Woodward 1984)
    #px.f_upw[cx[:,:,:] >= 0] = px.f_L[cx[:,:,:] >= 0]
    #px.f_upw[cx[:,:,:] <= 0] = px.f_R[cx[:,:,:] <= 0]
    mask = ne.evaluate('cx >= 0')
    px.f_upw[mask]  = px.f_L[mask]
    px.f_upw[~mask] = px.f_R[~mask]

###############################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_ppm_y(py, cy, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.nghost
    j0 = cs_grid.j0
    jend = cs_grid.jend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    #py.f_L[:,j0:jend+1,:] = py.q_R[:,j0-1:jend,:] - cy[:,j0:jend+1,:]*0.5*(py.dq[:,j0-1:jend,:] - (1.0-2.0/3.0*cy[:,j0:jend+1,:])*py.q6[:,j0-1:jend,:])
    q_R = py.q_R[:,j0-1:jend,:]
    c   = cy[:,j0:jend+1,:]
    dq  = py.dq[:,j0-1:jend,:]
    q6  = py.q6[:,j0-1:jend,:]
    py.f_L[:,j0:jend+1,:] = ne.evaluate("q_R - c*0.5*(dq - (1.0-(2.0/3.0)*c)*q6)")

    # Flux at right edges
    #py.f_R[:,j0:jend+1,:] = py.q_L[:,j0:jend+1,:] - cy[:,j0:jend+1,:]*0.5*(py.dq[:,j0:jend+1,:] + (1.0+2.0/3.0*cy[:,j0:jend+1,:])*py.q6[:,j0:jend+1,:])
    q_L = py.q_L[:,j0:jend+1,:]
    c   = cy[:,j0:jend+1,:]
    dq  = py.dq[:,j0:jend+1,:]
    q6  = py.q6[:,j0:jend+1,:]
    py.f_R[:,j0:jend+1,:] = ne.evaluate("q_L - c*0.5*(dq + (1.0+(2.0/3.0)*c)*q6)")

    # G - Formula 1.13 from Collela and Woodward 1984)
    #py.f_upw[cy[:,:,:] >= 0] = py.f_L[cy[:,:,:] >= 0]
    #py.f_upw[cy[:,:,:] <= 0] = py.f_R[cy[:,:,:] <= 0]
    mask = ne.evaluate('cy >= 0')
    py.f_upw[mask]  = py.f_L[mask]
    py.f_upw[~mask] = py.f_R[~mask]
