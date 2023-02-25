import numpy as np
import reconstruction_1d as rec
from constants import nbfaces
import numexpr as ne

####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def compute_flux_x(Q, px, cx, cs_grid, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_x(Q, px, cs_grid, simulation)
    # Compute the fluxes
    numerical_flux_ppm_x(px, cx, cs_grid, simulation)

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


####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  v_edges (velocity at edges)
####################################################################################
def compute_flux_y(Q, py, cy, cs_grid, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_y(Q, py, cs_grid, simulation)

    # Compute the fluxes
    numerical_flux_ppm_y(py, cy, cs_grid, simulation)

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

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def compute_fluxes(Qx, Qy, px, py, cx, cy, cs_grid, simulation):
    # Compute the fluxes in x direction
    compute_flux_x(Qx, px, cx, cs_grid, simulation)

    # Compute the fluxes in y direction
    compute_flux_y(Qy, py, cy, cs_grid, simulation)

####################################################################################
####################################################################################
def average_fluxes_at_cube_edges(flux_x, flux_y, cs_grid):
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # Panel 0 and 1
    flux_x[iend,j0:jend,0] = (flux_x[iend,j0:jend,0] + flux_x[i0,j0:jend,1] )*0.5
    flux_x[i0,j0:jend,1] = flux_x[iend,j0:jend,0]

    # Panel 1 and 2
    flux_x[iend,j0:jend,1] = (flux_x[iend,j0:jend,1] + flux_x[i0,j0:jend,2] )*0.5
    flux_x[i0,j0:jend,2] = flux_x[iend,j0:jend,1]

    # Panel 2 and 3
    flux_x[iend,j0:jend,2] = (flux_x[iend,j0:jend,2] + flux_x[i0,j0:jend,3] )*0.5
    flux_x[i0,j0:jend,3] = flux_x[iend,j0:jend,2]

    # Panel 3 and 0
    flux_x[iend,j0:jend,3] = (flux_x[iend,j0:jend,3] + flux_x[i0,j0:jend,0] )*0.5
    flux_x[i0,j0:jend,0] = flux_x[iend,j0:jend,3]

    # Panel 4 and 0
    flux_y[i0:iend,j0,4] = (flux_y[i0:iend,j0,4] + flux_y[i0:iend,jend,0])*0.5
    flux_y[i0:iend,jend,0] =  flux_y[i0:iend,j0,4]

    # Panel 4 and 1
    flux_x[iend,j0:jend,4] = (flux_x[iend,j0:jend,4] + flux_y[i0:iend,jend,1])*0.5
    flux_y[i0:iend,jend,1] = flux_x[iend,j0:jend,4]

    # Panel 4 and 2
    flux_y[i0:iend,jend,4] = (flux_y[i0:iend,jend,4] + np.flip(flux_y[i0:iend,jend,2]))*0.5
    flux_y[i0:iend,jend,2] =  np.flip(flux_y[i0:iend,jend,4])

    # Panel 4 and 3
    flux_x[i0,j0:jend,4] = (flux_x[i0,j0:jend,4] + np.flip(flux_y[i0:iend,jend,3]))*0.5
    flux_y[i0:iend,jend,3] = np.flip(flux_x[i0,j0:jend,4])

    # Panel 5 and 0
    flux_y[i0:iend,jend,5] = (flux_y[i0:iend,jend,5] + flux_y[i0:iend,j0,0])*0.5
    flux_y[i0:iend,j0,0] = flux_y[i0:iend,jend,5]

    # Panel 5 and 1
    flux_x[iend,j0:jend,5] = (flux_x[iend,j0:jend,5] + np.flip(flux_y[i0:iend,j0,1]))*0.5
    flux_y[i0:iend,j0,1] = np.flip(flux_x[iend,j0:jend,5])

    # Panel 5 and 2
    flux_y[i0:iend,j0,5] = (flux_y[i0:iend,j0,5] + np.flip(flux_y[i0:iend,j0,2]))*0.5
    flux_y[i0:iend,j0,2] = np.flip(flux_y[i0:iend,j0,5])

    # Panel 5 and 3
    flux_x[i0,j0:jend,5] = (flux_x[i0,j0:jend,5] + flux_y[i0:iend,j0,3])*0.5
    flux_y[i0:iend,j0,3] = flux_x[i0,j0:jend,5]

####################################################################################
####################################################################################
def fix_fluxes_at_cube_edges(flux_x, flux_y, cs_grid):
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # Panel 0 and 1
    flux_x[iend,j0:jend,0] = flux_x[i0,j0:jend,1]

    # Panel 1 and 2
    flux_x[iend,j0:jend,1] = flux_x[i0,j0:jend,2]

    # Panel 2 and 3
    flux_x[iend,j0:jend,2] = flux_x[i0,j0:jend,3]

    # Panel 3 and 0
    flux_x[iend,j0:jend,3] = flux_x[i0,j0:jend,0]

    # Panel 4 and 0
    flux_y[i0:iend,j0,4] = flux_y[i0:iend,jend,0]

    # Panel 4 and 1
    flux_x[iend,j0:jend,4] = flux_y[i0:iend,jend,1]

    # Panel 4 and 2
    flux_y[i0:iend,jend,4] = np.flip(flux_y[i0:iend,jend,2])

    # Panel 4 and 3
    flux_x[i0,j0:jend,4] = np.flip(flux_y[i0:iend,jend,3])

    # Panel 5 and 0
    flux_y[i0:iend,jend,5] = flux_y[i0:iend,j0,0]

    # Panel 5 and 1
    flux_x[iend,j0:jend,5] = np.flip(flux_y[i0:iend,j0,1])

    # Panel 5 and 2
    flux_y[i0:iend,j0,5] = np.flip(flux_y[i0:iend,j0,2])

    # Panel 5 and 3
    flux_x[i0,j0:jend,5] = flux_y[i0:iend,j0,3]

"""
####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  u_edges (zonal velocity at edges)
####################################################################################
def flux_ppm_x_stencil(Q, u_edges, flux_x, ax, cs_grid, simulation):
    i0   = cs_grid.i0
    iend = cs_grid.iend

    flux_x[i0:iend+1,:,:] = ax[0,i0:iend+1,:,:]*Q[i0-3:iend-2,:,:] +\
                            ax[1,i0:iend+1,:,:]*Q[i0-2:iend-1,:,:] +\
                            ax[2,i0:iend+1,:,:]*Q[i0-1:iend+0,:,:] +\
                            ax[3,i0:iend+1,:,:]*Q[i0+0:iend+1,:,:] +\
                            ax[4,i0:iend+1,:,:]*Q[i0+1:iend+2,:,:] +\
                            ax[5,i0:iend+1,:,:]*Q[i0+2:iend+3,:,:]

####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  v_edges (zonal velocity at edges)
####################################################################################
def flux_ppm_y_stencil(Q, v_edges, flux_y, ay, cs_grid, simulation):
    j0   = cs_grid.j0
    jend = cs_grid.jend

    flux_y[:,j0:jend+1,:] = ay[0,:,j0:jend+1,:]*Q[:,j0-3:jend-2,:] +\
                            ay[1,:,j0:jend+1,:]*Q[:,j0-2:jend-1,:] +\
                            ay[2,:,j0:jend+1,:]*Q[:,j0-1:jend+0,:] +\
                            ay[3,:,j0:jend+1,:]*Q[:,j0+0:jend+1,:] +\
                            ay[4,:,j0:jend+1,:]*Q[:,j0+1:jend+2,:] +\
                            ay[5,:,j0:jend+1,:]*Q[:,j0+2:jend+3,:]

"""
