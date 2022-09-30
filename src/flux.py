import numpy as np
import reconstruction_1d as rec
from monotonization_1d import monotonization_1d_x, monotonization_1d_y
from constants import nbfaces

####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def compute_flux_x(flux_x, Q, u_edges, ax, cs_grid, simulation):
    N = cs_grid.N

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_x(Q, cs_grid, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_x(Q, q_L, q_R, dq, q6, cs_grid, simulation)

    # Compute the fluxes
    numerical_flux_x(Q, q_R, q_L, dq, q6, u_edges, flux_x, ax, cs_grid, simulation)

####################################################################################
# Routine to compute the correct flux in x direction
####################################################################################
def numerical_flux_x(Q, q_R, q_L, dq, q6, u_edges, flux_x, ax, cs_grid, simulation):
    if simulation.mono == 1: # Monotonization
        numerical_flux_ppm_x(q_R, q_L, dq, q6, u_edges, flux_x, cs_grid, simulation)

    elif simulation.mono == 0: # No monotonization 
        if simulation.fvmethod == 'PPM':
            flux_ppm_x_stencil(Q, u_edges, flux_x, ax, cs_grid, simulation)

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(q_R, q_L, dq, q6, u_edges, flux_x, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N

    # Numerical fluxes at edges
    f_L = np.zeros((N+7, M+6, nbfaces)) # Left
    f_R = np.zeros((N+7, M+6, nbfaces)) # Right

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = u_edges*(simulation.dt/cs_grid.dx)

    # Flux at left edges
    f_L[3:N+4,:,:] = q_R[2:N+3,:,:] - c[3:N+4,:,:]*0.5*(dq[2:N+3,:,:] - (1.0-(2.0/3.0)*c[3:N+4,:,:])*q6[2:N+3,:,:])

    # Flux at right edges
    c = -c
    f_R[3:N+4,:,:] = q_L[3:N+4,:,:] + c[3:N+4,:,:]*0.5*(dq[3:N+4,:,:] + (1.0-(2.0/3.0)*c[3:N+4,:,:])*q6[3:N+4,:,:])

    # F - Formula 1.13 from Collela and Woodward 1984)
    flux_x[u_edges[:,:,:] >= 0] = f_L[u_edges[:,:,:] >= 0]
    flux_x[u_edges[:,:,:] <= 0] = f_R[u_edges[:,:,:] <= 0]


####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  v_edges (velocity at edges)
####################################################################################
def compute_flux_y(flux_y, Q, v_edges, ay, cs_grid, simulation):
    M = cs_grid.N

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_y(Q, cs_grid, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_y(Q, q_L, q_R, dq, q6, cs_grid, simulation)

    # Compute the fluxes
    numerical_flux_y(Q, q_R, q_L, dq, q6, v_edges, flux_y, ay, cs_grid, simulation)

####################################################################################
# Routine to compute the correct flux in x direction
####################################################################################
def numerical_flux_y(Q, q_R, q_L, dq, q6, v_edges, flux_y, ay, cs_grid, simulation):
    if simulation.mono == 1: # Monotonization
        numerical_flux_ppm_y(q_R, q_L, dq, q6, v_edges, flux_y, cs_grid, simulation)

    elif simulation.mono == 0: # No monotonization 
        if simulation.fvmethod == 'PPM':
            flux_ppm_y_stencil(Q, v_edges, flux_y, ay, cs_grid, simulation)

###############################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_ppm_y(q_R, q_L, dq, q6, v_edges, flux_y, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N

    # Numerical fluxes at edges
    g_L = np.zeros((N+6, M+7, nbfaces))# Left
    g_R = np.zeros((N+6, M+7, nbfaces))# Rigth

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = v_edges*(simulation.dt/cs_grid.dy)

    # Flux at left edges
    g_L[:,3:M+4,:]= q_R[:,2:M+3,:] - c[:,3:M+4,:]*0.5*(dq[:,2:M+3,:] - (1.0-2.0/3.0*c[:,3:M+4,:])*q6[:,2:M+3,:])

    # Flux at right edges
    c = -c
    g_R[:,3:M+4,:] = q_L[:,3:M+4,:] + c[:,3:M+4,:]*0.5*(dq[:,3:M+4,:] + (1.0-2.0/3.0*c[:,3:M+4,:])*q6[:,3:M+4,:])

    # G - Formula 1.13 from Collela and Woodward 1984)
    flux_y[v_edges[:,:,:] >= 0] = g_L[v_edges[:,:,:] >= 0]
    flux_y[v_edges[:,:,:] <= 0] = g_R[v_edges[:,:,:] <= 0]

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

    flux_x[i0:iend+1,:,:] = flux_x[i0:iend+1,:,:]/12.0

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

    flux_y[:,j0:jend+1,:] = flux_y[:,j0:jend+1,:]/12.0
