import numpy   as np
import numexpr as ne
from constants         import nbfaces
from reconstruction_1d import ppm_reconstruction

####################################################################################
# Compute the 1d flux operators
####################################################################################
def compute_fluxes(Qx, Qy, px, py, cx, cy, cs_grid, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    ppm_reconstruction(Qx, Qy, px, py, cs_grid, simulation)

    # Compute the fluxes
    numerical_flux_ppm_x(px, cx, cs_grid, simulation)
    numerical_flux_ppm_y(py, cy, cs_grid, simulation)

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(px, cx, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.ng
    i0, iend = cs_grid.i0, cs_grid.iend
    j0, jend = cs_grid.j0, cs_grid.jend

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
    ng = cs_grid.ng
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

    """
    i0 = cs_grid.i0
    j0 = cs_grid.j0
    iend = cs_grid.iend
    jend = cs_grid.jend

    # 0-1; 1-2; 2-3
    e1_L = np.amax(abs(px.f_L[iend,j0:jend,0:3]-px.f_L[i0,j0:jend,1:4]))
    e1_R = np.amax(abs(px.f_R[iend,j0:jend,0:3]-px.f_R[i0,j0:jend,1:4]))
    e1   = np.amax(abs(cx[iend,j0:jend,0:3]-cx[i0,j0:jend,1:4]))

    # 3-0
    e2_L = np.amax(abs(px.f_L[iend,j0:jend,3]-px.f_L[i0,j0:jend,0]))
    e2_R = np.amax(abs(px.f_R[iend,j0:jend,3]-px.f_R[i0,j0:jend,0]))
    e2   = np.amax(abs(cx[iend,j0:jend,3]-cx[i0,j0:jend,0]))

    # 0-4
    e3_L = np.amax(abs(py.f_L[i0:iend,jend,0]-py.f_L[i0:iend,j0,4]))
    e3_R = np.amax(abs(py.f_L[i0:iend,jend,0]-py.f_L[i0:iend,j0,4]))
    e3   = np.amax(abs(cy[i0:iend,jend,0]-cy[i0:iend,j0,4]))

    # 1-4
    e4_L = np.amax(abs(py.f_L[i0:iend,jend,1]-px.f_R[iend,j0:jend,4]))
    e4_R = np.amax(abs(py.f_R[i0:iend,jend,1]-px.f_L[iend,j0:jend,4]))
    e4   = np.amax(abs(cy[i0:iend,jend,1]+cx[iend,j0:jend,4]))

    # 2-4
    e5_L = np.amax(abs(py.f_L[i0:iend,jend,2]-np.flip(py.f_R[i0:iend,jend,4])))
    e5_R = np.amax(abs(py.f_R[i0:iend,jend,2]-np.flip(py.f_L[i0:iend,jend,4])))
    e5   = np.amax(abs(cy[i0:iend,jend,2]+np.flip(cy[i0:iend,jend,4])))

    # 3-4
    e6_L = np.amax(abs(py.f_L[i0:iend,jend,3]-np.flip(px.f_L[i0,j0:jend,4])))
    e6_R = np.amax(abs(py.f_R[i0:iend,jend,3]-np.flip(px.f_R[i0,j0:jend,4])))
    e6   = np.amax(abs(cy[i0:iend,jend,3]-np.flip(cx[i0,j0:jend,4])))

    # 0-5
    e7_L = np.amax(abs(py.f_L[i0:iend,j0,0]-py.f_L[i0:iend,jend,5]))
    e7_R = np.amax(abs(py.f_R[i0:iend,j0,0]-py.f_R[i0:iend,jend,5]))
    e7   = np.amax(abs(cy[i0:iend,j0,0]-cy[i0:iend,jend,5]))

    # 1-5
    e8_L = np.amax(abs(px.f_L[iend,j0:jend,5]-np.flip(py.f_L[i0:iend,j0,1])))
    e8_R = np.amax(abs(px.f_R[iend,j0:jend,5]-np.flip(py.f_R[i0:iend,j0,1])))
    e8   = np.amax(abs(cx[iend,j0:jend,5]-np.flip(cy[i0:iend,j0,1])))

    # 2-5
    e9_L = np.amax(abs(py.f_L[i0:iend,j0,5]-np.flip(py.f_R[i0:iend,j0,2])))
    e9_R = np.amax(abs(py.f_R[i0:iend,j0,5]-np.flip(py.f_L[i0:iend,j0,2])))
    e9 = np.amax(abs(cy[i0:iend,j0,5]+np.flip(cy[i0:iend,j0,2])))

    # 3-5
    e10_L = np.amax(abs(py.f_L[i0:iend,j0,3]-px.f_R[i0,j0:jend,5]))
    e10_R = np.amax(abs(py.f_R[i0:iend,j0,3]-px.f_L[i0,j0:jend,5]))
    e10   = np.amax(abs(cy[i0:iend,j0,3]+cx[i0,j0:jend,5]))

    print('---------------------')
    print(e1_L, e1_R, e1)
    print(e2_L, e2_R, e2)
    print(e3_L, e3_R, e3)
    print(e4_L, e4_R, e4)
    print(e5_L, e5_R, e5)
    print(e6_L, e6_R, e6)
    print(e7_L, e7_R, e7)
    print(e8_L, e8_R, e8)
    print(e9_L, e9_R, e9)
    print(e10_L, e10_R, e10)
    print('---------------------')

    e = max(e1_L, e1_R, e1)
    e = max(e, e2_L, e2_R, e2)
    e = max(e, e3_L, e3_R, e3)
    e = max(e, e4_L, e4_R, e4)
    e = max(e, e5_L, e5_R, e5)
    e = max(e, e6_L, e6_R, e6)
    e = max(e, e7_L, e7_R, e7)
    e = max(e, e8_L, e8_R, e8)
    e = max(e, e9_L, e9_R, e9)
    e = max(e, e10_L, e10_R, e10)
    print('error flux = ',e)
    """
