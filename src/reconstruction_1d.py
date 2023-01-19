####################################################################################
#
# Piecewise Parabolic Method (PPM) polynomial reconstruction module
# Luan da Fonseca Santos - March 2022
#
# References:
# -  Phillip Colella, Paul R Woodward, The Piecewise Parabolic Method (PPM) for gas-dynamical simulations,
# Journal of Computational Physics, Volume 54, Issue 1, 1984, Pages 174-201, ISSN 0021-9991,
# https://doi.org/10.1016/0021-9991(84)90143-8.
#
# -  Carpenter , R. L., Jr., Droegemeier, K. K., Woodward, P. R., & Hane, C. E. (1990).
# Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling, Monthly Weather Review, 118(3),
# 586-612. Retrieved Mar 31, 2022,
# from https://journals.ametsoc.org/view/journals/mwre/118/3/1520-0493_1990_118_0586_aotppm_2_0_co_2.xml
#
####################################################################################

import numpy as np
from constants import nbfaces

####################################################################################
# Given the average values of a scalar field Q, this routine constructs
# a piecewise parabolic aproximation of Q using its average value.
####################################################################################
def ppm_reconstruction_x(Q, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.nghost
    i0 = cs_grid.i0
    iend = cs_grid.iend

    # Aux vars
    q_L = np.zeros((N+ng+1, M+ng, nbfaces))
    q_R = np.zeros((N+ng+1, M+ng, nbfaces))
    dq  = np.zeros((N+ng, M+ng, nbfaces))
    q6  = np.zeros((N+ng, M+ng, nbfaces))

    if simulation.recon_name == 'PPM': # PPM from CW84 paper
        # Values of Q at right edges (q_(j+1/2)) - Formula 1.9 from Collela and Woodward 1984
        Q_edges = np.zeros((N+ng+1,M+ng, nbfaces))
        Q_edges[i0-1:iend+2,:,:] = (7.0/12.0)*(Q[i0-1:iend+2,:,:] + Q[i0-2:iend+1,:,:]) - (Q[i0:iend+3,:,:] + Q[i0-3:iend,:,:])/12.0

        # Assign values of Q_R and Q_L
        q_R[i0-1:iend+1,:,:] = Q_edges[i0:iend+2,:,:]
        q_L[i0-1:iend+1,:,:] = Q_edges[i0-1:iend+1,:,:]

    elif simulation.recon_name == 'PPM_hybrid': # Hybrid PPM from PL07
        # coeffs from equations 41 and 42 from PL07
        a1 =   2.0/60.0
        a2 = -13.0/60.0
        a3 =  47.0/60.0
        a4 =  27.0/60.0
        a5 =  -3.0/60.0

        # Assign values of Q_R and Q_L
        q_R[i0-1:iend+1,:,:] = a1*Q[i0-3:iend-1,:,:] + a2*Q[i0-2:iend,:,:] + a3*Q[i0-1:iend+1,:,:] + a4*Q[i0:iend+2,:,:] + a5*Q[i0+1:iend+3,:,:]
        q_L[i0-1:iend+1,:,:] = a5*Q[i0-3:iend-1,:,:] + a4*Q[i0-2:iend,:,:] + a3*Q[i0-1:iend+1,:,:] + a2*Q[i0:iend+2,:,:] + a1*Q[i0+1:iend+3,:,:]


    elif simulation.recon_name == 'PPM_mono_CW84':  #PPM with monotonization from CW84
        # Compute the slopes dQ0 (centered finite difference)
        # Formula 1.7 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        dQ0 = np.zeros((N+ng, M+ng, nbfaces))
        dQ0[i0-2:iend+2,:,:] = 0.5*(Q[i0-1:iend+3,:,:] - Q[i0-3:iend+1,:,:]) # Interior values are in 3:N+3

         #Avoid overshoot
        # Compute the slopes dQ (1-sided finite difference)
        # Formula 1.8 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        dQ1 = np.zeros((N+ng, M+ng, nbfaces)) # Right 1-sided difference
        dQ2 = np.zeros((N+ng, M+ng, nbfaces)) # Left  1-sided difference

        # Right 1-sided finite difference
        dQ1[i0-2:iend+2,:,:] = Q[i0-1:iend+3,:,:] - Q[i0-2:iend+2,:,:]
        dQ1 = dQ1*2.0

        # Left  1-sided finite difference
        dQ2[i0-2:iend+2,:,:] = Q[i0-2:iend+2,:,:] - Q[i0-3:iend+1,:,:]
        dQ2 = dQ2*2.0

        # Final slope - Formula 1.8 from Collela and Woodward 1984
        dQ = np.zeros((N+ng, M+ng, nbfaces))
        dQ[i0-2:iend+2,:,:] = np.minimum(abs(dQ0[i0-2:iend+2,:,:]), abs(dQ1[i0-2:iend+2,:,:]))
        dQ[i0-2:iend+2,:,:] = np.minimum(dQ[i0-2:iend+2,:,:], abs(dQ2[i0-2:iend+2,:,:]))*np.sign(dQ0[i0-2:iend+2,:,:])
        mask = ( (Q[i0-1:iend+3,:,:] - Q[i0-2:iend+2,:,:]) * (Q[i0-2:iend+2,:,:] - Q[i0-3:iend+1,:,:]) > 0.0 ) # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        dQ[i0-2:iend+2,:,:][~mask] = 0.0

        # Values of Q at right edges (q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1984
        Q_edges = np.zeros((N+ng+1, M+ng, nbfaces))
        Q_edges[i0-1:iend+2,:,:] = 0.5*(Q[i0-1:iend+2,:,:] + Q[i0-2:iend+1,:,:]) - (dQ[i0-1:iend+2,:,:] - dQ[i0-2:iend+1,:,:])/6.0

        # Assign values of Q_R and Q_L
        q_R[i0-1:iend+1,:,:] = Q_edges[i0:iend+2,:,:]
        q_L[i0-1:iend+1,:,:] = Q_edges[i0-1:iend+1,:,:]

        # Compute the polynomial coefs
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        dq[i0-1:iend+1,:,:] = q_R[i0-1:iend+1,:,:] - q_L[i0-1:iend+1,:,:]
        q6[i0-1:iend+1,:,:] = 6*Q[i0-1:iend+1,:,:] - 3*(q_R[i0-1:iend+1,:,:] + q_L[i0-1:iend+1,:,:])

        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        local_maximum = (q_R[i0-1:iend+1,:,:]-Q[i0-1:iend+1,:,:])*(Q[i0-1:iend+1,:,:]-q_L[i0-1:iend+1,:,:])<=0

        # In this case (local maximum), the interpolation is a constant equal to Q
        q_R[i0-1:iend+1,:,:][local_maximum] = Q[i0-1:iend+1,:,:][local_maximum]
        q_L[i0-1:iend+1,:,:][local_maximum] = Q[i0-1:iend+1,:,:][local_maximum]

        # Check overshot
        overshoot  = (abs(dq[i0-1:iend+1,:,:]) < abs(q6[i0-1:iend+1,:,:]))

        # Move left
        move_left  = (q_R[i0-1:iend+1,:,:]-q_L[i0-1:iend+1,:,:])*(Q[i0-1:iend+1,:,:]-0.5*(q_R[i0-1:iend+1,:,:]+q_L[i0-1:iend+1,:,:])) > ((q_R[i0-1:iend+1,:,:]-q_L[i0-1:iend+1,:,:])**2)/6.0

        # Move right
        move_right = (-((q_R[i0-1:iend+1,:,:]-q_L[i0-1:iend+1,:,:])**2)/6.0 > (q_R[i0-1:iend+1,:,:]-q_L[i0-1:iend+1,:,:])*(Q[i0-1:iend+1,:,:]-0.5*(q_R[i0-1:iend+1,:,:]+q_L[i0-1:iend+1,:,:])) )

        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        q_L[i0-1:iend+1,:,:][overshoot_move_left]  = 3.0*Q[i0-1:iend+1,:,:][overshoot_move_left]  - 2.0*q_R[i0-1:iend+1,:,:][overshoot_move_left]
        q_R[i0-1:iend+1,:,:][overshoot_move_right] = 3.0*Q[i0-1:iend+1,:,:][overshoot_move_right] - 2.0*q_L[i0-1:iend+1,:,:][overshoot_move_right]
    elif simulation.recon_name == 'PPM_mono_L04':  #PPM with monotonization from Lin 04 paper
        # Formula B1 from Lin 04
        dQ      = np.zeros((N+ng,M+ng,nbfaces))
        dQ_min  = np.zeros((N+ng,M+ng,nbfaces))
        dQ_max  = np.zeros((N+ng,M+ng,nbfaces))
        dQ_mono = np.zeros((N+ng,M+ng,nbfaces))

        dQ[i0-3:iend+3,:,:] = 0.25*(Q[i0-2:iend+4,:,:] - Q[i0-4:iend+2,:,:])
        dQ_min[i0-3:iend+3,:,:]  = np.maximum(np.maximum(Q[i0-4:iend+2,:,:], Q[i0-3:iend+3,:,:]), Q[i0-2:iend+4,:,:]) - Q[i0-3:iend+3,:,:]
        dQ_max[i0-3:iend+3,:,:]  = Q[i0-3:iend+3,:,:] - np.minimum(np.minimum(Q[i0-4:iend+2,:,:], Q[i0-3:iend+3,:,:]), Q[i0-2:iend+4,:,:])
        dQ_mono[i0-3:iend+3,:,:] = np.minimum(np.minimum(abs(dQ[i0-3:iend+3,:,:]), dQ_min[i0-3:iend+3,:,:]), dQ_max[i0-3:iend+3,:,:]) * np.sign(dQ[i0-3:iend+3,:,:])

        # Formula B2 from Lin 04
        Q_edges = np.zeros((N+ng+1,M+ng,nbfaces))
        Q_edges[i0-1:iend+2,:,:] = 0.5*(Q[i0-1:iend+2,:,:] + Q[i0-2:iend+1,:,:]) - (dQ_mono[i0-1:iend+2,:,:] - dQ_mono[i0-2:iend+1,:,:])/3.0

        # Assign values of Q_R and Q_L
        q_R[i0-1:iend+1,:,:] = Q_edges[i0:iend+2,:,:]
        q_L[i0-1:iend+1,:,:] = Q_edges[i0-1:iend+1,:,:]

        # Formula B3 from Lin 04
        q_L[i0-1:iend+1,:,:] = Q[i0-1:iend+1,:,:] - np.minimum(2.0*abs(dQ_mono[i0-1:iend+1,:,:]), abs(q_L[i0-1:iend+1,:,:]-Q[i0-1:iend+1,:,:])) * np.sign(2.0*dQ_mono[i0-1:iend+1,:,:])

        # Formula B4 from Lin 04
        q_R[i0-1:iend+1,:,:] = Q[i0-1:iend+1,:,:] + np.minimum(2.0*abs(dQ_mono[i0-1:iend+1,:,:]), abs(q_R[i0-1:iend+1,:,:]-Q[i0-1:iend+1,:,:])) * np.sign(2.0*dQ_mono[i0-1:iend+1,:,:])

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    dq[i0-1:iend+1,:,:]= q_R[i0-1:iend+1,:,:] - q_L[i0-1:iend+1,:,:]
    q6[i0-1:iend+1,:,:] = 6*Q[i0-1:iend+1,:,:] - 3*(q_R[i0-1:iend+1,:,:] + q_L[i0-1:iend+1,:,:])

    return dq, q6, q_L, q_R

def ppm_reconstruction_y(Q, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.nghost
    j0 = cs_grid.j0
    jend = cs_grid.jend

    # Aux vars
    q_L = np.zeros((N+ng, M+ng+1, nbfaces))
    q_R = np.zeros((N+ng, M+ng+1, nbfaces))
    dq  = np.zeros((N+ng, M+ng, nbfaces))
    q6  = np.zeros((N+ng, M+ng, nbfaces))

    if simulation.recon_name == 'PPM': # PPM from CW84 paper
        # Values of Q at right edges (q_(j+1/2)) - Formula 1.9 from Collela and Woodward 1984
        Q_edges = np.zeros((N+ng, M+ng+1, nbfaces))
        Q_edges[:,j0-1:jend+2,:] = (7.0/12.0)*(Q[:,j0-1:jend+2,:] + Q[:,j0-2:jend+1,:]) - (Q[:,j0:jend+3,:] + Q[:,j0-3:jend,:])/12.0

        # Assign values of Q_R and Q_L
        q_R[:,j0-1:jend+1,:] = Q_edges[:,j0:jend+2,:]
        q_L[:,j0-1:jend+1,:] = Q_edges[:,j0-1:jend+1,:]

    elif simulation.recon_name == 'PPM_hybrid': # Hybrid PPM from PL07
        # coeffs from equations 41 and 42 from PL07
        a1 =   2.0/60.0
        a2 = -13.0/60.0
        a3 =  47.0/60.0
        a4 =  27.0/60.0
        a5 =  -3.0/60.0
        # Assign values of Q_R and Q_L
        q_R[:,j0-1:jend+1,:] = a1*Q[:,j0-3:jend-1,:] + a2*Q[:,j0-2:jend,:] + a3*Q[:,j0-1:jend+1,:] + a4*Q[:,j0:jend+2,:] + a5*Q[:,j0+1:jend+3,:]
        q_L[:,j0-1:jend+1,:] = a5*Q[:,j0-3:jend-1,:] + a4*Q[:,j0-2:jend,:] + a3*Q[:,j0-1:jend+1,:] + a2*Q[:,j0:jend+2,:] + a1*Q[:,j0+1:jend+3,:]


    elif simulation.recon_name == 'PPM_mono_CW84':  #PPM with monotonization from CW84
        # Compute the slopes dQ0 (centered finite difference)
        # Formula 1.7 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        dQ0 = np.zeros((N+ng, M+ng, nbfaces))
        dQ0[:,j0-2:jend+2,:] = 0.5*(Q[:,j0-1:jend+3,:] - Q[:,j0-3:jend+1,:]) # Interior values are in 3:N+3

        # Compute the slopes dQ (1-sided finite difference)
        # Formula 1.8 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        dQ1 = np.zeros((N+ng, M+ng, nbfaces)) # Right 1-sided difference
        dQ2 = np.zeros((N+ng, M+ng, nbfaces)) # Left  1-sided difference

        # Right 1-sided finite difference
        dQ1[:,j0-2:jend+2,:] = Q[:,j0-1:jend+3,:] - Q[:,j0-2:jend+2,:]
        dQ1 = dQ1*2.0

        # Left  1-sided finite difference
        dQ2[:,j0-2:jend+2,:] = Q[:,j0-2:jend+2,:] - Q[:,j0-3:jend+1,:]
        dQ2 = dQ2*2.0

        # Final slope - Formula 1.8 from Collela and Woodward 1984
        dQ = np.zeros((N+ng, M+ng, nbfaces))
        dQ[:,j0-2:jend+2,:] = np.minimum(abs(dQ0[:,j0-2:jend+2,:]), abs(dQ1[:,j0-2:jend+2,:]))
        dQ[:,j0-2:jend+2,:] = np.minimum(dQ[:,j0-2:jend+2,:], abs(dQ2[:,j0-2:jend+2,:]))*np.sign(dQ0[:,j0-2:jend+2,:])
        mask = ( (Q[:,j0-1:jend+3,:] - Q[:,j0-2:jend+2,:]) * (Q[:,j0-2:jend+2,:] - Q[:,j0-3:jend+1,:]) > 0.0 ) # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        dQ[:,j0-2:jend+2,:][~mask] = 0.0

        # Values of Q at right edges (q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1984
        Q_edges = np.zeros((N+ng, M+ng+1, nbfaces))
        Q_edges[:,j0-1:jend+2,:] = 0.5*(Q[:,j0-1:jend+2,:] + Q[:,j0-2:jend+1,:]) - (dQ[:,j0-1:jend+2,:] - dQ[:,j0-2:jend+1,:])/6.0

        # Assign values of Q_R and Q_L
        q_R[:,j0-1:jend+1,:] = Q_edges[:,j0:jend+2,:]
        q_L[:,j0-1:jend+1,:] = Q_edges[:,j0-1:jend+1,:]

        # Compute the polynomial coefs
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        dq[:,j0-1:jend+1,:]= q_R[:,j0-1:jend+1,:] - q_L[:,j0-1:jend+1,:]
        q6[:,j0-1:jend+1,:] = 6*Q[:,j0-1:jend+1,:] - 3*(q_R[:,j0-1:jend+1,:] + q_L[:,j0-1:jend+1,:])

        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        local_maximum = (q_R[:,j0-1:jend+1,:]-Q[:,j0-1:jend+1,:])*(Q[:,j0-1:jend+1,:]-q_L[:,j0-1:jend+1,:])<=0

        # In this case (local maximum), the interpolation is a constant equal to Q
        q_R[:,j0-1:jend+1,:][local_maximum] = Q[:,j0-1:jend+1,:][local_maximum]
        q_L[:,j0-1:jend+1,:][local_maximum] = Q[:,j0-1:jend+1,:][local_maximum]

        # Check overshot
        overshoot  = (abs(dq[:,j0-1:jend+1,:]) < abs(q6[:,j0-1:jend+1,:]))

        # Move left
        move_left  = (q_R[:,j0-1:jend+1,:]-q_L[:,j0-1:jend+1,:])*(Q[:,j0-1:jend+1,:]-0.5*(q_R[:,j0-1:jend+1,:]+q_L[:,j0-1:jend+1,:])) > ((q_R[:,j0-1:jend+1,:]-q_L[:,j0-1:jend+1,:])**2)/6.0

        # Move right
        move_right = (-((q_R[:,j0-1:jend+1,:]-q_L[:,j0-1:jend+1,:])**2)/6.0 > (q_R[:,j0-1:jend+1,:]-q_L[:,j0-1:jend+1,:])*(Q[:,j0-1:jend+1,:]-0.5*(q_R[:,j0-1:jend+1,:]+q_L[:,j0-1:jend+1,:])) )

        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        q_L[:,j0-1:jend+1,:][overshoot_move_left]  = 3.0*Q[:,j0-1:jend+1,:][overshoot_move_left]  - 2.0*q_R[:,j0-1:jend+1,:][overshoot_move_left]
        q_R[:,j0-1:jend+1,:][overshoot_move_right] = 3.0*Q[:,j0-1:jend+1,:][overshoot_move_right] - 2.0*q_L[:,j0-1:jend+1,:][overshoot_move_right]

    elif simulation.recon_name == 'PPM_mono_L04':  #PPM with monotonization from Lin 04 paper
        # Formula B1 from Lin 04
        dQ      = np.zeros((N+ng,M+ng,nbfaces))
        dQ_min  = np.zeros((N+ng,M+ng,nbfaces))
        dQ_max  = np.zeros((N+ng,M+ng,nbfaces))
        dQ_mono = np.zeros((N+ng,M+ng,nbfaces))

        dQ[:,j0-3:jend+3,:] = 0.25*(Q[:,j0-2:jend+4,:] - Q[:,j0-4:jend+2,:])
        dQ_min[:,j0-3:jend+3,:]  = np.maximum(np.maximum(Q[:,j0-4:jend+2,:], Q[:,j0-3:jend+3,:]), Q[:,j0-2:jend+4,:]) - Q[:,j0-3:jend+3,:]
        dQ_max[:,j0-3:jend+3,:]  = Q[:,j0-3:jend+3,:] - np.minimum(np.minimum(Q[:,j0-4:jend+2,:], Q[:,j0-3:jend+3,:]), Q[:,j0-2:jend+4,:])
        dQ_mono[:,j0-3:jend+3,:] = np.minimum(np.minimum(abs(dQ[:,j0-3:jend+3,:]), dQ_min[:,j0-3:jend+3,:]), dQ_max[:,j0-3:jend+3,:]) * np.sign(dQ[:,j0-3:jend+3,:])

        # Formula B2 from Lin 04
        Q_edges = np.zeros((N+ng,M+ng+1,nbfaces))
        Q_edges[:,j0-1:jend+2,:] = 0.5*(Q[:,j0-1:jend+2,:] + Q[:,j0-2:jend+1,:]) - (dQ_mono[:,j0-1:jend+2,:] - dQ_mono[:,j0-2:jend+1,:])/3.0

        # Assign values of Q_R and Q_L
        q_R[:,j0-1:jend+1,:] = Q_edges[:,j0:jend+2,:]
        q_L[:,j0-1:jend+1,:] = Q_edges[:,j0-1:jend+1,:]

        # Formula B3 from Lin 04
        q_L[:,j0-1:jend+1,:] = Q[:,j0-1:jend+1,:] - np.minimum(2.0*abs(dQ_mono[:,j0-1:jend+1,:]), abs(q_L[:,j0-1:jend+1,:]-Q[:,j0-1:jend+1,:])) * np.sign(2.0*dQ_mono[:,j0-1:jend+1,:])

        # Formula B4 from Lin 04
        q_R[:,j0-1:jend+1,:] = Q[:,j0-1:jend+1,:] + np.minimum(2.0*abs(dQ_mono[:,j0-1:jend+1,:]), abs(q_R[:,j0-1:jend+1,:]-Q[:,j0-1:jend+1,:])) * np.sign(2.0*dQ_mono[:,j0-1:jend+1,:])

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    dq[:,j0-1:jend+1,:]= q_R[:,j0-1:jend+1,:] - q_L[:,j0-1:jend+1,:]
    q6[:,j0-1:jend+1,:] = 6*Q[:,j0-1:jend+1,:] - 3*(q_R[:,j0-1:jend+1,:] + q_L[:,j0-1:jend+1,:])
    return dq, q6, q_L, q_R
