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
import numexpr as ne
from constants import nbfaces
from edges_treatment import average_parabola_cube_edges, edges_extrapolation
####################################################################################
# Given the average values of a scalar field Q, this routine constructs
# a piecewise parabolic aproximation of Q using its average value.
####################################################################################
def ppm_reconstruction_x(Q, px, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.nghost
    i0 = cs_grid.i0
    iend = cs_grid.iend

    if px.recon_name == 'PPM-0': # PPM from CW84 paper
        # Values of Q at right edges (q_(j+1/2)) - Formula 1.9 from Collela and Woodward 1984
        #px.Q_edges[i0-1:iend+2,:] = (7.0/12.0)*(Q[i0-1:iend+2,:] + Q[i0-2:iend+1,:]) - (Q[i0:iend+3,:] + Q[i0-3:iend,:])/12.0
        Q1 = Q[i0-1:iend+2,:,:]
        Q2 = Q[i0-2:iend+1,:,:]
        Q3 = Q[i0:iend+3,:,:]
        Q4 = Q[i0-3:iend,:,:]
        px.Q_edges[i0-1:iend+2,:,:] = ne.evaluate("(7.0/12.0)*(Q1+Q2) - (Q3+Q4)/12.0")
        # Assign values of Q_R and Q_L
        px.q_R[i0-1:iend+1,:,:] = px.Q_edges[i0:iend+2,:,:]
        px.q_L[i0-1:iend+1,:,:] = px.Q_edges[i0-1:iend+1,:,:]


    elif px.recon_name == 'PPM-PL07': # Hybrid PPM from PL07
        # coeffs from equations 41 and 42 from PL07
        a1 =   2.0/60.0
        a2 = -13.0/60.0
        a3 =  47.0/60.0
        a4 =  27.0/60.0
        a5 =  -3.0/60.0
        # Assign values of Q_R and Q_L
        #px.q_R[i0-1:iend+1,:] = a1*Q[i0-3:iend-1,:] + a2*Q[i0-2:iend,:] + a3*Q[i0-1:iend+1,:] + a4*Q[i0:iend+2,:] + a5*Q[i0+1:iend+3,:]
        #px.q_L[i0-1:iend+1,:] = a5*Q[i0-3:iend-1,:] + a4*Q[i0-2:iend,:] + a3*Q[i0-1:iend+1,:] + a2*Q[i0:iend+2,:] + a1*Q[i0+1:iend+3,:]
        q1 = Q[i0-3:iend-1,:,:]
        q2 = Q[i0-2:iend,:,:]
        q3 = Q[i0-1:iend+1,:,:]
        q4 = Q[i0:iend+2,:,:]
        q5 = Q[i0+1:iend+3,:,:]
        px.q_R[i0-1:iend+1,:,:] = ne.evaluate('a1*q1 + a2*q2 + a3*q3 + a4*q4 + a5*q5')
        px.q_L[i0-1:iend+1,:,:] = ne.evaluate('a5*q1 + a4*q2 + a3*q3 + a2*q4 + a1*q5')

    elif px.recon_name == 'PPM-CW84':  #PPM with monotonization from CW84
        # Compute the slopes dQ0 (centered finite difference)
        # Formula 1.7 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        Q1 = Q[i0-3:iend+1,:,:]
        Q2 = Q[i0-2:iend+2,:,:]
        Q3 = Q[i0-1:iend+3,:,:]
        #px.dQ0[i0-2:iend+2,:] = 0.5*(Q[i0-1:iend+3,:] - Q[i0-3:iend+1,:])
        px.dQ0[i0-2:iend+2,:,:] = ne.evaluate('0.5*(Q3-Q1)')

        #Avoid overshoot
        # Compute the slopes dQ (1-sided finite difference)
        # Formula 1.8 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        # Right 1-sided finite difference
        #px.dQ1[i0-2:iend+2,:] = Q[i0-1:iend+3,:] - Q[i0-2:iend+2,:]
        #px.dQ1 = px.dQ1*2.0
        px.dQ1[i0-2:iend+2,:,:] = ne.evaluate('2.0*(Q3-Q2)')

        # Left  1-sided finite difference
        #px.dQ2[i0-2:iend+2,:] = Q[i0-2:iend+2,:] - Q[i0-3:iend+1,:]
        #px.dQ2 = px.dQ2*2.0
        px.dQ2[i0-2:iend+2,:,:] = ne.evaluate('2.0*(Q2-Q1)')

        # Final slope - Formula 1.8 from Collela and Woodward 1984
        px.dQ[i0-2:iend+2,:,:] = np.minimum(abs(px.dQ0[i0-2:iend+2,:,:]), abs(px.dQ1[i0-2:iend+2,:,:]))
        px.dQ[i0-2:iend+2,:,:] = np.minimum(px.dQ[i0-2:iend+2,:,:], abs(px.dQ2[i0-2:iend+2,:,:]))*np.sign(px.dQ0[i0-2:iend+2,:,:])
        #mask = ( (Q[i0-1:iend+3,:] - Q[i0-2:iend+2,:]) * (Q[i0-2:iend+2,:] - Q[i0-3:iend+1,:]) > 0.0 ) # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        Q1 = Q[i0-1:iend+3,:,:]
        Q2 = Q[i0-2:iend+2,:,:]
        Q3 = Q[i0-3:iend+1,:,:]
        mask = ne.evaluate('(Q1-Q2)*(Q2-Q3)>0.0') # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        px.dQ[i0-2:iend+2,:,:][~mask] = 0.0

        # Values of Q at right edges (q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1984
        #px.Q_edges[i0-1:iend+2,:] = 0.5*(Q[i0-1:iend+2,:] + Q[i0-2:iend+1,:]) - (px.dQ[i0-1:iend+2,:] - px.dQ[i0-2:iend+1,:])/6.0
        Q1 = Q[i0-1:iend+2,:,:]
        Q2 = Q[i0-2:iend+1,:,:]
        dQ1 = px.dQ[i0-1:iend+2,:,:]
        dQ2 = px.dQ[i0-2:iend+1,:,:]
        px.Q_edges[i0-1:iend+2,:,:] = ne.evaluate('0.5*(Q1+Q2)-(dQ1-dQ2)/6.0')

        # Assign values of Q_R and Q_L
        px.q_R[i0-1:iend+1,:,:] = px.Q_edges[i0:iend+2,:,:]
        px.q_L[i0-1:iend+1,:,:] = px.Q_edges[i0-1:iend+1,:,:]

        # Compute the polynomial coefs
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        #px.dq[i0-1:iend+1,:] = px.q_R[i0-1:iend+1,:] - px.q_L[i0-1:iend+1,:]
        #px.q6[i0-1:iend+1,:] = 6*Q[i0-1:iend+1,:] - 3*(px.q_R[i0-1:iend+1,:] + px.q_L[i0-1:iend+1,:])
        q_R = px.q_R[i0-1:iend+1,:,:]
        q_L = px.q_L[i0-1:iend+1,:,:]
        q = Q[i0-1:iend+1,:,:]
        px.dq[i0-1:iend+1,:,:] = ne.evaluate('q_R - q_L')
        px.q6[i0-1:iend+1,:,:] = ne.evaluate('6*q - 3*(q_R+q_L)')

        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        #local_maximum = (px.q_R[i0-1:iend+1,:]-Q[i0-1:iend+1,:])*(Q[i0-1:iend+1,:]-px.q_L[i0-1:iend+1,:])<=0
        local_maximum = ne.evaluate('(q_R-q)*(q-q_L)<=0')

        # In this case (local maximum), the interpolation is a constant equal to Q
        px.q_R[i0-1:iend+1,:,:][local_maximum] = Q[i0-1:iend+1,:,:][local_maximum]
        px.q_L[i0-1:iend+1,:,:][local_maximum] = Q[i0-1:iend+1,:,:][local_maximum]

        # Check overshot
        #overshoot  = (abs(px.dq[i0-1:iend+1,:]) < abs(px.q6[i0-1:iend+1,:]))
        dq = px.dq[i0-1:iend+1,:,:]
        q6 = px.q6[i0-1:iend+1,:,:]
        overshoot  = ne.evaluate('abs(dq) < abs(q6)')

        # Move left
        #move_left  = (px.q_R[i0-1:iend+1,:]-px.q_L[i0-1:iend+1,:])*(Q[i0-1:iend+1,:]-0.5*(px.q_R[i0-1:iend+1,:]+px.q_L[i0-1:iend+1,:])) > ((px.q_R[i0-1:iend+1,:]-px.q_L[i0-1:iend+1,:])**2)/6.0
        q_R = px.q_R[i0-1:iend+1,:,:]
        q_L = px.q_L[i0-1:iend+1,:,:]
        q = Q[i0-1:iend+1,:,:]
        move_left  = ne.evaluate('(q_R-q_L)*(q-0.5*(q_R+q_L)) > ((q_R-q_L)**2)/6.0')

        # Move right
        #move_right = (-((px.q_R[i0-1:iend+1,:]-px.q_L[i0-1:iend+1,:])**2)/6.0 > (px.q_R[i0-1:iend+1,:]-px.q_L[i0-1:iend+1,:])*(Q[i0-1:iend+1,:]-0.5*(px.q_R[i0-1:iend+1,:]+px.q_L[i0-1:iend+1,:])) )
        move_right = ne.evaluate('-((q_R-q_L)**2)/6.0 > (q_R-q_L)*(q-0.5*(q_R+q_L))')
        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        #px.q_L[i0-1:iend+1,:][overshoot_move_left]  = 3.0*Q[i0-1:iend+1,:][overshoot_move_left]  - 2.0*px.q_R[i0-1:iend+1,:][overshoot_move_left]
        #px.q_R[i0-1:iend+1,:][overshoot_move_right] = 3.0*Q[i0-1:iend+1,:][overshoot_move_right] - 2.0*px.q_L[i0-1:iend+1,:][overshoot_move_right]
        q = Q[i0-1:iend+1,:,:][overshoot_move_left]
        q_R = px.q_R[i0-1:iend+1,:,:][overshoot_move_left]
        px.q_L[i0-1:iend+1,:,:][overshoot_move_left]  = ne.evaluate('3.0*q-2.0*q_R')
        q = Q[i0-1:iend+1,:,:][overshoot_move_right]
        q_L = px.q_L[i0-1:iend+1,:,:][overshoot_move_right]
        px.q_R[i0-1:iend+1,:,:][overshoot_move_right] = ne.evaluate('3.0*q-2.0*q_L')

    elif px.recon_name == 'PPM-L04':  #PPM with monotonization from Lin 04 paper
        # Formula B1 from Lin 04
        #px.dQ[i0-3:iend+3,:] = 0.25*(Q[i0-2:iend+4,:] - Q[i0-4:iend+2,:])
        Q1 = Q[i0-2:iend+4,:,:]
        Q2 = Q[i0-4:iend+2,:,:]
        px.dQ[i0-3:iend+3,:,:] = ne.evaluate('0.25*(Q1-Q2)')

        #px.dQ_min[i0-3:iend+3,:]  = np.maximum(np.maximum(Q[i0-4:iend+2,:], Q[i0-3:iend+3,:]), Q[i0-2:iend+4,:]) - Q[i0-3:iend+3,:]
        #px.dQ_max[i0-3:iend+3,:]  = Q[i0-3:iend+3,:] - np.minimum(np.minimum(Q[i0-4:iend+2,:], Q[i0-3:iend+3,:]), Q[i0-2:iend+4,:])
        Q_max = np.maximum(np.maximum(Q[i0-4:iend+2,:,:], Q[i0-3:iend+3,:,:]), Q[i0-2:iend+4,:,:])
        Q_min = np.minimum(np.minimum(Q[i0-4:iend+2,:,:], Q[i0-3:iend+3,:,:]), Q[i0-2:iend+4,:,:])
        Q0 = Q[i0-3:iend+3,:,:]
        px.dQ_min[i0-3:iend+3,:,:]  = ne.evaluate('Q_max-Q0')
        px.dQ_max[i0-3:iend+3,:,:]  = ne.evaluate('Q0-Q_min')
        px.dQ_mono[i0-3:iend+3,:,:] = np.minimum(np.minimum(abs(px.dQ[i0-3:iend+3,:,:]),px.dQ_min[i0-3:iend+3,:,:]),px.dQ_max[i0-3:iend+3,:,:])*np.sign(px.dQ[i0-3:iend+3,:,:])

        # Formula B2 from Lin 04
        #px.Q_edges[i0-1:iend+2,:] = 0.5*(Q[i0-1:iend+2,:] + Q[i0-2:iend+1,:]) - (px.dQ_mono[i0-1:iend+2,:] - px.dQ_mono[i0-2:iend+1,:])/3.0
        Q1 = Q[i0-1:iend+2,:,:]
        Q2 = Q[i0-2:iend+1,:,:]
        dQ1= px.dQ_mono[i0-1:iend+2,:,:]
        dQ2= px.dQ_mono[i0-2:iend+1,:,:]
        px.Q_edges[i0-1:iend+2,:,:] = ne.evaluate("0.5*(Q1+Q2) - (dQ1-dQ2)/3.0")

        # Assign values of Q_R and Q_L
        px.q_R[i0-1:iend+1,:,:] = px.Q_edges[i0:iend+2,:,:]
        px.q_L[i0-1:iend+1,:,:] = px.Q_edges[i0-1:iend+1,:,:]

        # Formula B3 from Lin 04
        #px.q_L[i0-1:iend+1,:] = Q[i0-1:iend+1,:] - np.minimum(2.0*abs(px.dQ_mono[i0-1:iend+1,:]), abs(px.q_L[i0-1:iend+1,:]-Q[i0-1:iend+1,:])) * np.sign(2.0*px.dQ_mono[i0-1:iend+1,:])
        qmin =  np.minimum(2.0*abs(px.dQ_mono[i0-1:iend+1,:,:]), abs(px.q_L[i0-1:iend+1,:,:]-Q[i0-1:iend+1,:,:])) * np.sign(2.0*px.dQ_mono[i0-1:iend+1,:,:])
        q0 = Q[i0-1:iend+1,:,:]
        px.q_L[i0-1:iend+1,:,:] = ne.evaluate('q0-qmin')

        # Formula B4 from Lin 04
        #px.q_R[i0-1:iend+1,:] = Q[i0-1:iend+1,:] + np.minimum(2.0*abs(px.dQ_mono[i0-1:iend+1,:]), abs(px.q_R[i0-1:iend+1,:]-Q[i0-1:iend+1,:])) * np.sign(2.0*px.dQ_mono[i0-1:iend+1,:])
        qmin = np.minimum(2.0*abs(px.dQ_mono[i0-1:iend+1,:,:]), abs(px.q_R[i0-1:iend+1,:,:]-Q[i0-1:iend+1,:,:])) * np.sign(2.0*px.dQ_mono[i0-1:iend+1,:,:])
        px.q_R[i0-1:iend+1,:,:] = ne.evaluate('q0+qmin')

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    #px.dq[i0-1:iend+1,:] = px.q_R[i0-1:iend+1,:] - px.q_L[i0-1:iend+1,:]
    #px.q6[i0-1:iend+1,:] = 6*Q[i0-1:iend+1,:] - 3*(px.q_R[i0-1:iend+1,:] + px.q_L[i0-1:iend+1,:])
    q_L =  px.q_L[i0-1:iend+1,:,:]
    q_R =  px.q_R[i0-1:iend+1,:,:]
    q = Q[i0-1:iend+1,:,:]
    px.dq[i0-1:iend+1,:,:] = ne.evaluate('q_R-q_L')
    px.q6[i0-1:iend+1,:,:] = ne.evaluate('6*q- 3*(q_R + q_L)')

def ppm_reconstruction_y(Q, py, cs_grid, simulation):
    N = cs_grid.N
    M = cs_grid.N
    ng = cs_grid.nghost
    j0 = cs_grid.j0
    jend = cs_grid.jend

    if py.recon_name == 'PPM-0': # PPM from CW84 paper
        # Values of Q at right edges (q_(j+1/2)) - Formula 1.9 from Collela and Woodward 1984
        #py.Q_edges[:,j0-1:jend+2] = (7.0/12.0)*(Q[:,j0-1:jend+2] + Q[:,j0-2:jend+1]) - (Q[:,j0:jend+3] + Q[:,j0-3:jend])/12.0
        Q1 = Q[:,j0-1:jend+2,:]
        Q2 = Q[:,j0-2:jend+1,:]
        Q3 = Q[:,j0:jend+3,:]
        Q4 = Q[:,j0-3:jend,:]
        py.Q_edges[:,j0-1:jend+2,:] = ne.evaluate("(7.0/12.0)*(Q1+Q2) - (Q3+Q4)/12.0")

        # Assign values of Q_R and Q_L
        py.q_R[:,j0-1:jend+1,:] = py.Q_edges[:,j0:jend+2,:]
        py.q_L[:,j0-1:jend+1,:] = py.Q_edges[:,j0-1:jend+1,:]


    elif py.recon_name == 'PPM-PL07': # Hybrid PPM from PL07
        # coeffs from equations 41 and 42 from PL07
        a1 =   2.0/60.0
        a2 = -13.0/60.0
        a3 =  47.0/60.0
        a4 =  27.0/60.0
        a5 =  -3.0/60.0
        # Assign values of Q_R and Q_L
        #py.q_R[:,j0-1:jend+1] = a1*Q[:,j0-3:jend-1] + a2*Q[:,j0-2:jend] + a3*Q[:,j0-1:jend+1] + a4*Q[:,j0:jend+2] + a5*Q[:,j0+1:jend+3]
        #py.q_L[:,j0-1:jend+1] = a5*Q[:,j0-3:jend-1] + a4*Q[:,j0-2:jend] + a3*Q[:,j0-1:jend+1] + a2*Q[:,j0:jend+2] + a1*Q[:,j0+1:jend+3]
        q1 = Q[:,j0-3:jend-1,:]
        q2 = Q[:,j0-2:jend,:]
        q3 = Q[:,j0-1:jend+1,:]
        q4 = Q[:,j0:jend+2,:]
        q5 = Q[:,j0+1:jend+3,:]
        py.q_R[:,j0-1:jend+1,:] = ne.evaluate('a1*q1 + a2*q2 + a3*q3 + a4*q4 + a5*q5')
        py.q_L[:,j0-1:jend+1,:] = ne.evaluate('a5*q1 + a4*q2 + a3*q3 + a2*q4 + a1*q5')

    elif py.recon_name == 'PPM-CW84':  #PPM with monotonization from CW84
        # Compute the slopes dQ0 (centered finite difference)
        # Formula 1.7 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        Q1 = Q[:,j0-3:jend+1,:]
        Q2 = Q[:,j0-2:jend+2,:]
        Q3 = Q[:,j0-1:jend+3,:]
        #py.dQ0[:,j0-2:jend+2] = 0.5*(Q[:,j0-1:jend+3] - Q[:,j0-3:jend+1])
        py.dQ0[:,j0-2:jend+2,:] = ne.evaluate('0.5*(Q3-Q1)')

        #Avoid overshoot
        # Compute the slopes dQ (1-sided finite difference)
        # Formula 1.8 from Collela and Woodward 1984 and Figure 2 from Carpenter et al 1990.
        # Right 1-sided finite difference
        #py.dQ1[:,j0-2:jend+2] = Q[:,j0-1:jend+3] - Q[:,j0-2:jend+2]
        #py.dQ1 = py.dQ1*2.0
        py.dQ1[:,j0-2:jend+2,:] = ne.evaluate('2.0*(Q3-Q2)')

        # Left  1-sided finite difference
        #py.dQ2[:,j0-2:jend+2] = Q[:,j0-2:jend+2] - Q[:,j0-3:jend+1]
        #py.dQ2 = py.dQ2*2.0
        py.dQ2[:,j0-2:jend+2,:] = ne.evaluate('2.0*(Q2-Q1)')

        # Final slope - Formula 1.8 from Collela and Woodward 1984
        py.dQ[:,j0-2:jend+2,:] = np.minimum(abs(py.dQ0[:,j0-2:jend+2,:]), abs(py.dQ1[:,j0-2:jend+2,:]))
        py.dQ[:,j0-2:jend+2,:] = np.minimum(py.dQ[:,j0-2:jend+2,:], abs(py.dQ2[:,j0-2:jend+2,:]))*np.sign(py.dQ0[:,j0-2:jend+2,:])

        #mask = ( (Q[:,j0-1:jend+3] - Q[:,j0-2:jend+2]) * (Q[:,j0-2:jend+2] - Q[:,j0-3:jend+1]) > 0.0 ) # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        Q1 = Q[:,j0-1:jend+3,:]
        Q2 = Q[:,j0-2:jend+2,:]
        Q3 = Q[:,j0-3:jend+1,:]
        mask = ne.evaluate('(Q1-Q2)*(Q2-Q3)>0.0') # Indexes where (Q_{j+1}-Q_{j})*(Q_{j}-Q{j-1}) > 0
        py.dQ[:,j0-2:jend+2,:][~mask] = 0.0

        # Values of Q at right edges (q_(j+1/2)) - Formula 1.6 from Collela and Woodward 1984
        #py.Q_edges[:,j0-1:jend+2] = 0.5*(Q[:,j0-1:jend+2] + Q[:,j0-2:jend+1]) - (py.dQ[:,j0-1:jend+2] - py.dQ[:,j0-2:jend+1])/6.0
        Q1 = Q[:,j0-1:jend+2,:]
        Q2 = Q[:,j0-2:jend+1,:]
        dQ1 = py.dQ[:,j0-1:jend+2,:]
        dQ2 = py.dQ[:,j0-2:jend+1,:]
        py.Q_edges[:,j0-1:jend+2,:] = ne.evaluate('0.5*(Q1+Q2)-(dQ1-dQ2)/6.0')

        # Assign values of Q_R and Q_L
        py.q_R[:,j0-1:jend+1,:] = py.Q_edges[:,j0:jend+2,:]
        py.q_L[:,j0-1:jend+1,:] = py.Q_edges[:,j0-1:jend+1,:]

        # Compute the polynomial coefs
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        #py.dq[:,j0-1:jend+1] = py.q_R[:,j0-1:jend+1] - py.q_L[:,j0-1:jend+1]
        #py.q6[:,j0-1:jend+1] = 6*Q[:,j0-1:jend+1] - 3*(py.q_R[:,j0-1:jend+1] + py.q_L[:,j0-1:jend+1])
        q_R = py.q_R[:,j0-1:jend+1,:]
        q_L = py.q_L[:,j0-1:jend+1,:]
        q = Q[:,j0-1:jend+1,:]
        py.dq[:,j0-1:jend+1,:] = ne.evaluate('q_R - q_L')
        py.q6[:,j0-1:jend+1,:] = ne.evaluate('6*q - 3*(q_R+q_L)')

        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        #local_maximum = (py.q_R[:,j0-1:jend+1]-Q[:,j0-1:jend+1])*(Q[:,j0-1:jend+1]-py.q_L[:,j0-1:jend+1])<=0
        local_maximum = ne.evaluate('(q_R-q)*(q-q_L)<=0')

        # In this case (local maximum), the interpolation is a constant equal to Q
        py.q_R[:,j0-1:jend+1,:][local_maximum] = Q[:,j0-1:jend+1,:][local_maximum]
        py.q_L[:,j0-1:jend+1,:][local_maximum] = Q[:,j0-1:jend+1,:][local_maximum]

        # Check overshot
        #overshoot  = (abs(py.dq[:,j0-1:jend+1]) < abs(py.q6[:,j0-1:jend+1]))
        dq = py.dq[:,j0-1:jend+1,:]
        q6 = py.q6[:,j0-1:jend+1,:]
        overshoot  = ne.evaluate('abs(dq) < abs(q6)')

        # Move left
        #move_left  = (py.q_R[:,j0-1:jend+1]-py.q_L[:,j0-1:jend+1])*(Q[:,j0-1:jend+1]-0.5*(py.q_R[:,j0-1:jend+1]+py.q_L[:,j0-1:jend+1])) > ((py.q_R[:,j0-1:jend+1]-py.q_L[:,j0-1:jend+1])**2)/6.0
        q_R = py.q_R[:,j0-1:jend+1,:]
        q_L = py.q_L[:,j0-1:jend+1,:]
        q = Q[:,j0-1:jend+1,:]
        move_left  = ne.evaluate('(q_R-q_L)*(q-0.5*(q_R+q_L)) > ((q_R-q_L)**2)/6.0')

        # Move right
        #move_right = (-((py.q_R[:,j0-1:jend+1]-py.q_L[:,j0-1:jend+1])**2)/6.0 > (py.q_R[:,j0-1:jend+1]-py.q_L[:,j0-1:jend+1])*(Q[:,j0-1:jend+1]-0.5*(py.q_R[:,j0-1:jend+1]+py.q_L[:,j0-1:jend+1])) )
        move_right = ne.evaluate('-((q_R-q_L)**2)/6.0 > (q_R-q_L)*(q-0.5*(q_R+q_L))')

        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        #py.q_L[:,j0-1:jend+1][overshoot_move_left]  = 3.0*Q[:,j0-1:jend+1][overshoot_move_left]  - 2.0*py.q_R[:,j0-1:jend+1][overshoot_move_left]
        #py.q_R[:,j0-1:jend+1][overshoot_move_right] = 3.0*Q[:,j0-1:jend+1][overshoot_move_right] - 2.0*py.q_L[:,j0-1:jend+1][overshoot_move_right]
        q = Q[:,j0-1:jend+1,:][overshoot_move_left]
        q_R = py.q_R[:,j0-1:jend+1,:][overshoot_move_left]
        py.q_L[:,j0-1:jend+1,:][overshoot_move_left]  = ne.evaluate('3.0*q-2.0*q_R')
        q = Q[:,j0-1:jend+1,:][overshoot_move_right]
        q_L = py.q_L[:,j0-1:jend+1,:][overshoot_move_right]
        py.q_R[:,j0-1:jend+1,:][overshoot_move_right] = ne.evaluate('3.0*q-2.0*q_L')

    elif py.recon_name == 'PPM-L04':  #PPM with monotonization from Lin 04 paper
        # Formula B1 from Lin 04
        #py.dQ[:,j0-3:jend+3] = 0.25*(Q[:,j0-2:jend+4] - Q[:,j0-4:jend+2])
        Q1 = Q[:,j0-2:jend+4,:]
        Q2 = Q[:,j0-4:jend+2,:]
        py.dQ[:,j0-3:jend+3,:] = ne.evaluate("0.25*(Q1-Q2)")

        #py.dQ_min[:,j0-3:jend+3]  = np.maximum(np.maximum(Q[:,j0-4:jend+2], Q[:,j0-3:jend+3]), Q[:,j0-2:jend+4]) - Q[:,j0-3:jend+3]
        #py.dQ_max[:,j0-3:jend+3]  = Q[:,j0-3:jend+3] - np.minimum(np.minimum(Q[:,j0-4:jend+2], Q[:,j0-3:jend+3]), Q[:,j0-2:jend+4])
        Q_max = np.maximum(np.maximum(Q[:,j0-4:jend+2,:], Q[:,j0-3:jend+3,:]), Q[:,j0-2:jend+4,:])
        Q_min = np.minimum(np.minimum(Q[:,j0-4:jend+2,:], Q[:,j0-3:jend+3,:]), Q[:,j0-2:jend+4,:])
        Q0 = Q[:,j0-3:jend+3,:]
        py.dQ_min[:,j0-3:jend+3,:]  = ne.evaluate('Q_max-Q0')
        py.dQ_max[:,j0-3:jend+3,:]  = ne.evaluate('Q0-Q_min')
        py.dQ_mono[:,j0-3:jend+3,:] = np.minimum(np.minimum(abs(py.dQ[:,j0-3:jend+3,:]), py.dQ_min[:,j0-3:jend+3,:]), py.dQ_max[:,j0-3:jend+3,:]) * np.sign(py.dQ[:,j0-3:jend+3,:])

        # Formula B2 from Lin 04
        #py.Q_edges[:,j0-1:jend+2] = 0.5*(Q[:,j0-1:jend+2] + Q[:,j0-2:jend+1]) - (py.dQ_mono[:,j0-1:jend+2] - py.dQ_mono[:,j0-2:jend+1])/3.0
        Q1 = Q[:,j0-1:jend+2,:]
        Q2 = Q[:,j0-2:jend+1,:]
        dQ1= py.dQ_mono[:,j0-1:jend+2,:]
        dQ2= py.dQ_mono[:,j0-2:jend+1,:]
        py.Q_edges[:,j0-1:jend+2,:] = ne.evaluate("0.5*(Q1+Q2) - (dQ1-dQ2)/3.0")

        # Assign values of Q_R and Q_L
        py.q_R[:,j0-1:jend+1,:] = py.Q_edges[:,j0:jend+2,:]
        py.q_L[:,j0-1:jend+1,:] = py.Q_edges[:,j0-1:jend+1,:]

        # Formula B3 from Lin 04
        #py.q_L[:,j0-1:jend+1] = Q[:,j0-1:jend+1] - np.minimum(2.0*abs(py.dQ_mono[:,j0-1:jend+1]), abs(py.q_L[:,j0-1:jend+1]-Q[:,j0-1:jend+1])) * np.sign(2.0*py.dQ_mono[:,j0-1:jend+1])
        q0 = Q[:,j0-1:jend+1,:]
        qmin = np.minimum(2.0*abs(py.dQ_mono[:,j0-1:jend+1,:]), abs(py.q_L[:,j0-1:jend+1,:]-Q[:,j0-1:jend+1,:])) * np.sign(2.0*py.dQ_mono[:,j0-1:jend+1,:])
        py.q_L[:,j0-1:jend+1,:] = ne.evaluate('q0-qmin')

        # Formula B4 from Lin 04
        #py.q_R[:,j0-1:jend+1] = Q[:,j0-1:jend+1] + np.minimum(2.0*abs(py.dQ_mono[:,j0-1:jend+1]), abs(py.q_R[:,j0-1:jend+1]-Q[:,j0-1:jend+1])) * np.sign(2.0*py.dQ_mono[:,j0-1:jend+1])
        qmin = np.minimum(2.0*abs(py.dQ_mono[:,j0-1:jend+1,:]), abs(py.q_R[:,j0-1:jend+1,:]-Q[:,j0-1:jend+1,:])) * np.sign(2.0*py.dQ_mono[:,j0-1:jend+1,:])
        py.q_R[:,j0-1:jend+1,:] = ne.evaluate('q0+qmin')

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    #py.dq[:,j0-1:jend+1] = py.q_R[:,j0-1:jend+1] - py.q_L[:,j0-1:jend+1]
    #py.q6[:,j0-1:jend+1] = 6*Q[:,j0-1:jend+1] - 3*(py.q_R[:,j0-1:jend+1] + py.q_L[:,j0-1:jend+1])
    q_L =  py.q_L[:,j0-1:jend+1,:]
    q_R =  py.q_R[:,j0-1:jend+1,:]
    q = Q[:,j0-1:jend+1,:]
    py.dq[:,j0-1:jend+1,:] = ne.evaluate('q_R-q_L')
    py.q6[:,j0-1:jend+1,:] = ne.evaluate('6*q- 3*(q_R + q_L)')

#########################################################################
#
# This routine computes the parabolas of PPM in x and y direction
#
########################################################################
def ppm_reconstruction(Q, px, py, cs_grid, simulation):
    # Reconstruct the values at edged
    ppm_reconstruction_x(Q, px, cs_grid, simulation)
    ppm_reconstruction_y(Q, py, cs_grid, simulation)

    if simulation.edge_treatment==2:
        # Extrapolation at cells near the cube edges
        edges_extrapolation(Q, px, py, cs_grid, simulation)

    average_parabola_cube_edges(px, py, cs_grid)


