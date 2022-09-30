####################################################################################
#
# Module for advection test case set up (initial condition, exact solution and etc)
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np
from constants import*
from sphgeo import sph2cart, cart2sph
from scipy.special import sph_harm

####################################################################################
# Advection simulation class
####################################################################################
class adv_simulation_par:
    def __init__(self, dt, Tf, ic, tc, mono):
        # Initial condition
        self.ic = ic

        # Advection test case
        self.tc = tc

        # Time step
        self.dt = dt

        # Total period definition
        self.Tf = Tf

        # Monotonization
        self.mono = mono

        # Define the initial condition name
        if ic == 1:
            name = 'Constant flow'
        elif ic == 2:
            name = 'Constant flow'
        elif ic == 3:
            name = 'Gaussian hill'
        elif ic == 4:
            name = 'Gaussian hill'
        else:
            print("Error - invalid initial condition")
            exit()

        # IC name
        self.icname = name

        # Monotonization:
        if mono == 0:
            monot = 'none'
        elif mono == 1:
            monot = 'WC84' # Woodward and Collela 84 paper
        else:
           print("Error - invalid monotization method")
           exit()

        # Monotonization method
        self.monot = monot

        # Finite volume method
        self.fvmethod = 'PPM'

        # Simulation title
        if tc == 1:
            self.title = '2D Advection '
        elif tc == 2:
            self.title = '2D advection errors '
        else:
            print("Error - invalid test case")
            exit()

####################################################################################
# Initial condition
####################################################################################
def q0_adv(lon, lat, simulation):
    q0 = qexact_adv(lon, lat, 0, simulation)
    return q0

####################################################################################
# Exact solution to the advection problem
####################################################################################
def qexact_adv(lon, lat, t, simulation):
    # Constant flow
    if simulation.ic == 1 or simulation.ic == 2:
        return np.ones(np.shape(lon))

    # Gaussian hill
    elif simulation.ic == 3 or simulation.ic == 4:
        if simulation.ic == 3:
            alpha =   0.0*deg2rad # Rotation angle
        if simulation.ic == 4:
            alpha = -55.0*deg2rad # Rotation angle
        # Wind speed
        u0 =  2.0*erad*pi/(12.0*day2sec) # Wind speed
        ws = -u0/erad
        wt = ws*t

        #Rotation parameters
        cosa  = np.cos(alpha)
        cos2a = cosa*cosa
        sina  = np.sin(alpha)
        sin2a = sina*sina
        coswt = np.cos(wt)
        sinwt = np.sin(wt)

        X, Y, Z = sph2cart(lon, lat)

        rotX = (coswt*cos2a+sin2a)*X -sinwt*cosa*Y + (coswt*cosa*sina-cosa*sina)*Z
        rotY =  sinwt*cosa*X + coswt*Y + sina*sinwt*Z
        rotZ = (coswt*sina*cosa-sina*cosa)*X -sinwt*sina*Y + (coswt*sin2a+cos2a)*Z 

        # Gaussian center
        if simulation.ic == 3 or simulation.ic == 4:
            lon0, lat0 = 0.0, 0.0
            X0, Y0, Z0 = sph2cart(lon0, lat0)
            b0 = 10.0
            q = np.exp(-b0*((rotX-X0)**2+ (rotY-Y0)**2 + (rotZ-Z0)**2))
        # Spherical harmonic
        if simulation.ic == 5:
            m = 4
            n = 10
            lon_rot, lat_rot = cart2sph(rotX, rotY, rotZ)
            Ymn = sph_harm(m, n, lon_rot+pi, lat_rot+pi)
            q = Ymn.real
    else:
        print('Invalid initial condition.\n')
        exit()
    return q

####################################################################################
# Velocity field
# Return the wind components in geographical coordinates tangent vectors.
####################################################################################
def velocity_adv(lon, lat, t, simulation):
    if simulation.ic >=1 and simulation.ic <= 4:
        if simulation.ic == 1 or simulation.ic == 3:
           alpha = 0.0*deg2rad # Rotation angle
        elif simulation.ic == 2 or simulation.ic == 4:
           alpha = -55.0*deg2rad # Rotation angle
        u0    =  2.0*erad*pi/(12.0*day2sec) # Wind speed
        ulon  =  u0*(np.cos(lat)*np.cos(alpha) + np.sin(lat)*np.cos(lon)*np.sin(alpha))
        vlat  = -u0*np.sin(lon)*np.sin(alpha)
    elif simulation.ic == 5:
        print('Dunno what to do!')
        exit()
    return ulon, vlat
