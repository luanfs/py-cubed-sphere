####################################################################################
#
# Module for advection test case set up (initial condition, exact solution and etc)
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
#
# Test cases are based in the paper "A class of deformational ﬂow test cases for linear
# transport problems  on the sphere", 2010, Ramachandran D. Nair and Peter H. Lauritzen
#
####################################################################################
import numpy as np
from constants import*
from sphgeo import sph2cart, cart2sph
from scipy.special import sph_harm

####################################################################################
# Advection simulation class
####################################################################################
class adv_simulation_par:
    def __init__(self, dt, Tf, ic, vf, tc, mono, degree):
        # Initial condition
        self.ic = ic

        # Vector field
        self.vf = vf

        # Advection test case
        self.tc = tc

        # Time step
        self.dt = dt

        # Total period definition
        self.Tf = Tf

        # Monotonization
        self.mono = mono

        # Interpolation degree
        self.degree = degree

        # Define the initial condition name
        if ic == 1:
            name = 'Constant flow'
        elif ic == 2:
            name = 'Gaussian hill'
        elif ic == 3:
            name = 'Two gaussian hills'
        else:
            print("Error - invalid initial condition")
            exit()

        # IC name
        self.icname = name

        # Define the vector field name
        if vf == 1:
            name = 'Zonal wind'
        elif vf == 2:
            name = 'Rotated zonal wind'
        elif vf == 3:
            name = 'Non divergent field 1 from Nair and Lauritzen 2010'
        elif vf == 4:
            name = 'Non divergent field 2 from Nair and Lauritzen 2010'
        elif vf == 5:
            name = 'Non divergent field 4 from Nair and Lauritzen 2010'
        elif vf == 6:
            name = 'Divergent field 3 from Nair and Lauritzen 2010'
        else:
            print("Error - invalid vector field")
            exit()

        # IC name
        self.vfname = name

        # Monotonization:
        if mono == 0:
            monot = 'none'
        elif mono == 1:
            monot = 'CW84' # Collela and Woodward 84 paper
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
    if simulation.ic == 1:
        return np.ones(np.shape(lon))

    # Gaussian hill
    elif simulation.ic == 2:
        if simulation.vf == 1 or simulation.vf == 2:
            if simulation.vf == 1:
                alpha =   0.0*deg2rad # Rotation angle
            if simulation.vf == 2:
                alpha = -45.0*deg2rad # Rotation angle
            # Wind speed
            u0 =  2.0*pi/5.0 # Wind speed
            ws = -u0
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

            lon0, lat0 = 0.0, 0.0
            X0, Y0, Z0 = sph2cart(lon0, lat0)
            b0 = 10.0
            q = np.exp(-b0*((rotX-X0)**2+ (rotY-Y0)**2 + (rotZ-Z0)**2))
        else:
            X, Y, Z = sph2cart(lon, lat)
            lon0, lat0 = 0.0, 0.0
            X0, Y0, Z0 = sph2cart(lon0, lat0)
            b0 = 10.0
            q = np.exp(-b0*((X-X0)**2+ (Y-Y0)**2 + (Z-Z0)**2))

    # Two Gaussian hills
    elif simulation.ic == 3:
        X, Y, Z = sph2cart(lon, lat)
        if simulation.vf == 1 or simulation.vf == 2 or simulation.vf == 3:
            # Gaussian hill centers
            lon1, lat1 = 0,  pi/3.0
            lon2, lat2 = 0, -pi/3.0
        elif simulation.vf == 4 or  simulation.vf == 5 or  simulation.vf == 6:
            # Gaussian hill centers
            lon1, lat1 = -pi/6.0, 0
            lon2, lat2 =  pi/6.0, 0
        X1, Y1, Z1 = sph2cart(lon1, lat1)
        X2, Y2, Z2 = sph2cart(lon2, lat2)
        b0 = 5.0
        q = np.exp(-b0*((X-X1)**2+ (Y-Y1)**2 + (Z-Z1)**2)) + np.exp(-b0*((X-X2)**2+ (Y-Y2)**2 + (Z-Z2)**2))
    else:
        print('Invalid initial condition.\n')
        exit()
    return q

####################################################################################
# Velocity field
# Return the wind components in geographical coordinates tangent vectors.
####################################################################################
def velocity_adv(lon, lat, t, simulation):
    if simulation.vf == 1 or simulation.vf == 2:
        if simulation.vf == 1:
           alpha = 0.0*deg2rad   # Rotation angle
        elif simulation.vf == 2:
           alpha = -45.0*deg2rad # Rotation angle
        u0    =  2.0*pi/5.0 # Wind speed
        ulon  =  u0*(np.cos(lat)*np.cos(alpha) + np.sin(lat)*np.cos(lon)*np.sin(alpha))
        vlat  = -u0*np.sin(lon)*np.sin(alpha)

    elif simulation.vf == 3: # Non divergent field 1 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        ulon =  k     * np.sin((lon+pi)/2.0)**2 * np.sin(2.0*lat) * np.cos(pi*t/T)
        vlat =  k/2.0 * np.sin(lon+pi)          * np.cos(lat)     * np.cos(pi*t/T)

    elif simulation.vf == 4: # Non divergent field 2 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        ulon = k*np.sin(lon+pi)**2 * np.sin(2.0*lat) * np.cos(pi*t/T)
        vlat = k*np.sin(2*(lon+pi)) * np.cos(lat) * np.cos(pi*t/T)

    elif simulation.vf == 5: # Non divergent field 4 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        lonp = lon-2*pi*t/T
        ulon = k*(np.sin((lonp+pi))**2)*(np.sin(2.*lat))*(np.cos(pi*t/T))+2.*pi*np.cos(lat)/T
        vlat = k*(np.sin(2*(lonp+pi)))*(np.cos(lat))*(np.cos(pi*t/T))

    elif simulation.vf == 6: # Divergent field 3 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 1.0
        ulon = -k*(np.sin((lon+pi)/2.0)**2)*(np.sin(2.0*lat))*(np.cos(lat)**2)*(np.cos(pi*t/T))
        vlat = (k/2.0)*(np.sin((lon+pi)))*(np.cos(lat)**3)*(np.cos(pi*t/T))

    return ulon, vlat
