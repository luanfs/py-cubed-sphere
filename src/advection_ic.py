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
import numexpr as ne
from constants import*
from sphgeo import sph2cart, cart2sph
from cs_datastruct import ppm_parabola, velocity

####################################################################################
# Advection simulation class
####################################################################################
class adv_simulation_par:
    def __init__(self, cs_grid, dt, Tf, ic, vf, tc, recon, dp, opsplit, et):
        # Initial condition
        self.ic = ic

        # Vector field
        self.vf = vf

        # Test case
        self.tc = tc

        # Time step
        self.dt = dt
        self.dto2  = dt*0.5
        self.twodt = dt*2.0

        # Flux method
        self.recon = recon

        # Departure point scheme
        self.dp = dp

        # Splitting method
        self.opsplit= opsplit

        # Total period definition
        self.Tf = Tf

        # Degree for interpolation
        self.degree = 3

        # Define the initial condition name
        if ic == 1:
            name = 'Constant flow'
        elif ic == 2:
            name = 'Gaussian hill'
        elif ic == 3:
            name = 'Two gaussian hills'
        else:
            print("Error in adv_simulation_par - invalid initial condition")
            exit()

        # Define the vector field name
        if vf == 1:
            name = 'Divergence free zonal wind'
        elif vf == 2:
            name = 'Divergence free rotated zonal wind'
        elif vf == 3:
            name = 'Non divergent field 2 from Nair and Lauritzen 2010'
        elif vf == 4:
            name = 'Non divergent field 4 from Nair and Lauritzen 2010'
        elif vf == 5:
            name = 'Divergent field from Nair and Lauritzen 2010'
        elif vf == 6:
            name = 'Trigonometric vector field'
        else:
            print("Error in div_simulation_par - invalid vector field")
            exit()

        # IC name
        self.icname = name

        # Vector field name
        self.vfname = name

        # Flux scheme
        if recon == 1:
            recon_name = 'PPM-0'
        elif recon == 2:
            recon_name = 'PPM-CW84' #Monotonization from Collela and Woodward 84 paper
        elif recon == 3:
            recon_name = 'PPM-PL07' # Hybrid PPM from PL07 paper
        elif recon == 4:
            recon_name = 'PPM-L04' #Monotonization from Lin 04 paper
        else:
           print("Error in adv_simulation_par - invalid reconstruction method")
           exit()

        # Departure point scheme
        if dp == 1:
            dp_name = 'RK1'
        elif dp == 2:
            dp_name = 'RK2'
        else:
           print("Error in simulation_adv_par_1d - invalid departure point scheme", dp)
           exit()

        # Operator splitting scheme
        if opsplit == 1:
            opsplit_name = 'SP-AVLT' # Average Lie-Trotter splitting
        elif opsplit == 2:
            opsplit_name = 'SP-L04'  # Splitting from L04 paper
        elif opsplit == 3:
            opsplit_name = 'SP-PL07' #Splitting from Putman and Lin 07 paper
        else:
           print("Error in adv_simulation_par - invalid operator splitting method")
           exit()

        # Edges treatment
        if et==1:
            self.et_name='ET-S72'
        elif et==2:
            self.et_name='ET-PL07'
        elif et==3:
            self.et_name='ET-Z21'
        elif et==4:
            self.et_name='ET-Z21-AF'
        elif et==5:
            self.et_name='ET-Z21-PR'
        else:
            print('ERROR in recon_simulation_par: invalid ET')
            exit()
        self.edge_treatment = et

        # Splitting name
        self.opsplit_name = opsplit_name

        # Flux method name
        self.recon_name = recon_name

        # Departure point method
        self.dp_name = dp_name

        # Simulation title
        if tc == 1:
            self.title = '2D Advection '
        elif tc == 2:
            self.title = '2D advection errors '
        else:
            print("Error in adv_simulation_par- invalid test case")
            exit()

        # Variable of the advection model
        N = cs_grid.N
        ng = cs_grid.ng
     
        # PPM parabolas
        self.px = None
        self.py = None

        # average values of Q (initial condition) at cell pc
        self.Q = np.zeros((N+ng, N+ng, nbfaces))
        self.gQ = np.zeros((N+ng, N+ng, nbfaces))

        # Numerical divergence
        self.div = np.zeros((N+ng, N+ng, nbfaces))

        # Velocity at edges
        self.U_pu = velocity(cs_grid, 'pu')
        self.U_pv = velocity(cs_grid, 'pv')
        self.U_pc = velocity(cs_grid, 'pc')
     
        # CFL
        self.cx = np.zeros((N+ng+1, N+ng, nbfaces))
        self.cy = np.zeros((N+ng, N+ng+1, nbfaces))
        self.CFL = 0.0

        # Mass
        self.total_mass0 = 0.0
        self.total_mass  = 0.0
        self.mass_change = 0.0

        # Errors
        self.error_linf, self.error_l1, self.error_l2 = None, None, None

        # Lagrange polynomials
        self.lagrange_poly_edge, self.stencil_edge = None, None 
        self.lagrange_poly_ghost_pc, self.stencil_ghost_pc =  None, None 
        self.lagrange_poly_ghost_edge, self.stencil_ghost_edge = None, None 

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

            if simulation.vf == 1:
                lon0, lat0 = 0.0, 0.0
            elif simulation.vf == 2:
                lon0, lat0 = np.pi/4.0, np.pi/6.0
            X0, Y0, Z0 = sph2cart(lon0, lat0)
            b0 = 10.0
            q = np.exp(-b0*((rotX-X0)**2+ (rotY-Y0)**2 + (rotZ-Z0)**2))
            #q = np.exp(-b0*((X-X0)**2+ (Y-Y0)**2 + (Z-Z0)**2))
        else:
            X, Y, Z = sph2cart(lon, lat)
            lon0, lat0 = 0.0, 0.0
            X0, Y0, Z0 = sph2cart(lon0, lat0)
            b0 = 10.0
            q = np.exp(-b0*((X-X0)**2+ (Y-Y0)**2 + (Z-Z0)**2))

    # Two Gaussian hills
    elif simulation.ic == 3:
        X, Y, Z = sph2cart(lon, lat)
        if simulation.vf == 1 or simulation.vf == 2:
            # Gaussian hill centers
            lon1, lat1 = 0,  pi/3.0
            lon2, lat2 = 0, -pi/3.0
        elif simulation.vf >= 3:
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
# Vector field
# Return the wind components in geographical coordinates tangent vectors.
####################################################################################
def velocity_adv(lon, lat, t, simulation):
    if simulation.vf == 1 or simulation.vf == 2:
        if simulation.vf == 1:
            alpha = 0.0*deg2rad # Rotation angle
        else:
            alpha = -45.0*deg2rad # Rotation angle
        u0 = 2.0*pi/5.0 # Wind speed
        ulon  = ne.evaluate("u0*(cos(lat)*cos(alpha) + sin(lat)*cos(lon)*sin(alpha))")
        vlat  = ne.evaluate("-u0*sin(lon)*sin(alpha)")

    elif simulation.vf == 3: # Non divergent field 2 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        ulon = ne.evaluate("k*sin(lon+pi)**2 * sin(2.0*lat) * cos(pi*t/T)")
        vlat = ne.evaluate("k*sin(2*(lon+pi)) * cos(lat) * cos(pi*t/T)")

    elif simulation.vf == 4: # Non divergent field 4 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        lonp = ne.evaluate("lon-2*pi*t/T")
        ulon = ne.evaluate("k*(sin((lonp+pi))**2)*(sin(2.*lat))*(cos(pi*t/T))+2.*pi*cos(lat)/T")
        vlat = ne.evaluate("k*(sin(2*(lonp+pi)))*(cos(lat))*(cos(pi*t/T))")

    elif simulation.vf == 5: # Divergent field 3 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 1.0
        ulon = ne.evaluate("-k*(sin((lon+pi)/2.0)**2)*(sin(2.0*lat))*(cos(lat)**2)*(cos(pi*t/T))")
        vlat = ne.evaluate("(k/2.0)*(sin((lon+pi)))*(cos(lat)**3)*(cos(pi*t/T))")

    elif simulation.vf == 6: # Trigonometric field
        m = 1
        n = 1
        ulon = ne.evaluate("-m*(sin(lon)*sin(m*lon)*cos(n*lat)**3)")#/np.cos(lat)
        vlat = ne.evaluate("-4*n*(cos(n*lat)**3)*sin(n*lat)*cos(m*lon)*sin(lon)")

    return ulon, vlat

####################################################################################
# Return the divergence of the velocity fields
####################################################################################
def div_exact(lon, lat, simulation):
    if simulation.vf <= 5:
        div = np.zeros(np.shape(lon))
    else: # Trigonometric field
        m = 1
        n = 1
        div = (-np.cos(lon) * np.sin(m * lon) * m * np.cos(n * lat) ** 4 / np.cos(lat) - \
            np.sin(lon) * np.cos(m * lon) * m ** 2 * np.cos(n * lat) ** 4 / np.cos(lat) + \
            12.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 2 * np.sin(n *lat) ** 2 * n ** 2 * np.cos(lat) - \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 4 * n ** 2 * np.cos(lat) + \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 3 * np.sin(n * lat) * n * np.sin(lat)) / np.cos(lat)
    return div
