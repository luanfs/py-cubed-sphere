####################################################################################
#
# Module for interpolation test case
#
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np
from constants import*
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon, sph2cart, cart2sph
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid
from plot                   import plot_scalar_field
from errors                 import compute_errors, print_errors_simul, plot_convergence_rate, plot_errors_loglog
from configuration          import get_interpolation_parameters
from scipy.special          import sph_harm
from interpolation          import ll2cs, nearest_neighbour, ghost_cells_lagrange_interpolation
from lagrange               import lagrange_poly_ghostcells
from cs_transform           import metric_tensor, inverse_equiangular_gnomonic_map

####################################################################################
# Interpolation simulation class
####################################################################################
class interpolation_simulation_par:
    def __init__(self, ic, degree):
        # Scalar field
        self.ic = ic

        # Define the scalar field name
        if ic == 1 or ic == 2:
            name = 'Spherical harmonic'
        elif ic == 3:
            name = 'Metric tensor'
        else:
            print("Error - invalid scalar field")
            exit()

        # Order
        self.degree = degree

        # IC name
        self.icname = name

        self.title = 'Interpolation'

####################################################################################
# Scalar field to be interpolated
####################################################################################
def q_scalar_field(lon, lat, simulation):
    # Spherical harmonic
    if simulation.ic == 1 or simulation.ic == 2:
        if simulation.ic == 1:
            alpha =   0.0*deg2rad # Rotation angle
        if simulation.ic == 2:
            alpha = -55.0*deg2rad # Rotation angle

        # Wind speed
        u0 =  2.0*erad*pi/(12.0*day2sec) # Wind speed
        ws = -u0/erad
        wt = ws*80000

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

        m = 4
        n = 10
        lon_rot, lat_rot = cart2sph(rotX, rotY, rotZ)
        Ymn = sph_harm(m, n, lon_rot+pi, lat_rot+pi)
        q = Ymn.real
    elif simulation.ic == 3: # Metric tensor
        R = erad
        #X, Y, Z = sph2cart(lon, lat)
        #x, y = np.zeros(np.shape(X)), np.zeros(np.shape(X))
        #for p in range(0, nbfaces):
        #    x[:,:,p], y[:,:,p] = inverse_equiangular_gnomonic_map(X[:,:,p],Y[:,:,p],Z[:,:,p],p)
        #tanx, tany = np.tan(x), np.tan(y)
        #r = np.sqrt(1 + tanx*tanx + tany*tany)
        #G = 1.0/r**3
        #G = G/(np.cos(x)*np.cos(y))**2
        #print(np.shape(G))
        m = 1
        n = 1
        q = (-np.cos(lon) * np.sin(m * lon) * m * np.cos(n * lat) ** 4 / np.cos(lat) - \
            np.sin(lon) * np.cos(m * lon) * m ** 2 * np.cos(n * lat) ** 4 / np.cos(lat) + \
            12.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 2 * np.sin(n *lat) ** 2 * n ** 2 * np.cos(lat) - \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 4 * n ** 2 * np.cos(lat) + \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 3 * np.sin(n * lat) * n * np.sin(lat)) / np.cos(lat)
        q = q/R
    return q

###################################################################################
# Routine to compute the interpolation error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_interpolation(simulation, map_projection, transformation, showonscreen, gridload):
    # Initial condition
    ic = simulation.ic

    # Order
    degree = simulation.degree

    # Monotonization method
    #mono = simulation.mono

    # Number of tests
    Ntest = 7

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 10

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Let us test and compute the error!
    ic, degree = get_interpolation_parameters()

    for i in range(0, Ntest):
        simulation = interpolation_simulation_par(ic, degree)
        N = int(Nc[i])

        # Create CS mesh
        cs_grid = cubed_sphere(N, transformation, False, gridload)

        # Interior cells index (ignoring ghost cells)
        i0   = cs_grid.i0
        iend = cs_grid.iend
        j0   = cs_grid.j0
        jend = cs_grid.jend
        ngl = cs_grid.nghost_left
        ngr = cs_grid.nghost_right

        # Create the latlon mesh (for plotting)
        ll_grid = latlon_grid(Nlat, Nlon)
        ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

        # Exact field
        Q_exact = q_scalar_field(cs_grid.centers.lon, cs_grid.centers.lat, simulation)
        Qexact = scalar_field(cs_grid, 'Q', 'center')
        Qexact.f = Q_exact[i0:iend,j0:jend,:]

        # Interpolated field
        Q_numerical = np.zeros(np.shape(Q_exact))
        Q_numerical[i0:iend,j0:jend,:] = Q_exact[i0:iend,j0:jend,:]

        Q_ll = nearest_neighbour(Qexact, cs_grid, ll_grid)
        name = 'interp_q_ic_'+str(simulation.ic)
        plot_scalar_field(Q_ll, name, cs_grid, ll_grid, map_projection)

        # Get Lagrange polynomials
        lagrange_poly, Kmin, Kmax = lagrange_poly_ghostcells(cs_grid, simulation, transformation)

        # Interpolate to ghost cells
        ghost_cells_lagrange_interpolation(Q_numerical, cs_grid, transformation, simulation, lagrange_poly, Kmin, Kmax)

        # Compute the errors
        error_linf[i], error_l1[i], error_l2[i] = compute_errors(Q_numerical, Q_exact)

        # Print errors
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Outputs
    # Convergence rate
    title = "Convergence rate - interpolation"
    filename = graphdir+"interpolation_ic"+str(ic)+"_cr_rate_"+transformation
    plot_convergence_rate(Nc, error_linf, error_l1, error_l2, filename, title)

    # Error convergence
    title = "Convergence of error - interpolation"
    filename = graphdir+"interpolation_ic"+str(ic)+"_error_convergence_"+transformation
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)
