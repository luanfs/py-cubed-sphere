####################################################################################
#
# Module for divergence test case
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
from sphgeo                 import latlon_to_contravariant, contravariant_to_latlon
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid
from dimension_splitting    import F_operator, G_operator
from interpolation          import ll2cs, nearest_neighbour, ghost_cells_lagrange_interpolation, lagrange_poly_ghostcells
from plot                   import plot_scalar_and_vector_field
from errors                 import compute_errors, print_errors_simul, plot_convergence_rate, plot_errors_loglog
from configuration          import get_div_parameters
from cfl                    import cfl_x, cfl_y
from stencil                import flux_ppm_x_stencil_coefficients, flux_ppm_y_stencil_coefficients

####################################################################################
# Divergence simulation class
####################################################################################
class div_simulation_par:
    def __init__(self, vf, tc, mono):
        # Vector field
        self.vf = vf

        # Test case
        self.tc = tc

        # Time step
        self.dt = 1.0

        # Monotonization
        self.mono = mono

        # Define the vector field name
        if vf == 1:
            name = 'Divergence free zonal wind'
        elif vf == 2:
            name = 'Divergence free rotated zonal wind'
        elif vf == 3:
            name = 'Trigonometric vector field'
        elif vf == 4:
            name = 'Non divergent field 1 from Nair and Lauritzen 2010'
        elif vf == 5:
            name = 'Non divergent field 2 from Nair and Lauritzen 2010'
        elif vf == 6:
            name = 'Non divergent field 4 from Nair and Lauritzen 2010'
        else:
            print("Error - invalid vector field")
            exit()

        # Vector field name
        self.vfname = name

        # Monotonization:
        if mono == 0:
            monot = 'none'
        elif mono == 1:
            monot = 'CW84' # Woodward and Collela 84 paper
        else:
           print("Error - invalid monotization method")
           exit()

        # Monotonization method
        self.monot = monot

        # Finite volume method
        self.fvmethod = 'PPM'

        # Simulation title
        if tc == 1:
            self.title = 'Divergence '
        elif tc == 2:
            self.title = 'Divergence errors '
        else:
            print("Error - invalid test case")
            exit()

####################################################################################
# Vector field
# Return the wind components in geographical coordinates tangent vectors.
####################################################################################
def vector_field(lon, lat, t, simulation):
    if simulation.vf == 1 or simulation.vf == 2:
        if simulation.vf == 1:
            alpha = 0.0*deg2rad # Rotation angle
        else:
            alpha = -55.0*deg2rad # Rotation angle
        u0 = 2.0*pi/5.0 # Wind speed
        ulon  =  u0*(np.cos(lat)*np.cos(alpha) + np.sin(lat)*np.cos(lon)*np.sin(alpha))
        vlat  = -u0*np.sin(lon)*np.sin(alpha)

    elif simulation.vf == 3: # Trigonometric field
        m = 1
        n = 1
        ulon = -m*(np.sin(lon)*np.sin(m*lon)*np.cos(n*lat)**3)#/np.cos(lat)
        vlat = -4*n*(np.cos(n*lat)**3)*np.sin(n*lat)*np.cos(m*lon)*np.sin(lon)

    elif simulation.vf == 4: # Non divergent field 1 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        ulon =  k     * np.sin((lon+pi)/2.0)**2 * np.sin(2.0*lat) * np.cos(pi*t/T)
        vlat =  k/2.0 * np.sin(lon+pi)          * np.cos(lat)     * np.cos(pi*t/T)

    elif simulation.vf == 5: # Non divergent field 2 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        ulon = k*np.sin(lon+pi)**2 * np.sin(2.0*lat) * np.cos(pi*t/T)
        vlat = k*np.sin(2*(lon+pi)) * np.cos(lat) * np.cos(pi*t/T)

    elif simulation.vf == 6: # Non divergent field 4 from Nair and Lauritzen 2010
        T = 5.0 # Period
        k = 2.0
        lonp = lon-2*pi*t/T
        ulon = k*(np.sin((lonp+pi))**2)*(np.sin(2.*lat))*(np.cos(pi*t/T))+2.*pi*np.cos(lat)/T
        vlat = k*(np.sin(2*(lonp+pi)))*(np.cos(lat))*(np.cos(pi*t/T))
    return ulon, vlat

####################################################################################
# Return the wind components in geographical coordinates tangent vectors.
####################################################################################
def div(lon, lat, simulation):
    if simulation.vf != 3:
        div = np.zeros(np.shape(lon))
    else:
        m = 1
        n = 1
        div = (-np.cos(lon) * np.sin(m * lon) * m * np.cos(n * lat) ** 4 / np.cos(lat) - \
            np.sin(lon) * np.cos(m * lon) * m ** 2 * np.cos(n * lat) ** 4 / np.cos(lat) + \
            12.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 2 * np.sin(n *lat) ** 2 * n ** 2 * np.cos(lat) - \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 4 * n ** 2 * np.cos(lat) + \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 3 * np.sin(n * lat) * n * np.sin(lat)) / np.cos(lat)
    return div

####################################################################################
# Compute the divergence of the chosen velocity field using the
# dimension splitting scheme
####################################################################################
def div_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, plot):
    N  = cs_grid.N       # Number of cells in x direction
    vf = simulation.vf   # Funcion to have divergence computed
    dx = cs_grid.dx      # Grid spacing
    dy = cs_grid.dy      # Grid spacing
    ng = cs_grid.nghost  # Number of ghost cells
    tc = simulation.tc
    vfname = simulation.vfname
    mono   = simulation.mono  # Monotonization scheme
    nghost = cs_grid.nghost   # Number o ghost cells

    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Initialize the variables
    div_exact, div_numerical, flux_x, flux_y, ax, ay, cx, cy, cx2, cy2, \
    F_gQ, G_gQ, GF_gQ, FG_gQ, F_gQ_exact, G_gQ_exact, \
    ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy, \
    ulon_edx, vlat_edx, ulon_edy, vlat_edy, gQ, \
    lagrange_poly, lagrange_poly_flipped, Kmin, Kmax, interpol_method = init_vars_div(cs_grid, simulation, transformation, N, nghost) 

    # Applies F and G operators in each panel
    F_operator(F_gQ, gQ, ucontra_edx, flux_x, ax, cs_grid, simulation)
    G_operator(G_gQ, gQ, vcontra_edy, flux_y, ay, cs_grid, simulation)

    # Interpolate the F_operator at ghost cells  
    ghost_cells_lagrange_interpolation(F_gQ, cs_grid, transformation, simulation, interpol_method, lagrange_poly, lagrange_poly_flipped, Kmin, Kmax)

    # Interpolate the G_operator at ghost cells    
    ghost_cells_lagrange_interpolation(G_gQ, cs_grid, transformation, simulation, interpol_method, lagrange_poly, lagrange_poly_flipped, Kmin, Kmax)

    # Applies F and G operators in each panel again
    F_operator(FG_gQ, gQ+0.5*G_gQ, ucontra_edx, flux_x, ax, cs_grid, simulation)
    G_operator(GF_gQ, gQ+0.5*F_gQ, vcontra_edy, flux_y, ay, cs_grid, simulation)

    div_numerical.f = -(FG_gQ[i0:iend,j0:jend] + GF_gQ[i0:iend,j0:jend])/simulation.dt

    # Exact operators
    #F_gQ_exact[:,:,:] = (dy/cs_grid.areas[:,:,:])*(ucontra_edx[1:N+ng+1,:,:]*cs_grid.metric_tensor_edx[1:N+ng+1,:,:] - ucontra_edx[0:N+ng,:,:]*cs_grid.metric_tensor_edx[0:N+ng,:,:])
    #G_gQ_exact[:,:,:] = (dx/cs_grid.areas[:,:,:])*(vcontra_edy[:,1:N+ng+1,:]*cs_grid.metric_tensor_edy[:,1:N+ng+1,:] - vcontra_edy[:,0:N+ng,:]*cs_grid.metric_tensor_edy[:,0:N+ng,:])
    #FG_gQ_exact = G_gQ_exact + F_gQ_exact
    #div_numerical.f = FG_gQ_exact[i0:iend,j0:jend]

    error_linf, error_l1, error_l2 = output_div(cs_grid, ll_grid, plot, div_exact, div_numerical, \
                                                ulon_edx, vlat_edx, ulon_edy, vlat_edy, simulation, map_projection)

    return error_linf, error_l1, error_l2

####################################################################################
# Output for the divergence routine
####################################################################################
def output_div(cs_grid, ll_grid, plot, div_exact, div_numerical, \
               ulon_edx, vlat_edx, ulon_edy, vlat_edy, simulation, map_projection):
    if plot:
        # Interpolate to the latlon grid and plot
        # div exact
        div_exact_ll = nearest_neighbour(div_exact, cs_grid, ll_grid)
        colormap = 'jet'
        #qmin = np.amin(div_exact_ll)
        #qmax = np.amax(div_exact_ll)
        #plot_scalar_and_vector_field(div_exact_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, 'div_exact_vf'+str(simulation.vf)+"_mono_"+simulation.monot+"_"+simulation.fvmethod, cs_grid, ll_grid, map_projection, colormap, qmin, qmax)

        # numerical div
        div_ll = nearest_neighbour(div_numerical, cs_grid, ll_grid)
        colormap = 'jet'
        #qmin = -5.0*10**(-6)
        #qmax =  5.0*10**(-6)
        qmin = np.amin(div_ll)
        qmax = np.amax(div_ll)
        plot_scalar_and_vector_field(div_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, 'div_vf'+str(simulation.vf)+"_mono_"+simulation.monot+"_"+simulation.fvmethod, cs_grid, ll_grid, map_projection, colormap, qmin, qmax)

        # div error
        div_error_ll =  div_exact_ll - div_ll
        colormap = 'seismic'
        qmin = np.amin(div_error_ll)
        qmax = np.amax(div_error_ll)
        #qmin = -7.6*10**(-5)
        #qmax =  7.6*10**(-5)
        plot_scalar_and_vector_field(div_error_ll, ulon_edx, vlat_edx, ulon_edy, vlat_edy, 'div_error'+'_vf'+str(simulation.vf)+"_mono_"+simulation.monot+"_"+simulation.fvmethod, cs_grid, ll_grid, map_projection, colormap, qmin, qmax)

    # Compute the errors
    error_linf, error_l1, error_l2 = compute_errors(div_exact.f, div_numerical.f)
    return error_linf, error_l1, error_l2

####################################################################################
# This routine initializates the divergence routine variables
####################################################################################
def init_vars_div(cs_grid, simulation, transformation, N, nghost):
    # Interior cells index (ignoring ghost cells)
    i0   = cs_grid.i0
    iend = cs_grid.iend
    j0   = cs_grid.j0
    jend = cs_grid.jend

    # Get edges position in lat/lon system
    edx_lon = cs_grid.edx.lon
    edx_lat = cs_grid.edx.lat
    edy_lon = cs_grid.edy.lon
    edy_lat = cs_grid.edy.lat

    # Get center position in lat/lon system
    center_lon = cs_grid.centers.lon
    center_lat = cs_grid.centers.lat

    # Metric tensor
    g_metric = cs_grid.metric_tensor_centers

    # Velocity (latlon) at edges
    ulon_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    vlat_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    ulon_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))
    vlat_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))

    # Velocity (contravariant) at edges
    ucontra_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    vcontra_edx = np.zeros((N+nghost+1, N+nghost  , nbfaces))
    ucontra_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))
    vcontra_edy = np.zeros((N+nghost  , N+nghost+1, nbfaces))

    # Get velocities
    ulon_edx[:,:,:], vlat_edx[:,:,:] = vector_field(edx_lon, edx_lat, 0.0, simulation)
    ulon_edy[:,:,:], vlat_edy[:,:,:] = vector_field(edy_lon, edy_lat, 0.0, simulation)

    # Convert latlon to contravariant at ed_x
    ex_elon_edx = cs_grid.prod_ex_elon_edx
    ex_elat_edx = cs_grid.prod_ex_elat_edx
    ey_elon_edx = cs_grid.prod_ey_elon_edx
    ey_elat_edx = cs_grid.prod_ey_elat_edx
    det_edx    = cs_grid.determinant_ll2contra_edx
    ucontra_edx, vcontra_edx = latlon_to_contravariant(ulon_edx, vlat_edx, ex_elon_edx, ex_elat_edx, ey_elon_edx, ey_elat_edx, det_edx)

    # Convert latlon to contravariant at ed_y
    ex_elon_edy = cs_grid.prod_ex_elon_edy
    ex_elat_edy = cs_grid.prod_ex_elat_edy
    ey_elon_edy = cs_grid.prod_ey_elon_edy
    ey_elat_edy = cs_grid.prod_ey_elat_edy
    det_edy     = cs_grid.determinant_ll2contra_edy
    ucontra_edy, vcontra_edy = latlon_to_contravariant(ulon_edy, vlat_edy, ex_elon_edy, ex_elat_edy, ey_elon_edy, ey_elat_edy, det_edy)

    # Exact divergence
    div_exact = scalar_field(cs_grid, 'div_exact', 'center')
    div_exact.f = div(center_lon[i0:iend,j0:jend], center_lat[i0:iend,j0:jend], simulation)

    # Numerical divergence
    div_numerical = scalar_field(cs_grid, 'div_numerical', 'center')

    # Dimension splitting operators
    F_gQ  = np.zeros((N+nghost, N+nghost, nbfaces))
    G_gQ  = np.zeros((N+nghost, N+nghost, nbfaces))
    GF_gQ = np.zeros((N+nghost, N+nghost, nbfaces))
    FG_gQ = np.zeros((N+nghost, N+nghost, nbfaces))

    F_gQ_exact   = np.zeros((N+nghost, N+nghost, nbfaces))
    G_gQ_exact   = np.zeros((N+nghost, N+nghost, nbfaces))

    # Scalar field
    Q = np.ones((N+nghost, N+nghost, nbfaces))

    # Get Lagrange polynomials
    interpol_method = 3
    lagrange_poly, lagrange_poly_flipped, Kmin, Kmax = lagrange_poly_ghostcells(cs_grid, transformation, interpol_method)

    # Interpolate the scalar field    
    ghost_cells_lagrange_interpolation(Q, cs_grid, transformation, simulation, interpol_method, lagrange_poly, lagrange_poly_flipped, Kmin, Kmax)

    # Multiply the field Q (=1) by metric tensor
    gQ = g_metric*Q

    # Time step for CFL = 0.5
    simulation.dt = 0.5*cs_grid.dx/np.amax(abs(ucontra_edx))

    # CFL at edges - x direction
    cx, cx2 = cfl_x(ucontra_edx, cs_grid, simulation)

    # CFL at edges - y direction
    cy, cy2 = cfl_y(vcontra_edy, cs_grid, simulation)

    # Flux at edges
    flux_x = np.zeros((N+7, N+6, nbfaces))
    flux_y = np.zeros((N+6, N+7, nbfaces))

    # Stencil coefficients
    ax = np.zeros((6, N+7, N+6, nbfaces))
    ay = np.zeros((6, N+6, N+7, nbfaces))

    # Compute the coefficients
    flux_ppm_x_stencil_coefficients(ucontra_edx, ax, cx, cx2, simulation)
    flux_ppm_y_stencil_coefficients(vcontra_edy, ay, cy, cy2, simulation)

    return div_exact, div_numerical, flux_x, flux_y, ax, ay, cx, cy, cx2, cy2, \
           F_gQ, G_gQ, GF_gQ, FG_gQ, F_gQ_exact, G_gQ_exact, \
           ucontra_edx, vcontra_edx, ucontra_edy, vcontra_edy,\
           ulon_edx, vlat_edx, ulon_edy, vlat_edy, gQ, \
           lagrange_poly, lagrange_poly_flipped, Kmin, Kmax, interpol_method

###################################################################################
# Routine to compute the divergence error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_div(simulation, map_projection, plot, transformation, showonscreen, gridload):
    # Initial condition
    vf = simulation.vf

    # Monotonization method
    mono = simulation.mono

    # Test case
    tc = simulation.tc

    # Number of tests
    Ntest = 6

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 20

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2

    # Errors array
    error_linf = np.zeros(Ntest)
    error_l1   = np.zeros(Ntest)
    error_l2   = np.zeros(Ntest)

    # Let us test and compute the error!
    tc, vf, mono = get_div_parameters()

    for i in range(0, Ntest):
        simulation = div_simulation_par(vf, tc, mono)
        N = int(Nc[i])

        # Create CS mesh
        cs_grid = cubed_sphere(N, transformation, False, gridload)

        # Create the latlon mesh (for plotting)
        ll_grid = latlon_grid(Nlat, Nlon)
        ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

        # Get divergence error
        error_linf[i], error_l1[i], error_l2[i] = div_sphere(cs_grid, ll_grid, simulation, map_projection, transformation, True)

        # Print errors
        print_errors_simul(error_linf, error_l1, error_l2, i)

    # Outputs
    # Convergence rate
    title = "Convergence rate - divergence operator"
    filename = graphdir+"div_tc"+str(tc)+"_vf"+str(vf)+"_cr_rate_"+transformation+"_mono_"+simulation.monot+"_"+simulation.fvmethod
    plot_convergence_rate(Nc, error_linf, error_l1, error_l2, filename, title)

    # Error convergence
    title = "Convergence of error  - divergence operator"
    filename = graphdir+"div_tc"+str(tc)+"_vf"+str(vf)+"_error_convergence_"+transformation+"_mono_"+simulation.monot+"_"+simulation.fvmethod
    plot_errors_loglog(Nc, error_linf, error_l1, error_l2, filename, title)
