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
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid, ppm_parabola
from plot                   import plot_scalar_field
from errors                 import compute_errors, print_errors_simul, plot_convergence_rate, plot_errors_loglog
from configuration          import get_interpolation_parameters
from scipy.special          import sph_harm
from interpolation          import ll2cs, nearest_neighbour, ghost_cells_lagrange_interpolation, ghost_cells_adjacent_panels
from lagrange               import lagrange_poly_ghostcells
from cs_transform           import metric_tensor, inverse_equiangular_gnomonic_map
from reconstruction_1d      import ppm_reconstruction

####################################################################################
# Interpolation simulation class
####################################################################################
class interpolation_simulation_par:
    def __init__(self, ic, degree):
        # Scalar field
        self.ic = ic

        # Define the scalar field name
        if ic == 1:
            name = 'Gaussian hill'
        elif ic == 2:
            name = 'Trigonometric field'
        else:
            print("Error - invalid scalar field")
            exit()

        # IC name
        self.icname = name

        # degree
        self.degree = degree

        # Simulation title
        self.title = 'Interpolation'

####################################################################################
# Scalar field to be interpolated
####################################################################################
def q_scalar_field(lon, lat, simulation):
    # Spherical harmonic
    if simulation.ic == 1:
        lon0, lat0 = np.pi/4.0, np.pi/6.0
        X0, Y0, Z0 = sph2cart(lon0, lat0)
        X , Y , Z  = sph2cart(lon , lat )
        b0 = 10.0
        q = np.exp(-b0*((X-X0)**2+ (Y-Y0)**2 + (Z-Z0)**2))
    elif simulation.ic == 2: # Trigonometric field
        m = 1
        n = 1
        q = (-np.cos(lon) * np.sin(m * lon) * m * np.cos(n * lat) ** 4 / np.cos(lat) - \
            np.sin(lon) * np.cos(m * lon) * m ** 2 * np.cos(n * lat) ** 4 / np.cos(lat) + \
            12.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 2 * np.sin(n *lat) ** 2 * n ** 2 * np.cos(lat) - \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 4 * n ** 2 * np.cos(lat) + \
            4.0 * np.sin(lon) * np.cos(m * lon) * np.cos(n * lat) ** 3 * np.sin(n * lat) * n * np.sin(lat)) / np.cos(lat)
    return q

###################################################################################
# Routine to compute the interpolation error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_interpolation(map_projection, transformation, showonscreen, gridload):
    # Number of tests
    Ntest = 7

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 16

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2

    # Errors array
    degrees = (0,1,2,3,4)
    error_linf = np.zeros((Ntest, len(degrees)))
    error_l1   = np.zeros((Ntest, len(degrees)))
    error_l2   = np.zeros((Ntest, len(degrees)))

    # Let us test and compute the error!
    ic = get_interpolation_parameters()


    d = 0
    eold=1
    for degree in degrees:
        for i in range(0, Ntest):
            simulation = interpolation_simulation_par(ic, degree)
            N = int(Nc[i])
            print('\nParameters: N = '+str(int(Nc[i]))+', degree = '+str(degree))

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
            if degree == 0:
                plot_scalar_field(Q_ll, name, cs_grid, ll_grid, map_projection)

            # Get Lagrange polynomials
            lagrange_poly, Kmin, Kmax = lagrange_poly_ghostcells(cs_grid, simulation, transformation)

            # Interpolate to ghost cells
            ghost_cells_lagrange_interpolation(Q_numerical, cs_grid, transformation, simulation, lagrange_poly, Kmin, Kmax)

            # Compute the errors
            error_linf[i,d], error_l1[i,d], error_l2[i,d] = compute_errors(Q_numerical, Q_exact)

            # Print errors
            print_errors_simul(error_linf[:,d], error_l1[:,d], error_l2[:,d], i)

        d = d+1
        print()

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']
    e = 0
    for error in error_list:
        emin, emax = np.amin(error[:]), np.amax(error[:])

        # convergence rate min/max
        n = len(error)
        CR = np.abs(np.log(error[1:n])-np.log(error[0:n-1]))/np.log(2.0)
        CRmin, CRmax = np.amin(CR), np.amax(CR)
        errors = []
        name = []
        for d in degrees:
            errors.append(error[:,d])
            name.append('Degree='+str(d))

        title = 'Interpolation error, ic='+ str(simulation.ic)+', norm='+norm_title[e]
        filename = graphdir+'cs_interp_ic'+str(ic)+'_norm'+norm_list[e]+'_errors.pdf'
        plot_errors_loglog(Nc, errors, name, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Interpolation convergence rate, ic=' + str(simulation.ic)+', norm='+norm_title[e]
        filename = graphdir+'cs_interp_ic'+str(ic)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, name, filename, title, CRmin, CRmax)
        e = e+1

###################################################################################
# Routine to compute the error of the PPM reconstruction at edges in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_recon(map_projection, transformation, showonscreen, gridload):
    # Number of tests
    Ntest = 7

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 16

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i]  = Nc[i-1]*2

    # Errors array
    recons = (3,4)
    recon_names = ['PPM', 'PPM-CW84','PPM-PL07','PPM-L04']

    if transformation == 'gnomonic_equiangular':
        ets = (1,2,3) # Edge treatment 3 applies only to equiangular CS
    else:
        ets = (1,2)

    error_linf = np.zeros((Ntest, len(ets), len(recons)))
    error_l1   = np.zeros((Ntest, len(ets), len(recons)))
    error_l2   = np.zeros((Ntest, len(ets), len(recons)))

    # Edge treatment
    # Function to be reconstructed
    ic = get_interpolation_parameters()

    # colormap for plotting
    colormap = 'Blues'

    # Let us test and compute the error
    ET = 0
    for et in ets:
        rec = 0
        for recon in recons:
            for i in range(0, Ntest):
                N = int(Nc[i])

                # Get parameters
                simulation = recon_simulation_par(ic, recon, et)

                # Create CS mesh
                cs_grid = cubed_sphere(N, transformation, False, gridload)

                # Create the latlon mesh (for plotting)
                ll_grid = latlon_grid(Nlat, Nlon)
                ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

                # Interior cells index (ignoring ghost cells)
                i0   = cs_grid.i0
                iend = cs_grid.iend
                j0   = cs_grid.j0
                jend = cs_grid.jend
                N = cs_grid.N
                nghost = cs_grid.nghost

                # Get values at centers
                Q = np.zeros((N+nghost, N+nghost, nbfaces))
                Qexact = q_scalar_field(cs_grid.centers.lon, cs_grid.centers.lat, simulation)
                q_edx = q_scalar_field(cs_grid.edx.lon, cs_grid.edx.lat, simulation)
                q_edy = q_scalar_field(cs_grid.edy.lon, cs_grid.edy.lat, simulation)

                Q[i0:iend,j0:jend,:] = Qexact[i0:iend,j0:jend,:]

                print('\nParameters: N = '+str(int(Nc[i]))+', et = '+str(et)+' , recon = ', recon)

                if simulation.edge_treatment==1: # Uses adjacent cells values
                    ghost_cells_adjacent_panels(Q, cs_grid, simulation)

                if simulation.edge_treatment==3: # Uses ghost cells interpolation
                    # Get Lagrange polynomials
                    lagrange_poly, Kmin, Kmax = lagrange_poly_ghostcells(cs_grid, simulation, transformation)

                    # Interpolate to ghost cells
                    ghost_cells_lagrange_interpolation(Q, cs_grid, transformation, simulation, lagrange_poly, Kmin, Kmax)

                # PPM parabolas
                px = ppm_parabola(cs_grid,simulation,'x')
                py = ppm_parabola(cs_grid,simulation,'y')
                ppm_reconstruction(Q, px, py, cs_grid, simulation)

                # Plot the error
                error_plot = scalar_field(cs_grid, 'error', 'center')
                error_plot.f = 0.0
                error_plot.f = abs(q_edx[i0:iend,j0:jend,:]-px.q_L[i0:iend,j0:jend,:])
                error_plot.f = np.maximum(abs(q_edx[i0:iend,j0:jend,:]-px.q_L[i0:iend,j0:jend,:]),\
                                          abs(q_edx[i0+1:iend+1,j0:jend,:]-px.q_R[i0:iend,j0:jend,:]))
                error_plot.f = np.maximum(error_plot.f, abs(q_edy[i0:iend,j0:jend,:]-py.q_L[i0:iend:,j0:jend,:]))
                error_plot.f = np.maximum(error_plot.f, abs(q_edy[i0:iend,j0+1:jend+1,:]-py.q_R[i0:iend:,j0:jend,:]))

                # Relative errors in different metrics
                error_linf[i,ET,rec], error_l1[i,ET,rec], error_l2[i,ET,rec] = compute_errors(error_plot.f,0*error_plot.f)
                print_errors_simul(error_linf[:,ET,rec], error_l1[:,ET,rec], error_l2[:,ET,rec], i)

                #error_plot.f = Q[i0:iend,j0:jend,:]
                e_ll = nearest_neighbour(error_plot, cs_grid, ll_grid)
                emax_abs = np.amax(abs(e_ll))
                emin, emax = 0, emax_abs
                #emin, emax = np.amin(e_ll), np.amax(e_ll)
                name = 'recon_q_ic_'+str(simulation.ic)+'_recon'+simulation.recon_name\
                +'_et'+str(simulation.edge_treatment)
                filename = 'Reconstruction error, ic='+ str(simulation.ic)+\
                ', recon='+simulation.recon_name+', '+str(simulation.et_name)+', N='+str(cs_grid.N)
                plot_scalar_field(e_ll, name, cs_grid, ll_grid, map_projection, colormap, emin, emax, filename)
            rec = rec+1
        ET = ET + 1

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']

#    for d in range(0, len(deps)):
    e = 0
    for error in error_list:
        emin, emax = np.amin(error[:]), np.amax(error[:])

        # convergence rate min/max
        n = len(error)
        CR = np.abs(np.log(error[1:n])-np.log(error[0:n-1]))/np.log(2.0)
        CRmin, CRmax = np.amin(CR), np.amax(CR)
        errors = []
        name = []
        for r in range(0, len(recons)):
            for et in ets:
                errors.append(error[:,et-1,r])
                name.append(recon_names[recons[r]-1]+'-ET'+str(et))

        title ='Reconstruction error, ic='+ str(simulation.ic)+', norm='+norm_title[e]
        filename = graphdir+'cs_recon_ic'+str(ic)+'_norm'+norm_list[e]+'_parabola_errors.pdf'
        plot_errors_loglog(Nc, errors, name, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Reconstruction convergence rate, ic=' + str(simulation.ic)+', norm='+norm_title[e]
        filename = graphdir+'cs_recon_ic'+str(ic)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, name, filename, title, CRmin, CRmax)
        e = e+1

####################################################################################
# Reconstruction simulation class
####################################################################################
class recon_simulation_par:
    def __init__(self, ic, recon, et):
        # Scalar field
        self.ic = ic

        # Define the scalar field name
        if ic == 1:
            name = 'Gaussian hill'
        elif ic == 2:
            name = 'Trigonometric field'
        else:
            print("Error - invalid scalar field")
            exit()

        if recon == 1:
            self.recon_name = 'PPM-0'
        elif recon == 2:
            self.recon_name = 'PPM-CW84' #Monotonization from Collela and Woodward 84 paper
        elif recon == 3:
            self.recon_name = 'PPM-PL07' # Hybrid PPM from PL07 paper
        elif recon == 4:
            self.recon_name = 'PPM-L04' #Monotonization from Lin 04 paper

        # Interpolation degree
        self.degree = 3

        # IC name
        self.icname = name
        self.title = 'Reconstruction'

        # Edges treatment
        if et==1:
            self.et_name='ET-1'
        elif et==2:
            self.et_name='ET-2'
        elif et==3:
            self.et_name='ET-3'
        else:
            print('ERROR in recon_simulation_par: invalid ET')
            exit()
        self.edge_treatment = et
