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
from cs_datastruct          import scalar_field, cubed_sphere, latlon_grid, ppm_parabola, velocity
from plot                   import plot_scalar_field, save_grid_netcdf4
from errors                 import compute_errors, print_errors_simul, plot_convergence_rate, plot_errors_loglog
from configuration          import get_interpolation_parameters
from scipy.special          import sph_harm
from interpolation          import ll2cs, nearest_neighbour
from cs_transform           import metric_tensor, inverse_equiangular_gnomonic_map
from reconstruction_1d      import ppm_reconstruction
from edges_treatment        import edges_ghost_cell_treatment_scalar
from lagrange               import lagrange_poly_ghostcell_pc, wind_edges2center_lagrange_poly, wind_center2ghostedges_lagrange_poly_ghost
from interpolation          import ghost_cell_pc_lagrange_interpolation, wind_edges2center_lagrange_interpolation, wind_center2ghostedge_lagrange_interpolation
from advection_ic           import velocity_adv
import os.path

####################################################################################
# Interpolation simulation class
####################################################################################
class interpolation_simulation_par:
    def __init__(self, ic, degree):
        # Scalar field
        self.ic = ic
        self.vf = ic

        # Define the scalar field name
        #if ic == 1:
        #    name = 'Gaussian hill'
        #elif ic == 2:
        #    name = 'Trigonometric field'
        #else:
        #    print("Error - invalid scalar field")
        #    exit()

        # IC name
        #self.icname = name

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

cubed_sphere
###################################################################################
# Routine to call the test
###################################################################################
def interpolation_test(map_projection, transformation, showonscreen, gridload):
    # Let us test and compute the error!
    tc, ic, vf = get_interpolation_parameters()

    if tc==1:
        print("Test case 1: Interpolation of scalar field test case.\n")
        error_analysis_sf_interpolation(ic, map_projection, transformation, showonscreen,\
                                        gridload)
    elif tc==2:
        print("Test case 2: Interpolation of vector field at centers test case.\n")
        error_analysis_vf_interpolation_centers(vf, map_projection, transformation, showonscreen,\
                                        gridload)
    elif tc==3:
        print("Test case 3: Interpolation of vector field at ghost cells test case.\n")
        error_analysis_vf_interpolation_ghost_cells(vf, map_projection, transformation, showonscreen,\
                                        gridload)
    elif tc==4:
        print("Test case 4: Reconstruction test case.\n")
        error_analysis_recon(ic, map_projection, transformation, showonscreen, \
                             gridload)
    else:
        print('ERROR in interpolation_test: invalid test case ', tc)
        exit()

###################################################################################
# Routine to compute the scalar field interpolation error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_sf_interpolation(ic, map_projection, transformation, showonscreen, \
                                    gridload):
    # Number of tests
    Ntest = 6

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

    d = 0
    for degree in degrees:
        for i in range(0, Ntest):
            simulation = interpolation_simulation_par(ic, degree)
            N = int(Nc[i])
            print('\nParameters: N = '+str(int(Nc[i]))+', degree = '+str(degree))

            # Create CS mesh
            cs_grid = cubed_sphere(N, transformation, False, gridload)

            # Save the grid
            if not(os.path.isfile(cs_grid.netcdfdata_filename)):
                save_grid_netcdf4(cs_grid)

            # Interior cells index (ignoring ghost cells)
            i0   = cs_grid.i0
            iend = cs_grid.iend
            j0   = cs_grid.j0
            jend = cs_grid.jend
            ngl = cs_grid.ngl
            ngr = cs_grid.ngr

            # Create the latlon mesh (for plotting)
            ll_grid = latlon_grid(Nlat, Nlon)
            ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

            # Exact field
            Q_exact = q_scalar_field(cs_grid.pc.lon, cs_grid.pc.lat, simulation)
            Qexact = scalar_field(cs_grid, 'Q', 'center')
            Qexact.f = Q_exact[i0:iend,j0:jend,:]

            # Interpolated field
            Q_numerical = np.zeros(np.shape(Q_exact))
            Q_numerical[i0:iend,j0:jend,:] = Q_exact[i0:iend,j0:jend,:]

            Q_ll = nearest_neighbour(Qexact, cs_grid, ll_grid)
            name = 'interp_q_ic_'+str(simulation.ic)
            #if degree == 0:
            #    plot_scalar_field(Q_ll, name, cs_grid, ll_grid, map_projection)

            # Get Lagrange polynomials
            lagrange_poly_ghostcell_pc(cs_grid, simulation)

            # Interpolate to ghost cells
            ghost_cell_pc_lagrange_interpolation(Q_numerical, cs_grid, simulation)

            # Compute the errors
            error_linf[i,d], _, _ = compute_errors(Q_numerical, Q_exact)

            # Print errors
            print_errors_simul(error_linf[:,d], error_linf[:,d], error_linf[:,d], i)

        d = d+1
        print()

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf,]
    norm_list  = ['linf',]
    norm_title  = [r'$L_{\infty}$',]
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
# Routine to compute the vector field interpolation error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_vf_interpolation_centers(vf, map_projection, transformation, showonscreen,\
                                    gridload):
    # Number of tests
    Ntest = 6

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 16

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2

    # Errors array
    degrees = (0,1,2,3,)
    error_linf = np.zeros((Ntest, len(degrees)))
    error_l1   = np.zeros((Ntest, len(degrees)))
    error_l2   = np.zeros((Ntest, len(degrees)))

    # Let us test and compute the error!
    d = 0
    for degree in degrees:
        for i in range(0, Ntest):
            simulation = interpolation_simulation_par(vf, degree)
            N = int(Nc[i])
            print('\nParameters: N = '+str(int(Nc[i]))+', degree = '+str(degree))

            # Create CS mesh
            cs_grid = cubed_sphere(N, transformation, False, gridload)

            # Save the grid
            if not(os.path.isfile(cs_grid.netcdfdata_filename)):
                save_grid_netcdf4(cs_grid)

            # Create the latlon mesh (for plotting)
            ll_grid = latlon_grid(Nlat, Nlon)
            ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)


            # Interior cells index (ignoring ghost cells)
            i0   = cs_grid.i0
            iend = cs_grid.iend
            j0   = cs_grid.j0
            jend = cs_grid.jend
            ngl = cs_grid.ngl
            ngr = cs_grid.ngr

            # Order
            interpol_degree = simulation.degree
            order = interpol_degree + 1

            # Velocity at edges
            U_pu = velocity(cs_grid, 'pu')
            U_pu_exact = velocity(cs_grid, 'pu')
            U_pv = velocity(cs_grid, 'pv')
            U_pv_exact = velocity(cs_grid, 'pv')
            U_pc = velocity(cs_grid, 'pc')
            U_pc_exact = velocity(cs_grid, 'pc')

            # Get velocities at pu, pv, pc
            U_pu_exact.ulon[:,:,:], U_pu_exact.vlat[:,:,:] = velocity_adv(cs_grid.pu.lon, cs_grid.pu.lat, 0.0, simulation)
            U_pv_exact.ulon[:,:,:], U_pv_exact.vlat[:,:,:] = velocity_adv(cs_grid.pv.lon, cs_grid.pv.lat, 0.0, simulation)
            U_pc_exact.ulon[:,:,:], U_pc_exact.vlat[:,:,:] = velocity_adv(cs_grid.pc.lon, cs_grid.pc.lat, 0.0, simulation)

            # Convert latlon to contravariant at pu
            U_pu_exact.ucontra[:,:,:], U_pu_exact.vcontra[:,:,:] = latlon_to_contravariant(U_pu_exact.ulon, U_pu_exact.vlat, cs_grid.prod_ex_elon_pu, cs_grid.prod_ex_elat_pu,\
                                                               cs_grid.prod_ey_elon_pu, cs_grid.prod_ey_elat_pu, cs_grid.determinant_ll2contra_pu)

            # Convert latlon to contravariant at pv
            U_pv_exact.ucontra[:,:,:], U_pv_exact.vcontra[:,:,:] = latlon_to_contravariant(U_pv_exact.ulon, U_pv_exact.vlat, cs_grid.prod_ex_elon_pv, cs_grid.prod_ex_elat_pv,\
                                                               cs_grid.prod_ey_elon_pv, cs_grid.prod_ey_elat_pv, cs_grid.determinant_ll2contra_pv)

            # Convert latlon to contravariant at pc
            U_pc_exact.ucontra[:,:,:], U_pc_exact.vcontra[:,:,:] = latlon_to_contravariant(U_pc_exact.ulon, U_pc_exact.vlat, cs_grid.prod_ex_elon_pc, cs_grid.prod_ex_elat_pc,\
                                                               cs_grid.prod_ey_elon_pc, cs_grid.prod_ey_elat_pc, cs_grid.determinant_ll2contra_pc)

            U_pu.ucontra[i0:iend+1,j0:jend,:] = U_pu_exact.ucontra[i0:iend+1,j0:jend,:]
            U_pv.vcontra[i0:iend,j0:jend+1,:] = U_pv_exact.vcontra[i0:iend,j0:jend+1,:]

            # Compute the Lagrange polynomials
            wind_edges2center_lagrange_poly(cs_grid, simulation)

            if cs_grid.projection == 'gnomonic_equiangular':
                lagrange_poly_ghostcell_pc(cs_grid, simulation)

            # Interpolate the wind to cells pc
            wind_edges2center_lagrange_interpolation(U_pc, U_pu, U_pv, cs_grid, simulation)
 
            # Error at pc
            eu = abs(U_pc.ulon[i0:iend,j0:jend,:]-U_pc_exact.ulon[i0:iend,j0:jend,:])#np.amax(abs(U_pc_exact.ulon))
            ev = abs(U_pc.vlat[i0:iend,j0:jend,:]-U_pc_exact.vlat[i0:iend,j0:jend,:])#np.amax(abs(U_pc_exact.vlat))
            #eu = abs(U_pc.ucontra[i0:iend,j0:jend,:]-U_pc_exact.ucontra[i0:iend,j0:jend,:])#np.amax(abs(U_pc_exact.ulon))
            #ev = abs(U_pc.vcontra[i0:iend,j0:jend,:]-U_pc_exact.vcontra[i0:iend,j0:jend,:])#np.amax(abs(U_pc_exact.vlat))
            error_linf[i,d] = np.amax(np.maximum(eu, ev))

            # plot the error
            error_plot = scalar_field(cs_grid, 'error', 'center')
            error_plot.f[:,:,:] = np.maximum(eu, ev)
            e_ll = nearest_neighbour(error_plot, cs_grid, ll_grid)
            emax_abs = np.amax(abs(e_ll))
            emin, emax = 0, emax_abs
            colormap = 'Blues' 
            #print(emax_abs)
            #emin, emax = np.amin(e_ll), np.amax(e_ll)
            name = 'reconwind_vf_'+str(simulation.vf)+'_degree'+str(d)
            filename = 'reconstruction error, vf='+ str(simulation.vf)
            #plot_scalar_field(e_ll, name, cs_grid, ll_grid, map_projection, colormap, emin, emax, filename)

            # Print errors
            print_errors_simul(error_linf[:,d], error_linf[:,d], error_linf[:,d], i)

        d = d+1
        print()

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf,]
    norm_list  = ['linf',]
    norm_title  = [r'$L_{\infty}$',]
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

        title = 'Interpolation error, vf='+ str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_interp_vf'+str(vf)+'_norm'+norm_list[e]+'_errors.pdf'
        plot_errors_loglog(Nc, errors, name, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Interpolation convergence rate, vf=' + str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_interp_vf'+str(vf)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, name, filename, title, CRmin, CRmax)
        e = e+1


###################################################################################
# Routine to compute the vector field interpolation error convergence in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_vf_interpolation_ghost_cells(vf, map_projection, transformation, showonscreen,\
                                    gridload):
    # Number of tests
    Ntest = 6

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 16

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i] = Nc[i-1]*2

    # Errors array
    degrees = (0,1,2,3,4,)
    error_linf = np.zeros((Ntest, len(degrees)))
    error_l1   = np.zeros((Ntest, len(degrees)))
    error_l2   = np.zeros((Ntest, len(degrees)))

    # Let us test and compute the error!
    d = 0
    for degree in degrees:
        for i in range(0, Ntest):
            simulation = interpolation_simulation_par(vf, degree)
            N = int(Nc[i])
            print('\nParameters: N = '+str(int(Nc[i]))+', degree = '+str(degree))

            # Create CS mesh
            cs_grid = cubed_sphere(N, transformation, False, gridload)

            # Save the grid
            if not(os.path.isfile(cs_grid.netcdfdata_filename)):
                save_grid_netcdf4(cs_grid)

            # Interior cells index (ignoring ghost cells)
            i0   = cs_grid.i0
            iend = cs_grid.iend
            j0   = cs_grid.j0
            jend = cs_grid.jend
            ngl = cs_grid.ngl
            ngr = cs_grid.ngr

            # Order
            interpol_degree = simulation.degree
            order = interpol_degree + 1

            # Velocity at edges
            U_pu = velocity(cs_grid, 'pu')
            U_pu_exact = velocity(cs_grid, 'pu')
            U_pv = velocity(cs_grid, 'pv')
            U_pv_exact = velocity(cs_grid, 'pv')
            U_pc = velocity(cs_grid, 'pc')
            U_pc_exact = velocity(cs_grid, 'pc')

            # Get velocities at pu, pv, pc
            U_pu_exact.ulon[:,:,:], U_pu_exact.vlat[:,:,:] = velocity_adv(cs_grid.pu.lon, cs_grid.pu.lat, 0.0, simulation)
            U_pv_exact.ulon[:,:,:], U_pv_exact.vlat[:,:,:] = velocity_adv(cs_grid.pv.lon, cs_grid.pv.lat, 0.0, simulation)
            U_pc_exact.ulon[:,:,:], U_pc_exact.vlat[:,:,:] = velocity_adv(cs_grid.pc.lon, cs_grid.pc.lat, 0.0, simulation)

            # Convert latlon to contravariant at pu
            U_pu_exact.ucontra[:,:,:], U_pu_exact.vcontra[:,:,:] = latlon_to_contravariant(U_pu_exact.ulon, U_pu_exact.vlat, cs_grid.prod_ex_elon_pu, cs_grid.prod_ex_elat_pu,\
                                                               cs_grid.prod_ey_elon_pu, cs_grid.prod_ey_elat_pu, cs_grid.determinant_ll2contra_pu)

            # Convert latlon to contravariant at pv
            U_pv_exact.ucontra[:,:,:], U_pv_exact.vcontra[:,:,:] = latlon_to_contravariant(U_pv_exact.ulon, U_pv_exact.vlat, cs_grid.prod_ex_elon_pv, cs_grid.prod_ex_elat_pv,\
                                                               cs_grid.prod_ey_elon_pv, cs_grid.prod_ey_elat_pv, cs_grid.determinant_ll2contra_pv)

            # Convert latlon to contravariant at pc
            U_pc_exact.ucontra[:,:,:], U_pc_exact.vcontra[:,:,:] = latlon_to_contravariant(U_pc_exact.ulon, U_pc_exact.vlat, cs_grid.prod_ex_elon_pc, cs_grid.prod_ex_elat_pc,\
                                                               cs_grid.prod_ey_elon_pc, cs_grid.prod_ey_elat_pc, cs_grid.determinant_ll2contra_pc)

            U_pu.ucontra[i0:iend+1,j0:jend,:] = U_pu_exact.ucontra[i0:iend+1,j0:jend,:]
            U_pv.vcontra[i0:iend,j0:jend+1,:] = U_pv_exact.vcontra[i0:iend,j0:jend+1,:]

            # Compute the Lagrange polynomials
            lagrange_poly_ghostcell_pc(cs_grid, simulation)
            wind_edges2center_lagrange_poly(cs_grid, simulation)
            wind_center2ghostedges_lagrange_poly_ghost(cs_grid, simulation)

            # Interpolate the wind to cells pc
            wind_edges2center_lagrange_interpolation(U_pc, U_pu, U_pv, cs_grid, simulation)

            # Interpolate the wind from pc to ghost cells edges
            wind_center2ghostedge_lagrange_interpolation(U_pc, U_pu, U_pv, cs_grid, simulation)

            # Error at pc
            eu = np.amax(abs(U_pc.ulon-U_pc_exact.ulon))#/np.amax(abs(U_pc_exact.ulon))
            ev = np.amax(abs(U_pc.vlat-U_pc_exact.vlat))#/np.amax(abs(U_pc_exact.vlat))
            e_pc = max(eu, ev)

            # Error at ghost cell edges
            # East
            eu_east = np.amax(abs(U_pv.ucontra[iend:,j0:jend+1,:] - U_pv_exact.ucontra[iend:,j0:jend+1,:]))/ np.amax(abs(U_pv_exact.ucontra[iend:,j0:jend+1,:]))
            ev_east = np.amax(abs(U_pv.vcontra[iend:,j0:jend+1,:] - U_pv_exact.vcontra[iend:,j0:jend+1,:]))/ np.amax(abs(U_pv_exact.vcontra[iend:,j0:jend+1,:]))

            # West
            eu_west = np.amax(abs(U_pv.ucontra[:i0,j0:jend+1,:] - U_pv_exact.ucontra[:i0,j0:jend+1,:]))/ np.amax(abs(U_pv_exact.ucontra[:i0,j0:jend+1,:]))
            ev_west = np.amax(abs(U_pv.vcontra[:i0,j0:jend+1,:] - U_pv_exact.vcontra[:i0,j0:jend+1,:]))/ np.amax(abs(U_pv_exact.vcontra[:i0,j0:jend+1,:]))

            # North
            eu_north = np.amax(abs(U_pu.ucontra[i0:iend+1,jend:,:] - U_pu_exact.ucontra[i0:iend+1,jend:,:]))/ np.amax(abs(U_pu_exact.ucontra[i0:iend+1,jend:,:]))
            ev_north = np.amax(abs(U_pu.vcontra[i0:iend+1,iend:,:] - U_pu_exact.vcontra[i0:iend+1,jend:,:]))/ np.amax(abs(U_pu_exact.vcontra[i0:iend+1,jend:,:]))

            # South
            eu_south = np.amax(abs(U_pu.ucontra[i0:iend+1,:j0,:] - U_pu_exact.ucontra[i0:iend+1,:j0,:]))/ np.amax(abs(U_pu_exact.ucontra[i0:iend+1,:j0,:]))
            ev_south = np.amax(abs(U_pu.vcontra[i0:iend+1,:j0,:] - U_pu_exact.vcontra[i0:iend+1,:j0,:]))/ np.amax(abs(U_pu_exact.vcontra[i0:iend+1,:j0,:]))

            e_east = max(eu_east, ev_east)
            e_west = max(eu_west, ev_west)
            e_north = max(eu_north, ev_north)
            e_south = max(eu_south, ev_south)
            e_edges = max(e_east, e_west, e_north, e_south)

            error_linf[i,d] = e_edges
            #error_linf[i,d] = e_pc
            # Print errors
            print_errors_simul(error_linf[:,d], error_linf[:,d], error_linf[:,d], i)

        d = d+1
        print()

    # Outputs
    # plot errors for different all schemes in  different norms
    error_list = [error_linf,]
    norm_list  = ['linf',]
    norm_title  = [r'$L_{\infty}$',]
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

        title = 'Interpolation error, vf='+ str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_interp_vf'+str(vf)+'_norm'+norm_list[e]+'_errors.pdf'
        plot_errors_loglog(Nc, errors, name, filename, title, emin, emax)

        # Plot the convergence rate
        title = 'Interpolation convergence rate, vf=' + str(simulation.vf)+', norm='+norm_title[e]
        filename = graphdir+'cs_interp_vf'+str(vf)+'_norm'+norm_list[e]+'_convergence_rate.pdf'
        plot_convergence_rate(Nc, errors, name, filename, title, CRmin, CRmax)
        e = e+1

###################################################################################
# Routine to compute the error of the PPM reconstruction at edges in L_inf, L1 and L2 norms
####################################################################################
def error_analysis_recon(ic, map_projection, transformation, showonscreen, gridload):
    # Number of tests
    Ntest = 6

    # Number of cells along a coordinate axis
    Nc = np.zeros(Ntest)
    Nc[0] = 16

    # Compute number of cells for each simulation
    for i in range(1, Ntest):
        Nc[i]  = Nc[i-1]*2

    # Errors array
    recons = (1,)
    recon_names = ['PPM-0', 'PPM-CW84','PPM-PL07','PPM-L04']
    et_names = ['ET-S72','ET-PL07','ET-Z21']

    if transformation == 'gnomonic_equiangular':
        ets = (1,2,3) # Edge treatment 3 applies only to equiangular CS
        #ets = (3,) # Edge treatment 3 applies only to equiangular CS
    elif transformation == 'overllaped':
        ets = (2,)
    else:
        ets = (1,2)

    error_linf = np.zeros((Ntest, len(ets), len(recons)))
    error_l1   = np.zeros((Ntest, len(ets), len(recons)))
    error_l2   = np.zeros((Ntest, len(ets), len(recons)))

    # colormap for plotting
    colormap = 'Blues'
    #colormap = 'seismic'

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
                cs_grid = cubed_sphere(N, transformation, False, True)

                # Save the grid
                if not(os.path.isfile(cs_grid.netcdfdata_filename)):
                    save_grid_netcdf4(cs_grid)

                # Create the latlon mesh (for plotting)
                ll_grid = latlon_grid(Nlat, Nlon)
                ll_grid.ix, ll_grid.jy, ll_grid.mask = ll2cs(cs_grid, ll_grid)

                # Interior cells index (ignoring ghost cells)
                i0   = cs_grid.i0
                iend = cs_grid.iend
                j0   = cs_grid.j0
                jend = cs_grid.jend
                N = cs_grid.N
                ng = cs_grid.ng

                # Get values at pc
                Q = np.zeros((N+ng, N+ng, nbfaces))
                Qexact = q_scalar_field(cs_grid.pc.lon, cs_grid.pc.lat, simulation)
                q_pu = q_scalar_field(cs_grid.pu.lon, cs_grid.pu.lat, simulation)
                q_pv = q_scalar_field(cs_grid.pv.lon, cs_grid.pv.lat, simulation)
                Q[i0:iend,j0:jend,:] = Qexact[i0:iend,j0:jend,:]

                print('\nParameters: N = '+str(int(Nc[i]))+', et = '+str(et)+' , recon = ', recon)

                # get lagrange_poly
                if cs_grid.projection == 'gnomonic_equiangular':
                    lagrange_poly_ghostcell_pc(cs_grid, simulation)

                # Fill halo data
                edges_ghost_cell_treatment_scalar(Q, Q, cs_grid, simulation)

                # PPM parabolas
                px = ppm_parabola(cs_grid,simulation,'x')
                py = ppm_parabola(cs_grid,simulation,'y')
                ppm_reconstruction(Q, Q, px, py, cs_grid, simulation)

                # plot the error
                error_plot = scalar_field(cs_grid, 'error', 'center')
                error_plot.f[:,:,:] = abs(q_pu[i0:iend,j0:jend,:]-px.q_L[i0:iend,j0:jend,:])
                error_plot.f = np.maximum(error_plot.f, abs(q_pu[i0+1:iend+1,j0:jend,:]-px.q_R[i0:iend,j0:jend,:]))
                error_plot.f = np.maximum(error_plot.f, abs(q_pv[i0:iend,j0:jend,:]-py.q_L[i0:iend:,j0:jend,:]))
                error_plot.f = np.maximum(error_plot.f, abs(q_pv[i0:iend,j0+1:jend+1,:]-py.q_R[i0:iend:,j0:jend,:]))

                #error_plot.f = q[i0:iend,j0:jend,:]
                #error_plot.f[0,:,:] = 0.0
                #error_plot.f[1,:,:] = 0.0
                #error_plot.f[n-1,:,:] = 0.0
                #error_plot.f[n-2,:,:] = 0.0
                #error_plot.f[:,0,:] = 0.0
                #error_plot.f[:,1,:] = 0.0
                #error_plot.f[:,n-2,:] = 0.0
                #error_plot.f[:,n-1,:] = 0.0
                #error_plot.f[:,:,4:6] = 0.0

                # relative errors in different metrics
                error_linf[i,ET,rec], error_l1[i,ET,rec], error_l2[i,ET,rec] = compute_errors(error_plot.f,0*error_plot.f)
                print_errors_simul(error_linf[:,ET,rec], error_l1[:,ET,rec], error_l2[:,ET,rec], i)

                e_ll = nearest_neighbour(error_plot, cs_grid, ll_grid)
                emax_abs = np.amax(abs(e_ll))
                emin, emax = 0, emax_abs
                #print(emax_abs)
                #emin, emax = np.amin(e_ll), np.amax(e_ll)
                name = 'recon_q_ic_'+str(simulation.ic)+'_recon'+simulation.recon_name\
                +'_et'+str(simulation.edge_treatment)
                filename = 'reconstruction error, ic='+ str(simulation.ic)+\
                ', recon='+simulation.recon_name+', '+str(simulation.et_name)+', N='+str(cs_grid.N)
                plot_scalar_field(e_ll, name, cs_grid, ll_grid, map_projection, colormap, emin, emax, filename)
            rec = rec+1
        ET = ET + 1
    exit()
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
            for et in range(0, len(ets)):
                errors.append(error[:,et,r])
                name.append(recon_names[recons[r]-1]+'/'+et_names[ets[et]-1])

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
        elif ic == 3:
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
        #self.icname = name
        self.title = 'Reconstruction'

        # Edges treatment
        if et==1:
            self.et_name='ET-S72'
        elif et==2:
            self.et_name='ET-PL07'
        elif et==3:
            self.et_name='ET-Z21'
        else:
            print('ERROR in recon_simulation_par: invalid ET')
            exit()
        self.edge_treatment = et
