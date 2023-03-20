####################################################################################
#
# Module for cubed-sphere mesh generation and structuring.
#
# Based on "C. Ronchi, R. Iacono, P.S. Paolucci, The “Cubed Sphere”:
# A New Method for the Solution of Partial Differential Equations
# in Spherical Geometry."
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from constants import*
from cs_transform import*
from sphgeo import point, tg_vector_geodesic_edx_midpoints, tg_vector_geodesic_edy_midpoints
import time
import os.path
import netCDF4 as nc

####################################################################################
#  Global cubed-sphere grid structure
####################################################################################
class cubed_sphere:
    def __init__(self, N, transformation, showonscreen, gridload):
        # Panel indexes distribution
        #      +---+
        #      | 4 |
        #  +---+---+---+---+
        #  | 3 | 0 | 1 | 2 |
        #  +---+---+---+---+
        #      | 5 |
        #      +---+

        # Number of cells along a coordinate axis (same for all panels)
        self.N = N

        # Sphere radius
        #self.R = erad
        self.R = 1.0

        # Grid name
        self.projection = transformation
        self.name = self.projection+"_"+str(N)

        # Grid data netcdf filename
        self.netcdfdata_filename = griddir+self.name+".nc"

        # Load the grid
        if gridload==True and (os.path.isfile(self.netcdfdata_filename)):   # Check if grid file exists
            if showonscreen==True:
                print("--------------------------------------------------------")
                print('Loading grid file '+self.netcdfdata_filename)

            # Start time counting
            start_time = time.time()

            # Open grid data
            griddata = nc.Dataset(self.netcdfdata_filename,'r')

            # Grid spacing
            dx = griddata['dx'][:]
            dy = griddata['dy'][:]
            self.dx, self.dy = dx, dy

            # Get some integers
            self.nghost        = griddata['nghost'][:]
            self.nghost_left   = griddata['nghost_left'][:]
            self.nghost_right  = griddata['nghost_right'][:]

            # Interior indexes
            self.i0, self.iend = griddata['i0'][:], griddata['iend'][:]
            self.j0, self.jend = griddata['j0'][:], griddata['jend'][:]
            i0, iend = self.i0, self.iend
            j0, jend = self.j0, self.jend
            nghost = self.nghost

            # Create points
            vertices  = point(N+1+nghost, N+1+nghost)
            centers   = point(N+nghost, N+nghost)
            edx       = point(N+1+nghost, N+nghost)
            edy       = point(N+nghost, N+1+nghost)
            tg_ex_edx = point(N+1+nghost, N+nghost)
            tg_ey_edx = point(N+1+nghost, N+nghost)
            tg_ex_edy = point(N+nghost, N+1+nghost)
            tg_ey_edy = point(N+nghost, N+1+nghost)


            # Get values from file
            vertices.X   = griddata['vertices'][:,:,:,0]
            vertices.Y   = griddata['vertices'][:,:,:,1]
            vertices.Z   = griddata['vertices'][:,:,:,2]
            vertices.lon = griddata['vertices'][:,:,:,3]
            vertices.lat = griddata['vertices'][:,:,:,4]

            centers.X   = griddata['centers'][:,:,:,0]
            centers.Y   = griddata['centers'][:,:,:,1]
            centers.Z   = griddata['centers'][:,:,:,2]
            centers.lon = griddata['centers'][:,:,:,3]
            centers.lat = griddata['centers'][:,:,:,4]

            edx.X   = griddata['edx'][:,:,:,0]
            edx.Y   = griddata['edx'][:,:,:,1]
            edx.Z   = griddata['edx'][:,:,:,2]
            edx.lon = griddata['edx'][:,:,:,3]
            edx.lat = griddata['edx'][:,:,:,4]

            edy.X   = griddata['edy'][:,:,:,0]
            edy.Y   = griddata['edy'][:,:,:,1]
            edy.Z   = griddata['edy'][:,:,:,2]
            edy.lon = griddata['edy'][:,:,:,3]
            edy.lat = griddata['edy'][:,:,:,4]

            tg_ex_edx.X   = griddata['tg_ex_edx'][:,:,:,0]
            tg_ex_edx.Y   = griddata['tg_ex_edx'][:,:,:,1]
            tg_ex_edx.Z   = griddata['tg_ex_edx'][:,:,:,2]
            tg_ex_edx.lon = griddata['tg_ex_edx'][:,:,:,3]
            tg_ex_edx.lat = griddata['tg_ex_edx'][:,:,:,4]

            tg_ey_edx.X   = griddata['tg_ey_edx'][:,:,:,0]
            tg_ey_edx.Y   = griddata['tg_ey_edx'][:,:,:,1]
            tg_ey_edx.Z   = griddata['tg_ey_edx'][:,:,:,2]
            tg_ey_edx.lon = griddata['tg_ey_edx'][:,:,:,3]
            tg_ey_edx.lat = griddata['tg_ey_edx'][:,:,:,4]

            tg_ex_edy.X   = griddata['tg_ex_edy'][:,:,:,0]
            tg_ex_edy.Y   = griddata['tg_ex_edy'][:,:,:,1]
            tg_ex_edy.Z   = griddata['tg_ex_edy'][:,:,:,2]
            tg_ex_edy.lon = griddata['tg_ex_edy'][:,:,:,3]
            tg_ex_edy.lat = griddata['tg_ex_edy'][:,:,:,4]

            tg_ey_edx.X   = griddata['tg_ey_edy'][:,:,:,0]
            tg_ey_edx.Y   = griddata['tg_ey_edy'][:,:,:,1]
            tg_ey_edx.Z   = griddata['tg_ey_edy'][:,:,:,2]
            tg_ey_edx.lon = griddata['tg_ey_edy'][:,:,:,3]
            tg_ey_edx.lat = griddata['tg_ey_edy'][:,:,:,4]



            # Geometric properties
            areas           = griddata['areas'][:,:,:]
            length_edx      = griddata['length_edx'][:,:,:]
            length_edy      = griddata['length_edy'][:,:,:]

            # Metric tensor/contravariant-covariant conversion tools
            metric_tensor_centers = griddata['metric_tensor_centers'][:,:,:]

            prod_ex_elon_edx          = griddata['prod_ex_elon_edx'][:,:,:]
            prod_ex_elat_edx          = griddata['prod_ex_elat_edx'][:,:,:]
            prod_ey_elon_edx          = griddata['prod_ey_elon_edx'][:,:,:]
            prod_ey_elat_edx          = griddata['prod_ey_elat_edx'][:,:,:]
            determinant_ll2contra_edx = griddata['determinant_ll2contra_edx'][:,:,:]

            prod_ex_elon_edy          = griddata['prod_ex_elon_edy'][:,:,:]
            prod_ex_elat_edy          = griddata['prod_ex_elat_edy'][:,:,:]
            prod_ey_elon_edy          = griddata['prod_ey_elon_edy'][:,:,:]
            prod_ey_elat_edy          = griddata['prod_ey_elat_edy'][:,:,:]
            determinant_ll2contra_edy = griddata['determinant_ll2contra_edy'][:,:,:]

            Xu = griddata['Xu'][:,:,:]
            Yv = griddata['Yv'][:,:,:]


            # Attributes
            self.centers         = centers
            self.vertices        = vertices
            self.edx             = edx
            self.edy             = edy
            self.length_edx      = length_edx
            self.length_edy      = length_edy
            self.areas           = areas

            self.metric_tensor_centers = metric_tensor_centers

            self.prod_ex_elon_edx = prod_ex_elon_edx
            self.prod_ex_elat_edx = prod_ex_elat_edx
            self.prod_ey_elon_edx = prod_ey_elon_edx
            self.prod_ey_elat_edx = prod_ey_elat_edx
            self.determinant_ll2contra_edx = determinant_ll2contra_edx

            self.prod_ex_elon_edy = prod_ex_elon_edy
            self.prod_ex_elat_edy = prod_ex_elat_edy
            self.prod_ey_elon_edy = prod_ey_elon_edy
            self.prod_ey_elat_edy = prod_ey_elat_edy
            self.determinant_ll2contra_edy = determinant_ll2contra_edy

            self.Xu = Xu
            self.Yv = Yv

            # Close netcdf file
            griddata.close()

            # Finish time counting
            elapsed_time = time.time() - start_time

        else:
            # Generate the grid
            if showonscreen==True:
                print("--------------------------------------------------------")
                print("Generating "+self.projection+" cubed-sphere with "+str(nbfaces*N*N)+" cells...")

            if transformation == "gnomonic_equiangular":
                x_min, x_max, y_min, y_max = [-pio4, pio4, -pio4, pio4] # Angular coordinates
            elif transformation == "gnomonic_equidistant":
                a = self.R/np.sqrt(3.0)  # Half length of the cube
                x_min, x_max, y_min, y_max = [-a, a, -a, a]
            elif transformation == "conformal":
                a = 1.0
                x_min, x_max, y_min, y_max = [-1.0, 1.0, -1.0, 1.0]
            else:
                print("ERROR: invalid grid transformation.")
                exit()

            # Start time counting
            start_time = time.time()

            # Grid spacing
            dx = (x_max-x_min)/N
            dy = (y_max-y_min)/N
            self.dx, self.dy = dx, dy

            # Ghost cells size
            nghost_left  = 4
            nghost_right = 4

            nghost = nghost_right + nghost_left
            self.nghost_left  = nghost_left
            self.nghost_right = nghost_right
            self.nghost = nghost

            # Interior indexes
            i0, iend = nghost_left, nghost_left+N
            j0, jend = nghost_left, nghost_left+N
            self.i0, self.iend = nghost_left, nghost_left+N
            self.j0, self.jend = nghost_left, nghost_left+N

            # Generate cell vertices
            x = np.linspace(x_min-nghost_left*dx, x_max + nghost_right*dx, N+1+nghost) # vertices
            y = np.linspace(y_min-nghost_left*dx, y_max + nghost_right*dy, N+1+nghost) # vertices
            if showonscreen==True:
                print("Generating cell vertices...")

            vertices = point(N+1+nghost, N+1+nghost)

            if transformation == "gnomonic_equiangular":
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = equiangular_gnomonic_map(x, y, N+1+nghost, N+1+nghost, self.R)

            elif transformation=="gnomonic_equidistant":
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = equidistant_gnomonic_map(x, y, N+1+nghost, N+1+nghost, self.R)

            elif transformation=="conformal":
                vertices.X[i0:iend+1,j0:jend+1], vertices.Y[i0:iend+1,j0:jend+1], vertices.Z[i0:iend+1,j0:jend+1], vertices.lon[i0:iend+1,j0:jend+1], vertices.lat[i0:iend+1,j0:jend+1] = conformal_map(x[i0:iend+1], y[j0:jend+1], N+1, N+1)

            self.vertices = vertices

            # Generate cell centers
            if showonscreen==True:
                print("Generating cell centers...")

            centers = point(N+nghost, N+nghost)

            x = np.linspace(x_min+dx/2.0-nghost_left*dx, x_max-dx/2.0+nghost_right*dx, N+nghost) # Centers
            y = np.linspace(y_min+dy/2.0-nghost_left*dy, y_max-dy/2.0+nghost_right*dy, N+nghost) # Centers
            if transformation == "gnomonic_equiangular":
                centers.X, centers.Y, centers.Z, centers.lon, centers.lat = equiangular_gnomonic_map(x, y, N+nghost, N+nghost, self.R)

            elif transformation=="gnomonic_equidistant":
                centers.X, centers.Y, centers.Z, centers.lon, centers.lat = equidistant_gnomonic_map(x, y, N+nghost, N+nghost, self.R)

            elif transformation=="conformal":
                centers.X[i0:iend,j0:jend], centers.Y[i0:iend,j0:jend], centers.Z[i0:iend,j0:jend], centers.lon[i0:iend,j0:jend], centers.lat[i0:iend,j0:jend] \
                = conformal_map(x[i0:iend], y[j0:jend], N, N)

            self.centers = centers

            # Metric tensor on centers
            metric_tensor_centers = np.zeros((N+nghost, N+nghost, nbfaces))
            metric_tensor_centers[:,:,0] = metric_tensor(x, y, self.R, transformation)
            for p in range(1, nbfaces): metric_tensor_centers[:,:,p] = metric_tensor_centers[:,:,0]
            self.metric_tensor_centers = metric_tensor_centers

            # Generate cell edges in x direction
            if showonscreen==True:
                print("Generating cell edges and tangent vectors in x direction...")

            edx       = point(N+1+nghost, N+nghost)
            tg_ex_edx = point(N+1+nghost, N+nghost)
            tg_ey_edx = point(N+1+nghost, N+nghost)

            x = np.linspace(x_min-nghost_left*dx, x_max+nghost_right*dx, N+1+nghost) # Edges
            y = np.linspace(y_min+dy/2.0-nghost_left*dy, y_max-dy/2.0+nghost_right*dy, N+nghost) # Centers

            # edges
            Xu, Yu = np.meshgrid(x, y,indexing='ij')
            if transformation == "gnomonic_equiangular":
                edx.X, edx.Y, edx.Z, edx.lon, edx.lat = equiangular_gnomonic_map(x, y, N+1+nghost, N+nghost, self.R)
                tg_ex_edx.X, tg_ex_edx.Y, tg_ex_edx.Z = equiangular_tg_xdir(x, y, N+1+nghost, N+nghost, self.R)
                tg_ey_edx.X, tg_ey_edx.Y, tg_ey_edx.Z = equiangular_tg_ydir(x, y, N+1+nghost, N+nghost, self.R)
                #tg_ex_edx2 = point(N+1+nghost, N+nghost)
                #tg_ey_edx2 = point(N+1+nghost, N+nghost)
                #tg_ex_edx2, tg_ey_edx2 = tg_vector_geodesic_edx_midpoints(edx, vertices, N, nghost)
            elif transformation=="gnomonic_equidistant":
                edx.X, edx.Y, edx.Z, edx.lon, edx.lat = equidistant_gnomonic_map(x, y, N+1+nghost, N+nghost, self.R)
                tg_ex_edx.X, tg_ex_edx.Y, tg_ex_edx.Z = equidistant_tg_xdir(x, y, N+1+nghost, N+nghost, self.R)
                tg_ey_edx.X, tg_ey_edx.Y, tg_ey_edx.Z = equidistant_tg_ydir(x, y, N+1+nghost, N+nghost, self.R)
                #tg_ex_edx2 = point(N+1+nghost, N+nghost)
                #tg_ey_edx2 = point(N+1+nghost, N+nghost)
                #tg_ex_edx2, tg_ey_edx2 = tg_vector_geodesic_edx_midpoints(edx, vertices, N, nghost)

            elif transformation=="conformal":
                edx.X[i0:iend+1,j0:jend], edx.Y[i0:iend+1,j0:jend], edx.Z[i0:iend+1,j0:jend], edx.lon[i0:iend+1,j0:jend], edx.lat[i0:iend+1,j0:jend] \
                = conformal_map(x[i0:iend+1], y[j0:jend], N+1, N)

            self.edx = edx

            # Normalize the tangent vectors in x dir
            norm = tg_ex_edx.X*tg_ex_edx.X + tg_ex_edx.Y*tg_ex_edx.Y + tg_ex_edx.Z*tg_ex_edx.Z
            norm = np.sqrt(norm)
            tg_ex_edx.X = tg_ex_edx.X#/norm
            tg_ex_edx.Y = tg_ex_edx.Y#/norm
            tg_ex_edx.Z = tg_ex_edx.Z#/norm

            # Normalize the tangent vectors in y dir
            norm = tg_ey_edx.X*tg_ey_edx.X + tg_ey_edx.Y*tg_ey_edx.Y + tg_ey_edx.Z*tg_ey_edx.Z
            norm = np.sqrt(norm)
            tg_ey_edx.X = tg_ey_edx.X#/norm
            tg_ey_edx.Y = tg_ey_edx.Y#/norm
            tg_ey_edx.Z = tg_ey_edx.Z#/norm

            #print(np.amax(abs(tg_ex_edx2.X-tg_ex_edx.X)))
            #print(np.amax(abs(tg_ex_edx2.Y-tg_ex_edx.Y)))
            #print(np.amax(abs(tg_ex_edx2.Z-tg_ex_edx.Z)))
            #print(np.amax(abs(tg_ey_edx2.X-tg_ey_edx.X)))
            #print(np.amax(abs(tg_ey_edx2.Y-tg_ey_edx.Y)))
            #print(np.amax(abs(tg_ey_edx2.Z-tg_ey_edx.Z)))

            # Metric tensor on edges in x direction
            metric_tensor_edx = np.zeros((N+1+nghost, N+nghost, nbfaces))
            metric_tensor_edx[:,:,0] = metric_tensor(x, y, self.R, transformation)
            for p in range(1, nbfaces): metric_tensor_edx[:,:,p] = metric_tensor_edx[:,:,0]
            self.metric_tensor_edx = metric_tensor_edx

            # Generate cell edges in y direction
            if showonscreen==True:
                print("Generating cell edges and tangent vectors in y direction...")

            edy = point(N+nghost, N+nghost+1)
            tg_ex_edy = point(N+nghost, N+nghost+1)
            tg_ey_edy = point(N+nghost, N+nghost+1)

            x = np.linspace(x_min+dx/2.0-nghost_left*dx, x_max-dx/2.0+nghost_right*dx, N+nghost) # Centers
            y = np.linspace(y_min-nghost_left*dx, y_max + nghost_right*dy, N+1+nghost) # Edges

            # edges
            Xv, Yv = np.meshgrid(x, y,indexing='ij')

            if transformation == "gnomonic_equiangular":
                edy.X, edy.Y, edy.Z, edy.lon, edy.lat = equiangular_gnomonic_map(x, y, N+nghost, N+nghost+1, self.R)
                tg_ex_edy.X, tg_ex_edy.Y, tg_ex_edy.Z = equiangular_tg_xdir(x, y, N+nghost, N+nghost+1, self.R)
                tg_ey_edy.X, tg_ey_edy.Y, tg_ey_edy.Z = equiangular_tg_ydir(x, y, N+nghost, N+nghost+1, self.R)
                #tg_ex_edy2 = point(N+nghost, N+1+nghost)
                #tg_ey_edy2 = point(N+nghost, N+1+nghost)
                #tg_ex_edy2, tg_ey_edy2 = tg_vector_geodesic_edy_midpoints(edy, vertices, N, nghost)
            elif transformation=="gnomonic_equidistant":
                edy.X, edy.Y, edy.Z, edy.lon, edy.lat = equidistant_gnomonic_map(x, y, N+nghost, N+nghost+1, self.R)
                tg_ex_edy.X, tg_ex_edy.Y, tg_ex_edy.Z = equidistant_tg_xdir(x, y, N+nghost, N+nghost+1, self.R)
                tg_ey_edy.X, tg_ey_edy.Y, tg_ey_edy.Z = equidistant_tg_ydir(x, y, N+nghost, N+nghost+1, self.R)
                #tg_ex_edy2 = point(N+nghost, N+1+nghost)
                #tg_ey_edy2 = point(N+nghost, N+1+nghost)
                #tg_ex_edy2, tg_ey_edy2 = tg_vector_geodesic_edy_midpoints(edy, vertices, N, nghost)
            elif transformation=="conformal":
                edy.X[i0:iend,j0:jend+1], edy.Y[i0:iend,j0:jend+1], edy.Z[i0:iend,j0:jend+1], edy.lon[i0:iend,j0:jend+1], edy.lat[i0:iend,j0:jend+1] \
                = conformal_map(x[i0:iend], y[j0:jend+1], N, N+1)
            self.edy = edy

            # Normalize the tangent vectors in x dir
            norm = tg_ex_edy.X*tg_ex_edy.X + tg_ex_edy.Y*tg_ex_edy.Y + tg_ex_edy.Z*tg_ex_edy.Z
            norm = np.sqrt(norm)
            tg_ex_edy.X = tg_ex_edy.X#/norm
            tg_ex_edy.Y = tg_ex_edy.Y#/norm
            tg_ex_edy.Z = tg_ex_edy.Z#/norm

            # Normalize the tangent vectors in y dir
            norm = tg_ey_edy.X*tg_ey_edy.X + tg_ey_edy.Y*tg_ey_edy.Y + tg_ey_edy.Z*tg_ey_edy.Z
            norm = np.sqrt(norm)
            tg_ey_edy.X = tg_ey_edy.X#/norm
            tg_ey_edy.Y = tg_ey_edy.Y#/norm
            tg_ey_edy.Z = tg_ey_edy.Z#/norm

            #print(np.amax(abs(tg_ex_edy2.X-tg_ex_edy.X)))
            #print(np.amax(abs(tg_ex_edy2.Y-tg_ex_edy.Y)))
            #print(np.amax(abs(tg_ex_edy2.Z-tg_ex_edy.Z)))
            #print(np.amax(abs(tg_ey_edy2.X-tg_ey_edy.X)))
            #print(np.amax(abs(tg_ey_edy2.Y-tg_ey_edy.Y)))
            #print(np.amax(abs(tg_ey_edy2.Z-tg_ey_edy.Z)))

            # Metric tensor on edges in y direction
            metric_tensor_edy = np.zeros((N+nghost, N+nghost+1, nbfaces))
            metric_tensor_edy[:,:,0] = metric_tensor(x, y, self.R, transformation)
            for p in range(1, nbfaces): metric_tensor_edy[:,:,p] = metric_tensor_edy[:,:,0]
            self.metric_tensor_edy = metric_tensor_edy

            # Compute cell lenghts
            if showonscreen==True:
                print("Computing cell lengths...")

            # Compute the geodesic distance of cell edges in x direction
            # Given points
            p1 = [vertices.X[0:N+nghost,:,:], vertices.Y[0:N+nghost  ,:,:], vertices.Z[0:N+nghost  ,:,:]]
            p2 = [vertices.X[1:N+nghost+1,:,:], vertices.Y[1:N+nghost+1,:,:], vertices.Z[1:N+nghost+1,:,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost+1)*nbfaces))
            p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost+1)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1, p2)
            d = np.reshape(d,(N+nghost,N+nghost+1,nbfaces))
            length_x = d[:,:,:]
            # Compute the geodesic distance of cell edges in y direction
            # Given points
            p1 = [vertices.X[:,0:N+nghost  ,:], vertices.Y[:,0:N+nghost  ,:], vertices.Z[:,0:N+nghost  ,:]]
            p2 = [vertices.X[:,1:N+nghost+1,:], vertices.Y[:,1:N+nghost+1,:], vertices.Z[:,1:N+nghost+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost+1)*nbfaces))
            p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost+1)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+nghost+1,N+nghost,nbfaces))
            length_y = d[:,:,:]

            # Cell diagonal length
            # Given points
            p1 = [vertices.X[0:N+nghost,1:N+nghost+1,:], vertices.Y[0:N+nghost,1:N+nghost+1,:], vertices.Z[0:N+nghost,1:N+nghost+1,:]]
            p2 = [vertices.X[1:N+nghost+1,0:N+nghost,:], vertices.Y[1:N+nghost+1,0:N+nghost,:], vertices.Z[1:N+nghost+1,0:N+nghost,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost)*nbfaces))
            p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+nghost,N+nghost,nbfaces))
            length_diag = d[:,:,:]

            # Compute cell areas
            # Each edge of a cell is identified as below
            #    D---------C
            #    |         |
            #    |         |
            #    |         |
            #    |         |
            #    A---------B
            #
            # Compute the area of triangle ABD
            a = length_x[:,0:N+nghost,:] # AB
            b = length_y[0:N+nghost,:,:] # AD
            c = length_diag            # DB
            areas = sphgeo.tri_area(a, b, c)

            # Compute the area of triangle DCB
            a = length_x[:,1:N+nghost+1,:] # DC
            b = length_y[1:N+nghost+1,:,:] # CB
            c = length_diag              # DB
            areas = areas + sphgeo.tri_area(a, b, c)

            # Compute areas
            if showonscreen==True:
                print("Computing cell areas...")
            self.areas = (self.R**2)*areas

            # Cell edx length
            # Given points
            p1 = [edx.X[0:N+nghost,0:N+nghost,:]  , edx.Y[0:N+nghost,0:N+nghost,:]  , edx.Z[0:N+nghost,0:N+nghost,:]]
            p2 = [edx.X[1:N+nghost+1,0:N+nghost,:], edx.Y[1:N+nghost+1,0:N+nghost,:], edx.Z[1:N+nghost+1,0:N+nghost,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost)*nbfaces))
            p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+nghost,N+nghost,nbfaces))

            self.length_edx = self.R*d
            #print(np.amin(self.length_edx),np.amax(self.length_edx))

            # Cell edy length
            # Given points
            p1 = [edy.X[0:N+nghost,0:N+nghost,:]  , edy.Y[0:N+nghost,0:N+nghost,:]  , edy.Z[0:N+nghost,0:N+nghost,:]]
            p2 = [edy.X[0:N+nghost,1:N+nghost+1,:], edy.Y[0:N+nghost,1:N+nghost+1,:], edy.Z[0:N+nghost,1:N+nghost+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost)*nbfaces))
            p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+nghost,N+nghost,nbfaces))

            self.length_edy = self.R*d
            #print(np.amin(self.length_edy),np.amax(self.length_edy))

            # Generate tangent vectors
            if showonscreen==True:
                print("Generating latlon tangent vectors...")

            # Lat-lon tangent unit vectors
            self.elon_edx = sphgeo.tangent_geo_lon(edx.lon)
            self.elat_edx = sphgeo.tangent_geo_lat(edx.lon, edx.lat)
            self.elon_edy = sphgeo.tangent_geo_lon(edy.lon)
            self.elat_edy = sphgeo.tangent_geo_lat(edy.lon, edy.lat)

            # CS map tangent unit vectors at edges points
            # latlon coordinates
            tg_ex_edx.lon = tg_ex_edx.X[:,:,:]*self.elon_edx[:,:,:,0] + tg_ex_edx.Y[:,:,:]*self.elon_edx[:,:,:,1] + tg_ex_edx.Z[:,:,:]*self.elon_edx[:,:,:,2]
            tg_ex_edx.lat = tg_ex_edx.X[:,:,:]*self.elat_edx[:,:,:,0] + tg_ex_edx.Y[:,:,:]*self.elat_edx[:,:,:,1] + tg_ex_edx.Z[:,:,:]*self.elat_edx[:,:,:,2]
            #norm =  tg_ex_edx.lon* tg_ex_edx.lon +  tg_ex_edx.lat*tg_ex_edx.lat
            #print(norm)
            tg_ex_edy.lon = tg_ex_edy.X[:,:,:]*self.elon_edy[:,:,:,0] + tg_ex_edy.Y[:,:,:]*self.elon_edy[:,:,:,1] + tg_ex_edy.Z[:,:,:]*self.elon_edy[:,:,:,2]
            tg_ex_edy.lat = tg_ex_edy.X[:,:,:]*self.elat_edy[:,:,:,0] + tg_ex_edy.Y[:,:,:]*self.elat_edy[:,:,:,1] + tg_ex_edy.Z[:,:,:]*self.elat_edy[:,:,:,2]

            tg_ey_edx.lon = tg_ey_edx.X[:,:,:]*self.elon_edx[:,:,:,0] + tg_ey_edx.Y[:,:,:]*self.elon_edx[:,:,:,1] + tg_ey_edx.Z[:,:,:]*self.elon_edx[:,:,:,2]
            tg_ey_edx.lat = tg_ey_edx.X[:,:,:]*self.elat_edx[:,:,:,0] + tg_ey_edx.Y[:,:,:]*self.elat_edx[:,:,:,1] + tg_ey_edx.Z[:,:,:]*self.elat_edx[:,:,:,2]

            tg_ey_edy.lon = tg_ey_edy.X[:,:,:]*self.elon_edy[:,:,:,0] + tg_ey_edy.Y[:,:,:]*self.elon_edy[:,:,:,1] + tg_ey_edy.Z[:,:,:]*self.elon_edy[:,:,:,2]
            tg_ey_edy.lat = tg_ey_edy.X[:,:,:]*self.elat_edy[:,:,:,0] + tg_ey_edy.Y[:,:,:]*self.elat_edy[:,:,:,1] + tg_ey_edy.Z[:,:,:]*self.elat_edy[:,:,:,2]

            self.tg_ex_edx = tg_ex_edx
            self.tg_ey_edx = tg_ey_edx
            self.tg_ex_edy = tg_ex_edy
            self.tg_ey_edy = tg_ey_edy

            # Latlon/Contravariant conversion
            if showonscreen==True:
                print("Generating latlon/contravariant conversion variables... \n")

            # Latlon/Contravariant conversion  at edx
            prod_ex_elon_edx = tg_ex_edx.X*self.elon_edx[:,:,:,0] + tg_ex_edx.Y*self.elon_edx[:,:,:,1] + tg_ex_edx.Z*self.elon_edx[:,:,:,2]
            prod_ex_elat_edx = tg_ex_edx.X*self.elat_edx[:,:,:,0] + tg_ex_edx.Y*self.elat_edx[:,:,:,1] + tg_ex_edx.Z*self.elat_edx[:,:,:,2]
            prod_ey_elon_edx = tg_ey_edx.X*self.elon_edx[:,:,:,0] + tg_ey_edx.Y*self.elon_edx[:,:,:,1] + tg_ey_edx.Z*self.elon_edx[:,:,:,2]
            prod_ey_elat_edx = tg_ey_edx.X*self.elat_edx[:,:,:,0] + tg_ey_edx.Y*self.elat_edx[:,:,:,1] + tg_ey_edx.Z*self.elat_edx[:,:,:,2]
            determinant_ll2contra_edx = prod_ex_elon_edx*prod_ey_elat_edx - prod_ey_elon_edx*prod_ex_elat_edx

            self.prod_ex_elon_edx = prod_ex_elon_edx
            self.prod_ex_elat_edx = prod_ex_elat_edx
            self.prod_ey_elon_edx = prod_ey_elon_edx
            self.prod_ey_elat_edx = prod_ey_elat_edx
            self.determinant_ll2contra_edx = determinant_ll2contra_edx

            # Latlon/Contravariant conversion at edy
            prod_ex_elon_edy = tg_ex_edy.X*self.elon_edy[:,:,:,0] + tg_ex_edy.Y*self.elon_edy[:,:,:,1] + tg_ex_edy.Z*self.elon_edy[:,:,:,2]
            prod_ex_elat_edy = tg_ex_edy.X*self.elat_edy[:,:,:,0] + tg_ex_edy.Y*self.elat_edy[:,:,:,1] + tg_ex_edy.Z*self.elat_edy[:,:,:,2]
            prod_ey_elon_edy = tg_ey_edy.X*self.elon_edy[:,:,:,0] + tg_ey_edy.Y*self.elon_edy[:,:,:,1] + tg_ey_edy.Z*self.elon_edy[:,:,:,2]
            prod_ey_elat_edy = tg_ey_edy.X*self.elat_edy[:,:,:,0] + tg_ey_edy.Y*self.elat_edy[:,:,:,1] + tg_ey_edy.Z*self.elat_edy[:,:,:,2]
            determinant_ll2contra_edy = prod_ex_elon_edy*prod_ey_elat_edy - prod_ey_elon_edy*prod_ex_elat_edy

            self.prod_ex_elon_edy = prod_ex_elon_edy
            self.prod_ex_elat_edy = prod_ex_elat_edy
            self.prod_ey_elon_edy = prod_ey_elon_edy
            self.prod_ey_elat_edy = prod_ey_elat_edy
            self.determinant_ll2contra_edy = determinant_ll2contra_edy
            #tg_ex_edx, tg_ey_edx
            #tg_ex_edy, tg_ey_edy
            #self.elon_edx, self.elat_edx
            #self.elon_edy, self.elat_edy

            Xu = np.repeat(Xu[:, :, np.newaxis], 6, axis=2)
            Yv = np.repeat(Yv[:, :, np.newaxis], 6, axis=2)
            self.Xu, self.Yv = Xu, Yv

            # Finish time counting
            elapsed_time = time.time() - start_time

            if showonscreen==True:
                # Print some grid properties
                print("\nMin  edge length (km)  : ","{:.2e}".format(np.amin(self.length_edx[i0:iend+1,j0:jend,:])*erad/10**3))
                print("Max  edge length (km)  : ","{:.2e}".format(np.amax(self.length_edx[i0:iend+1,j0:jend,:])*erad/10**3))
                print("Mean edge length (km)  : ","{:.2e}".format(np.mean(self.length_edx[i0:iend+1,j0:jend,:])*erad/10**3))
                print("Ratio max/min length   : ","{:.2e}".format(np.amax(self.length_edx[i0:iend+1,j0:jend,:])/np.amin(self.length_edx[i0:iend+1,j0:jend+1,:])))

                print("Min  area (km2)        : ","{:.2e}".format(np.amin(self.areas[i0:iend,j0:jend,:])*erad*erad/10**6))
                print("Max  area (km2)        : ","{:.2e}".format(np.amax(self.areas[i0:iend,j0:jend,:])*erad*erad/10**6))
                print("Mean area (km2)        : ","{:.2e}".format(np.mean(self.areas[i0:iend,j0:jend,:])*erad*erad/10**6))
                print("Ratio max/min area     : ","{:.2e}".format(np.amax(self.areas[i0:iend,j0:jend,:])/np.amin(self.areas[i0:iend,j0:jend,:])))

                #m = N
                #ratio = (1+2*np.tan((pi/4))**2 * (1-1/m))**1.5
                #ratio = ratio*np.cos((pi/4))**4*(1-1/m)
                #print(ratio)
                #exit()
            if showonscreen==True:
                print("\nDone in ","{:.2e}".format(elapsed_time),"seconds.")
                print("--------------------------------------------------------\n")

####################################################################################
# Structure for scalar values on cubed-sphere grid
####################################################################################
class scalar_field:
    def __init__(self, grid, name, position):
        # Field name
        self.name = name

        # Field position
        self.position = position

        # Number of values in a coordinate axis (same for all panels)
        if position == "vertex":
            self.N = grid.N+1
        elif position == "center":
            self.N = grid.N
        else:
            print("ERROR in scalar_field: invalid position.")
            exit()

        # Field values along a coordinate axis (same for all panels)
        self.f = np.zeros((self.N, self.N, nbfaces))

####################################################################################
# Structure for lat-lon grid
####################################################################################
class latlon_grid:
    def __init__(self, Nlat, Nlon):
        self.Nlat, self.Nlon = Nlat, Nlon

        # Create a latlon grid
        lat = np.linspace(-pio2, pio2, Nlat)
        lon = np.linspace(  -pi,   pi, Nlon)
        self.lat, self.lon = np.meshgrid(lat, lon)

        # Convert to cartesian coordinates
        self.X, self.Y, self.Z = sphgeo.sph2cart(self.lon, self.lat)

        #Lat-lon to cubed sphere indexes
        self.ix, self.jy, self.mask = [], [], []

####################################################################################
# Compute execution time of the grid generation for different resolutions
####################################################################################
def cs_time_test():
    import time
    from matplotlib import pyplot as plt

    # Values of N
    Ns = (1,10,100,1000)

    # Number we generate the grid to compute the execution time
    Ntest = 10

    # Vector that save the execution time for each N
    times = np.zeros(len(Ns))

    # Calculate the average execution time using Ntest executions
    k = 0
    for N in Ns:
        for t in range(0,Ntest):
            start_time = time.time()
            grid = cubed_sphere(N)
            elapsed_time = time.time() - start_time
            times[k] = times[k] + elapsed_time/Ntest
            print(N,t,elapsed_time)
        k = k + 1
    print("N and average time (in seconds):")
    k = 0

    # Show the value of N and execution time on the screen
    for k in range(0,len(Ns)):
        print(Ns[k], times[k])

    print('\n')
    for k in range(1,len(Ns)):
        print(Ns[k], times[k]/times[k-1])

    # Plot the graph in logscale
    plt.figure()
    plt.loglog(Ns, times,marker='x')
    plt.Ylabel('Time (seconds)', fontsize=12)
    plt.Xlabel('$N$', fontsize=12)
    plt.savefig(graphdir+'execution_time'+'.eps', format='eps')
    plt.show()
    plt.close()



####################################################################################
#  Parabola class
####################################################################################
class ppm_parabola:
    def __init__(self, cs_grid, simulation, direction):
        # Number of cells
        N  = cs_grid.N
        M  = N
        ng = cs_grid.nghost

        # reconstruction name
        self.recon_name = simulation.recon_name

        # parabola coefficients
        # Notation from Colella and  Woodward 1984
        # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
        self.q_L = np.zeros((N+ng, M+ng, nbfaces))
        self.q_R = np.zeros((N+ng, M+ng, nbfaces))
        self.dq  = np.zeros((N+ng, M+ng, nbfaces))
        self.q6  = np.zeros((N+ng, M+ng, nbfaces))

        if direction == 'x':
            # parabola fluxes
            self.f_L = np.zeros((N+ng+1, M+ng, nbfaces))  # flux from left
            self.f_R = np.zeros((N+ng+1, M+ng, nbfaces))  # flux from right
            self.f_upw = np.zeros((N+ng+1, M+ng, nbfaces)) # upwind flux

            # Extra variables for each scheme
            if simulation.recon_name == 'PPM-0' or simulation.recon_name == 'PPM-CW84' or simulation.recon_name == 'PPM-L04':
                self.Q_edges =  np.zeros((N+ng+1, M+ng, nbfaces))
        elif direction == 'y':
            # parabola fluxes
            self.f_L = np.zeros((N+ng, M+ng+1, nbfaces))   # flux from left
            self.f_R = np.zeros((N+ng, M+ng+1, nbfaces))   # flux from right
            self.f_upw = np.zeros((N+ng, M+ng+1, nbfaces)) # upwind flux

            # Extra variables for each scheme
            if simulation.recon_name == 'PPM-0' or simulation.recon_name == 'PPM-CW84' or simulation.recon_name == 'PPM-L04':
                self.Q_edges =  np.zeros((N+ng, M+ng+1, nbfaces))

        self.dF = np.zeros((N+ng, M+ng, nbfaces)) # div flux

        if simulation.recon_name == 'PPM-CW84':
            self.dQ  = np.zeros((N+ng, M+ng, nbfaces))
            self.dQ0 = np.zeros((N+ng, M+ng, nbfaces))
            self.dQ1 = np.zeros((N+ng, M+ng, nbfaces))
            self.dQ2 = np.zeros((N+ng, M+ng, nbfaces))

        if simulation.recon_name == 'PPM-L04':
            self.dQ      = np.zeros((N+ng, M+ng, nbfaces))
            self.dQ_min  = np.zeros((N+ng, M+ng, nbfaces))
            self.dQ_max  = np.zeros((N+ng, M+ng, nbfaces))
            self.dQ_mono = np.zeros((N+ng, M+ng, nbfaces))

####################################################################################
#  Velocity class
#  The quadrilateral points are labeled as below
#
#  po-------pv--------po
#  |                  |
#  |                  |
#  |                  |
#  pu       pc        pu
#  |                  |
#  |                  |
#  |                  |
#  po--------pv-------po
#
####################################################################################
class velocity_edges:
    def __init__(self, cs_grid, pos):
        N = cs_grid.N
        ng = cs_grid.nghost

        if pos == 'pu':
            self.ulon = np.zeros((N+1+ng, N+ng, nbfaces))
            self.vlat = np.zeros((N+1+ng, N+ng, nbfaces))
            self.ucontra = np.zeros((N+1+ng, N+ng, nbfaces))
            self.vcontra = np.zeros((N+1+ng, N+ng, nbfaces))
            self.ucontra_averaged = np.zeros((N+1+ng, N+ng, nbfaces)) # used for departure point
            self.ucontra_old      = np.zeros((N+1+ng, N+ng, nbfaces)) # used for departure point
        elif pos == 'pv':
            self.ulon = np.zeros((N+ng, N+1+ng, nbfaces))
            self.vlat = np.zeros((N+ng, N+1+ng, nbfaces))
            self.ucontra = np.zeros((N+ng, N+1+ng, nbfaces))
            self.vcontra = np.zeros((N+ng, N+1+ng, nbfaces))
            self.vcontra_averaged = np.zeros((N+ng, N+1+ng, nbfaces)) # used for departure point
            self.vcontra_old      = np.zeros((N+ng, N+1+ng, nbfaces)) # used for departure point
        else:
            print('ERROR in  velocity class: invalid position, ', pos)

