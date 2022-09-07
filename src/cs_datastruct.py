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
import sphgeo
import time
import os.path
import netCDF4 as nc
####################################################################################
#  General spherical points structure
####################################################################################
class point:
    def __init__(self, N, M):
        # Cartesian coordinates - represented in capital letters (X,Y,Z)
        self.X   = np.zeros((N, M, nbfaces))
        self.Y   = np.zeros((N, M, nbfaces))
        self.Z   = np.zeros((N, M, nbfaces))

        # Spherical coordinates
        self.lat = np.zeros((N, M, nbfaces))
        self.lon = np.zeros((N, M, nbfaces))

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
        self.R = erad
        #self.R = 1.0

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

            # Create points
            vertices = point(N+1, N+1)
            centers  = point(N, N)

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

            # Geometric properties
            areas           = griddata['areas'][:,:,:]
            length_x        = griddata['length_x'][:,:,:]
            length_y        = griddata['length_y'][:,:,:]
            length_diag     = griddata['length_diag'][:,:,:]
            length_antidiag = griddata['length_antidiag'][:,:,:]
            angles          = griddata['angles'][:,:,:,:]

            # Attributes
            self.centers         = centers
            self.vertices        = vertices
            self.length_x        = length_x
            self.length_y        = length_y
            self.length_diag     = length_diag
            self.length_antidiag = length_antidiag
            self.angles          = angles
            self.areas           = areas

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

            # Ghost cells for each panel
            nghost_left  = 2
            nghost_right = 3
            nghost = nghost_right + nghost_left
            self.nghost_left  = nghost_left
            self.nghost_right = nghost_right 
            self.nghost = nghost
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
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = conformal_map(x, y, N+1+nghost, N+1+nghost)
            self.vertices = vertices

            # Metric tensor on vertices
            metric_tensor_vertices = np.zeros((N+1+nghost, N+1+nghost, nbfaces))
            metric_tensor_vertices[:,:,0] = metric_tensor(x, y, self.R, transformation)  
            for p in range(1, nbfaces): metric_tensor_vertices[:,:,p] = metric_tensor_vertices[:,:,0]  
            self.metric_tensor_vertices = metric_tensor_vertices

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
                centers.X, centers.Y, centers.Z, centers.lon, centers.lat = conformal_map(x, y, N+nghost, N+nghost)
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
            if transformation == "gnomonic_equiangular":
                edx.X, edx.Y, edx.Z, edx.lon, edx.lat = equiangular_gnomonic_map(x, y, N+1+nghost, N+nghost, self.R)
                tg_ex_edx.X, tg_ex_edx.Y, tg_ex_edx.Z = equiangular_tg_xdir(x, y, N+1+nghost, N+nghost, self.R)
                tg_ey_edx.X, tg_ey_edx.Y, tg_ey_edx.Z = equiangular_tg_ydir(x, y, N+1+nghost, N+nghost, self.R)
            elif transformation=="gnomonic_equidistant":
                edx.X, edx.Y, edx.Z, edx.lon, edx.lat = equidistant_gnomonic_map(x, y, N+1+nghost, N+nghost, self.R)
                tg_ex_edx.X, tg_ex_edx.Y, tg_ex_edx.Z = equidistant_tg_xdir(x, y, N+1+nghost, N+nghost, self.R)
                tg_ey_edx.X, tg_ey_edx.Y, tg_ey_edx.Z = equidistant_tg_ydir(x, y, N+1+nghost, N+nghost, self.R)
            elif transformation=="conformal":
                edx.X, edx.Y, edx.Z, edx.lon, edx.lat = conformal_map(x, y, N+1+nghost, N+nghost)

            self.edx = edx

            # Normalize the tangent vectors in x dir
            norm = tg_ex_edx.X*tg_ex_edx.X + tg_ex_edx.Y*tg_ex_edx.Y + tg_ex_edx.Z*tg_ex_edx.Z
            norm = np.sqrt(norm)
            tg_ex_edx.X = tg_ex_edx.X/norm
            tg_ex_edx.Y = tg_ex_edx.Y/norm
            tg_ex_edx.Z = tg_ex_edx.Z/norm

            # Normalize the tangent vectors in y dir
            norm = tg_ey_edx.X*tg_ey_edx.X + tg_ey_edx.Y*tg_ey_edx.Y + tg_ey_edx.Z*tg_ey_edx.Z
            norm = np.sqrt(norm)
            tg_ey_edx.X = tg_ey_edx.X/norm
            tg_ey_edx.Y = tg_ey_edx.Y/norm
            tg_ey_edx.Z = tg_ey_edx.Z/norm

            # Unit tangent vectors in x dir
            tg_ex_edx2 = point(N+1+nghost, N+nghost)
            P = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))
            Q = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))
            P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edx.X, edx.Y, edx.Z
            Q[0:N+nghost,:,:,0], Q[0:N+nghost,:,:,1], Q[0:N+nghost,:,:,2] = edx.X[0:N+nghost,:,:]-edx.X[1:N+nghost+1,:,:], edx.Y[0:N+nghost,:,:]-edx.Y[1:N+nghost+1,:,:], edx.Z[0:N+nghost,:,:]-edx.Z[1:N+nghost+1,:,:]
            Q[N+nghost,:,:,0], Q[N+nghost,:,:,1], Q[N+nghost,:,:,2] = edx.X[N+nghost-1,:,:]-edx.X[N+nghost,:,:], edx.Y[N+nghost-1,:,:]-edx.Y[N+nghost,:,:], edx.Z[N+nghost-1,:,:]-edx.Z[N+nghost,:,:]

            tg_ex_edx2.X, tg_ex_edx2.Y, tg_ex_edx2.Z = sphgeo.tangent_projection(P, Q)
            norm = tg_ex_edx2.X*tg_ex_edx2.X + tg_ex_edx2.Y*tg_ex_edx2.Y + tg_ex_edx2.Z*tg_ex_edx2.Z
            norm = np.sqrt(norm)
            tg_ex_edx2.X = tg_ex_edx2.X/norm
            tg_ex_edx2.Y = tg_ex_edx2.Y/norm
            tg_ex_edx2.Z = tg_ex_edx2.Z/norm

            #print(np.amax(abs(tg_ex_edx2.X-tg_ex_edx.X)))
            #print(np.amax(abs(tg_ex_edx2.Y-tg_ex_edx.Y)))
            #print(np.amax(abs(tg_ex_edx2.Z-tg_ex_edx.Z)))
            
            tg_ex_edx.X = tg_ex_edx2.X
            tg_ex_edx.Y = tg_ex_edx2.Y
            tg_ex_edx.Z = tg_ex_edx2.Z

            # Unit tangent vectors in y dir
            tg_ey_edx2 = point(N+1+nghost, N+nghost)
            P = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))
            Q = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))
            P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edx.X, edx.Y, edx.Z
            Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2] = edx.X[:,:,:]-vertices.X[:,1:N+nghost+1,:], edx.Y[:,:,:]-vertices.Y[:,1:N+nghost+1,:], edx.Z[:,:,:]-vertices.Z[:,1:N+nghost+1,:]

            tg_ey_edx2.X, tg_ey_edx2.Y, tg_ey_edx2.Z = sphgeo.tangent_projection(P, Q)
            norm = tg_ey_edx2.X*tg_ey_edx2.X + tg_ey_edx2.Y*tg_ey_edx2.Y + tg_ey_edx2.Z*tg_ey_edx2.Z
            norm = np.sqrt(norm)
            tg_ey_edx2.X = tg_ey_edx2.X/norm
            tg_ey_edx2.Y = tg_ey_edx2.Y/norm
            tg_ey_edx2.Z = tg_ey_edx2.Z/norm

            #print(np.amax(abs(tg_ey_edx2.X-tg_ey_edx.X)))
            #print(np.amax(abs(tg_ey_edx2.Y-tg_ey_edx.Y)))
            #print(np.amax(abs(tg_ey_edx2.Z-tg_ey_edx.Z)))

            tg_ey_edx.X = tg_ey_edx2.X
            tg_ey_edx.Y = tg_ey_edx2.Y
            tg_ey_edx.Z = tg_ey_edx2.Z

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
            if transformation == "gnomonic_equiangular":
                edy.X, edy.Y, edy.Z, edy.lon, edy.lat = equiangular_gnomonic_map(x, y, N+nghost, N+nghost+1, self.R)
                tg_ex_edy.X, tg_ex_edy.Y, tg_ex_edy.Z = equiangular_tg_xdir(x, y, N+nghost, N+nghost+1, self.R)
                tg_ey_edy.X, tg_ey_edy.Y, tg_ey_edy.Z = equiangular_tg_ydir(x, y, N+nghost, N+nghost+1, self.R)
            elif transformation=="gnomonic_equidistant":
                edy.X, edy.Y, edy.Z, edy.lon, edy.lat = equidistant_gnomonic_map(x, y, N+nghost, N+nghost+1, self.R)
                tg_ex_edy.X, tg_ex_edy.Y, tg_ex_edy.Z = equidistant_tg_xdir(x, y, N+nghost, N+nghost+1, self.R)
                tg_ey_edy.X, tg_ey_edy.Y, tg_ey_edy.Z = equidistant_tg_ydir(x, y, N+nghost, N+nghost+1, self.R)
            elif transformation=="conformal":
                edy.X, edy.Y, edy.Z, edy.lon, edy.lat = conformal_map(x, y, N+nghost, N+nghost+1)

            self.edy = edy

            # Normalize the tangent vectors in x dir
            norm = tg_ex_edy.X*tg_ex_edy.X + tg_ex_edy.Y*tg_ex_edy.Y + tg_ex_edy.Z*tg_ex_edy.Z
            norm = np.sqrt(norm)
            tg_ex_edy.X = tg_ex_edy.X/norm
            tg_ex_edy.Y = tg_ex_edy.Y/norm
            tg_ex_edy.Z = tg_ex_edy.Z/norm

            # Normalize the tangent vectors in y dir
            norm = tg_ey_edy.X*tg_ey_edy.X + tg_ey_edy.Y*tg_ey_edy.Y + tg_ey_edy.Z*tg_ey_edy.Z
            norm = np.sqrt(norm)
            tg_ey_edy.X = tg_ey_edy.X/norm
            tg_ey_edy.Y = tg_ey_edy.Y/norm
            tg_ey_edy.Z = tg_ey_edy.Z/norm

            # Unit tangent vectors in x dir
            tg_ex_edy2 = point(N+nghost, N+nghost+1)
            P = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))
            Q = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))
            P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edy.X, edy.Y, edy.Z
            Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2] = edy.X-vertices.X[1:N+nghost+1,:,:], edy.Y-vertices.Y[1:N+nghost+1,:,:], edy.Z-vertices.Z[1:N+nghost+1,:,:]
            tg_ex_edy2.X, tg_ex_edy2.Y, tg_ex_edy2.Z = sphgeo.tangent_projection(P, Q)
            norm = tg_ex_edy2.X*tg_ex_edy2.X + tg_ex_edy2.Y*tg_ex_edy2.Y + tg_ex_edy2.Z*tg_ex_edy2.Z
            norm = np.sqrt(norm)
            tg_ex_edy2.X = tg_ex_edy2.X/norm
            tg_ex_edy2.Y = tg_ex_edy2.Y/norm
            tg_ex_edy2.Z = tg_ex_edy2.Z/norm

            #print(np.amax(abs(tg_ex_edy2.X-tg_ex_edy.X)))
            #print(np.amax(abs(tg_ex_edy2.Y-tg_ex_edy.Y)))
            #print(np.amax(abs(tg_ex_edy2.Z-tg_ex_edy.Z)))

            tg_ex_edy.X = tg_ex_edy2.X
            tg_ex_edy.Y = tg_ex_edy2.Y
            tg_ex_edy.Z = tg_ex_edy2.Z

            # Unit tangent vectors in y dir
            tg_ey_edy2 = point(N+nghost, N+nghost+1)
            P = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))
            Q = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))
            P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edy.X, edy.Y, edy.Z
            Q[:,0:N+nghost,:,0], Q[:,0:N+nghost,:,1], Q[:,0:N+nghost,:,2] = edy.X[:,0:N+nghost,:]-edy.X[:,1:N+nghost+1,:], edy.Y[:,0:N+nghost,:]-edy.Y[:,1:N+nghost+1,:], edy.Z[:,0:N+nghost,:]-edy.Z[:,1:N+nghost+1,:]
            Q[:,N+nghost,:,0], Q[:,N+nghost,:,1], Q[:,N+nghost,:,2] = edy.X[:,N+nghost-1,:]-edy.X[:,N+nghost,:], edy.Y[:,N+nghost-1,:]-edy.Y[:,N+nghost,:], edy.Z[:,N+nghost-1,:]-edy.Z[:,N+nghost,:]

            tg_ey_edy2.X, tg_ey_edy2.Y, tg_ey_edy2.Z = sphgeo.tangent_projection(P, Q)
            norm = tg_ey_edy2.X*tg_ey_edy2.X + tg_ey_edy2.Y*tg_ey_edy2.Y + tg_ey_edy2.Z*tg_ey_edy2.Z
            norm = np.sqrt(norm)
            tg_ey_edy2.X = tg_ey_edy2.X/norm
            tg_ey_edy2.Y = tg_ey_edy2.Y/norm
            tg_ey_edy2.Z = tg_ey_edy2.Z/norm

            #print(np.amax(abs(tg_ey_edy2.X-tg_ey_edy.X)))
            #print(np.amax(abs(tg_ey_edy2.Y-tg_ey_edy.Y)))
            #print(np.amax(abs(tg_ey_edy2.Z-tg_ey_edy.Z)))

            tg_ey_edy.X = tg_ey_edy2.X
            tg_ey_edy.Y = tg_ey_edy2.Y
            tg_ey_edy.Z = tg_ey_edy2.Z

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
            self.length_x = d
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
            self.length_y = d

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
            self.length_diag = d


            # Cell antidiagonal length
            # Given points
            p1 = [vertices.X[0:N+nghost,0:N+nghost,:], vertices.Y[0:N+nghost,0:N+nghost,:], vertices.Z[0:N+nghost,0:N+nghost,:]]
            p2 = [vertices.X[1:N+nghost+1,1:N+nghost+1,:], vertices.Y[1:N+nghost+1,1:N+nghost+1,:], vertices.Z[1:N+nghost+1,1:N+nghost+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+nghost)*(N+nghost)*nbfaces))
            p2 = np.reshape(p2,(3,(N+nghost)*(N+nghost)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+nghost,N+nghost,nbfaces))
            self.length_antidiag = d
            # Compute cell angles
            # Each angle of a cell is identified as below
            #    D---------C
            #    |         |
            #    |         |
            #    |         |
            #    |         |
            #    A---------B
            #
            if showonscreen==True:
                print("Computing cell angles...")
            angles = np.zeros((N+nghost, N+nghost, nbfaces, 4))

            # Compute the angle A using the triangle ABD
            a = self.length_x[:,0:N+nghost,:] # AB
            b = self.length_y[0:N+nghost,:,:] # AD
            c = self.length_diag            # DB
            angles[:,:,:,0] = sphgeo.tri_angle(a, b, c)
            areas = sphgeo.tri_area(a, b, c)

            # Compute the angle B using the triangle ABC
            a = self.length_x[:,0:N+nghost,:]   # AB
            b = self.length_y[1:N+nghost+1,:,:] # BC
            c = self.length_antidiag          # AB
            angles[:,:,:,1] = sphgeo.tri_angle(a, b, c)

            # Compute the angle C using the triangle DCB
            a = self.length_x[:,1:N+nghost+1,:] # DC
            b = self.length_y[1:N+nghost+1,:,:] # CB
            c = self.length_diag              # DB
            angles[:,:,:,2] = sphgeo.tri_angle(a, b, c)
            areas = areas + sphgeo.tri_area(a, b, c)
      
            # Compute the angle D using the triangle ADC
            a = self.length_x[:,1:N+nghost+1,:] # DC
            b = self.length_y[0:N+nghost,:,:]   # AD
            c = self.length_antidiag          # CA
            angles[:,:,:,3] = sphgeo.tri_angle(a, b, c)

            # Angles attribute
            self.angles = angles

            # Compute areas
            if showonscreen==True:
                print("Computing cell areas...")    
            self.areas = (self.R**2)*areas
            self.length_x = self.R*self.length_x
            self.length_y = self.R*self.length_y
            # Use spherical excess formula
            #self.areas = sphgeo.quad_area(angles[:,:,:,0], angles[:,:,:,1], angles[:,:,:,2], angles[:,:,:,3])
            #self.areas = areas
            #x = np.linspace(x_min, x_max, N+1)
            #y = np.linspace(y_min, y_max, N+1)
            #x, y = np.meshgrid(x, y, indexing='ij')
            #areas2 = np.zeros((N, N, nbfaces))
            #areas2[:,:,:]= sphgeo.quad_areas(x, y)

            #print(np.amax(abs(areas2[:,:,0]-areas[:,:,0])/areas[:,:,0]))

            # Generate tangent vectors
            if showonscreen==True:
                print("Generating tangent vectors... \n")  

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
            #exit()
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
            
            # Finish time counting
            elapsed_time = time.time() - start_time
      
        if showonscreen==True:
            # Print some grid properties
            print("\nMin  edge length (km)  : ","{:.2e}".format(np.amin(self.length_x)))
            print("Max  edge length (km)  : ","{:.2e}".format(np.amax(self.length_x)))
            print("Mean edge length (km)  : ","{:.2e}".format(np.mean(self.length_x)))
            print("Ratio max/min length   : ","{:.2e}".format(np.amax(self.length_x)/np.amin(self.length_x)))
         
            print("Min  area (km2)        : ","{:.2e}".format(np.amin(self.areas)))
            print("Max  area (km2)        : ","{:.2e}".format(np.amax(self.areas)))
            print("Mean area (km2)        : ","{:.2e}".format(np.mean(self.areas)))         
            print("Ratio max/min area     : ","{:.2e}".format(np.amax(self.areas)/np.amin(self.areas)))
         
            print("Min  angle (degrees)   : ","{:.2e}".format(np.amin(self.angles*rad2deg)))
            print("Max  angle (degrees)   : ","{:.2e}".format(np.amax(self.angles*rad2deg)))
            print("Mean angle (degrees)   : ","{:.2e}".format(np.mean(self.angles*rad2deg)))
            print("Ratio max/min angle    : ","{:.2e}".format(np.amax(self.angles)/np.amin(self.angles)))
            #m = N
            #ratio = (1+2*np.tan((pi/4))**2 * (1-1/m))**1.5
            #ratio = ratio*np.cos((pi/4))**4*(1-1/m)
            #print(ratio)
            #exit()
        if (showonscreen==True):
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
