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
                a = pio4
            elif transformation == "gnomonic_equidistant":
                a = 1.0/np.sqrt(3.0) # Half length of the cube
            elif transformation == "conformal":
                a = 1.0
            else:
                print("ERROR: invalid grid transformation.")
                exit()

            x_min, x_max, y_min, y_max = [-a, a, -a, a]
      
            # Start time counting
            start_time = time.time()

            # Grid spacing
            dx = (x_max-x_min)/N
            dy = (y_max-y_min)/N
            x = np.linspace(x_min, x_max, N+1) # vertices
            y = np.linspace(y_min, y_max, N+1) # vertices

            # Generate cell vertices
            if showonscreen==True:
                print("Generating cell vertices...")
            
            vertices = point(N+1, N+1)

            if transformation == "gnomonic_equiangular": 
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = equiangular_gnomonic_map(x, y, N+1, N+1)

            elif transformation=="gnomonic_equidistant":
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = equidistant_gnomonic_map(x, y, N+1, N+1)

            elif transformation=="conformal":
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = conformal_map(x, y, N+1, N+1)
            self.vertices = vertices
 
            # Generate cell centers
            if showonscreen==True:
                print("Generating cell centers...")  

            centers = point(N, N)

            x = np.linspace(x_min+dx/2.0, x_max-dx/2.0, N) # Centers
            y = np.linspace(y_min+dy/2.0, y_max-dy/2.0, N) # Centers

            if transformation == "gnomonic_equiangular":
                centers.X, centers.Y, centers.Z, centers.lon, centers.lat = equiangular_gnomonic_map(x, y, N, N)

            elif transformation=="gnomonic_equidistant":
                centers.X, centers.Y, centers.Z, centers.lon, centers.lat = equidistant_gnomonic_map(x, y, N, N)

            elif transformation=="conformal":
                centers.X, centers.Y, centers.Z, centers.lon, centers.lat = conformal_map(x, y, N, N)

            self.centers = centers

            # Generate cell edges in x direction
            if showonscreen==True:
                print("Generating cell edges in x direction...")  

            edges_xdir = point(N+1, N)

            x = np.linspace(x_min, x_max, N+1) # Edges
            y = np.linspace(y_min+dy/2.0, y_max-dy/2.0, N) # Centers

            if transformation == "gnomonic_equiangular":
                edges_xdir.X, edges_xdir.Y, edges_xdir.Z, edges_xdir.lon, edges_xdir.lat = equiangular_gnomonic_map(x, y, N+1, N)

            elif transformation=="gnomonic_equidistant":
                edges_xdir.X, edges_xdir.Y, edges_xdir.Z, edges_xdir.lon, edges_xdir.lat = equidistant_gnomonic_map(x, y, N+1, N)

            elif transformation=="conformal":
                edges_xdir.X, edges_xdir.Y, edges_xdir.Z, edges_xdir.lon, edges_xdir.lat = conformal_map(x, y, N+1, N)

            self.edges_xdir = edges_xdir
            
            # Generate cell edges in y direction
            if showonscreen==True:
                print("Generating cell edges in x direction...")  

            edges_ydir = point(N, N+1)

            x = np.linspace(x_min+dx/2.0, x_max-dx/2.0, N) # Centers
            y = np.linspace(y_min, y_max, N+1) # Edges

            if transformation == "gnomonic_equiangular":
                edges_ydir.X, edges_ydir.Y, edges_ydir.Z, edges_ydir.lon, edges_ydir.lat = equiangular_gnomonic_map(x, y, N, N+1)

            elif transformation=="gnomonic_equidistant":
                edges_ydir.X, edges_ydir.Y, edges_ydir.Z, edges_ydir.lon, edges_ydir.lat = equidistant_gnomonic_map(x, y, N, N+1)

            elif transformation=="conformal":
                edges_ydir.X, edges_ydir.Y, edges_ydir.Z, edges_ydir.lon, edges_ydir.lat = conformal_map(x, y, N, N+1)

            self.edges_ydir = edges_ydir
 
            # Compute cell lenghts
            if showonscreen==True:
                print("Computing cell lengths...")

            # Compute the geodesic distance of cell edges in x direction
            # Given points
            p1 = [vertices.X[0:self.N  ,:,:]  , vertices.Y[0:self.N  ,:,:], vertices.Z[0:self.N  ,:,:]]
            p2 = [vertices.X[1:self.N+1,:,:], vertices.Y[1:self.N+1,:,:], vertices.Z[1:self.N+1,:,:]]

            # Reshape
            p1 = np.reshape(p1,(3,N*(N+1)*nbfaces))
            p2 = np.reshape(p2,(3,N*(N+1)*nbfaces))

            # Compute arclen      
            d = sphgeo.arclen(p1, p2)
            d = np.reshape(d,(N,N+1,nbfaces))
            self.length_x = d

            # Compute the geodesic distance of cell edges in y direction
            # Given points
            p1 = [vertices.X[:,0:self.N  ,:], vertices.Y[:,0:self.N  ,:], vertices.Z[:,0:self.N  ,:]]
            p2 = [vertices.X[:,1:self.N+1,:], vertices.Y[:,1:self.N+1,:], vertices.Z[:,1:self.N+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,N*(N+1)*nbfaces))
            p2 = np.reshape(p2,(3,N*(N+1)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+1,N,nbfaces))
            self.length_y = d

            # Cell diagonal length
            # Given points
            p1 = [vertices.X[0:self.N,1:self.N+1,:], vertices.Y[0:self.N,1:self.N+1,:], vertices.Z[0:self.N,1:self.N+1,:]]
            p2 = [vertices.X[1:self.N+1,0:self.N,:], vertices.Y[1:self.N+1,0:self.N,:], vertices.Z[1:self.N+1,0:self.N,:]]
      
            # Reshape
            p1 = np.reshape(p1,(3,N*N*nbfaces))
            p2 = np.reshape(p2,(3,N*N*nbfaces))
      
            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N,N,nbfaces))
            self.length_diag = d

            # Cell antidiagonal length
            # Given points
            p1 = [vertices.X[0:self.N,0:self.N,:], vertices.Y[0:self.N,0:self.N,:], vertices.Z[0:self.N,0:self.N,:]]
            p2 = [vertices.X[1:self.N+1,1:self.N+1,:], vertices.Y[1:self.N+1,1:self.N+1,:], vertices.Z[1:self.N+1,1:self.N+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,N*N*nbfaces))
            p2 = np.reshape(p2,(3,N*N*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N,N,nbfaces))
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
            angles = np.zeros((self.N, self.N, nbfaces, 4))

            # Compute the angle A using the triangle ABD
            a = self.length_x[:,0:self.N,:] # AB
            b = self.length_y[0:self.N,:,:] # AD
            c = self.length_diag            # DB
            angles[:,:,:,0] = sphgeo.tri_angle(a, b, c)
            areas = sphgeo.tri_area(a, b, c)

            # Compute the angle B using the triangle ABC
            a = self.length_x[:,0:self.N,:]   # AB
            b = self.length_y[1:self.N+1,:,:] # BC
            c = self.length_antidiag          # AB
            angles[:,:,:,1] = sphgeo.tri_angle(a, b, c)

            # Compute the angle C using the triangle DCB
            a = self.length_x[:,1:self.N+1,:] # DC
            b = self.length_y[1:self.N+1,:,:] # CB
            c = self.length_diag              # DB
            angles[:,:,:,2] = sphgeo.tri_angle(a, b, c)
            areas = areas + sphgeo.tri_area(a, b, c)
      
            # Compute the angle D using the triangle ADC
            a = self.length_x[:,1:self.N+1,:] # DC
            b = self.length_y[0:self.N,:,:]   # AD
            c = self.length_antidiag          # CA
            angles[:,:,:,3] = sphgeo.tri_angle(a, b, c)

            # Angles attribute
            self.angles = angles

            # Compute areas
            if showonscreen==True:
                print("Computing cell areas...\n")    

            # Compute areas
            # Use spherical excess formula
            #self.areas = sphgeo.quad_area(angles[:,:,:,0], angles[:,:,:,1], angles[:,:,:,2], angles[:,:,:,3])
            #self.areas = areas
            x = np.linspace(x_min, x_max, N+1)
            y = np.linspace(y_min, y_max, N+1)
            x, y = np.meshgrid(x, y, indexing='ij')
            areas2 = np.zeros((N, N, nbfaces))
            areas2[:,:,:]= sphgeo.quad_areas(x, y)
            self.areas = areas2
            #print(np.amax(abs(areas2[:,:,0]-areas[:,:,0])/areas[:,:,0]))

            # Finish time counting
            elapsed_time = time.time() - start_time
      
        if showonscreen==True:
            # Print some grid properties
            print("\nMin  edge length (km)  : ","{:.2e}".format(erad*np.amin(self.length_x)))
            print("Max  edge length (km)  : ","{:.2e}".format(erad*np.amax(self.length_x)))
            print("Mean edge length (km)  : ","{:.2e}".format(erad*np.mean(self.length_x)))
            print("Ratio max/min length   : ","{:.2e}".format(np.amax(self.length_x)/np.amin(self.length_x)))
         
            print("Min  area (km2)        : ","{:.2e}".format(erad*erad*np.amin(self.areas)))
            print("Max  area (km2)        : ","{:.2e}".format(erad*erad*np.amax(self.areas)))
            print("Mean area (km2)        : ","{:.2e}".format(erad*erad*np.mean(self.areas)))         
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
