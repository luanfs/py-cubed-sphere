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
from sphgeo import point, tg_vector_geodesic_pu_midpoints, tg_vector_geodesic_pv_midpoints
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
            self.ng  = griddata['ng'][:]
            self.ngl = griddata['ngl'][:]
            self.ngr = griddata['ngr'][:]

            # Interior indexes
            self.i0, self.iend = griddata['i0'][:], griddata['iend'][:]
            self.j0, self.jend = griddata['j0'][:], griddata['jend'][:]
            i0, iend = self.i0, self.iend
            j0, jend = self.j0, self.jend
            ng = self.ng

            # Create points
            vertices = point(N+1+ng, N+1+ng)
            pc       = point(N+ng, N+ng)
            pu       = point(N+1+ng, N+ng)
            pv       = point(N+ng, N+1+ng)
            ex_pc = point(N+ng, N+ng)
            ey_pc = point(N+ng, N+ng)
            ex_pu = point(N+1+ng, N+ng)
            ey_pu = point(N+1+ng, N+ng)
            ex_pv = point(N+ng, N+1+ng)
            ey_pv = point(N+ng, N+1+ng)

            # Get values from file
            vertices.X   = griddata['vertices'][:,:,:,0]
            vertices.Y   = griddata['vertices'][:,:,:,1]
            vertices.Z   = griddata['vertices'][:,:,:,2]
            vertices.lon = griddata['vertices'][:,:,:,3]
            vertices.lat = griddata['vertices'][:,:,:,4]

            pc.X   = griddata['pc'][:,:,:,0]
            pc.Y   = griddata['pc'][:,:,:,1]
            pc.Z   = griddata['pc'][:,:,:,2]
            pc.lon = griddata['pc'][:,:,:,3]
            pc.lat = griddata['pc'][:,:,:,4]

            pu.X   = griddata['pu'][:,:,:,0]
            pu.Y   = griddata['pu'][:,:,:,1]
            pu.Z   = griddata['pu'][:,:,:,2]
            pu.lon = griddata['pu'][:,:,:,3]
            pu.lat = griddata['pu'][:,:,:,4]

            pv.X   = griddata['pv'][:,:,:,0]
            pv.Y   = griddata['pv'][:,:,:,1]
            pv.Z   = griddata['pv'][:,:,:,2]
            pv.lon = griddata['pv'][:,:,:,3]
            pv.lat = griddata['pv'][:,:,:,4]

            ex_pc.X   = griddata['ex_pc'][:,:,:,0]
            ex_pc.Y   = griddata['ex_pc'][:,:,:,1]
            ex_pc.Z   = griddata['ex_pc'][:,:,:,2]
            ex_pc.lon = griddata['ex_pc'][:,:,:,3]
            ex_pc.lat = griddata['ex_pc'][:,:,:,4]

            ey_pc.X   = griddata['ey_pc'][:,:,:,0]
            ey_pc.Y   = griddata['ey_pc'][:,:,:,1]
            ey_pc.Z   = griddata['ey_pc'][:,:,:,2]
            ey_pc.lon = griddata['ey_pc'][:,:,:,3]
            ey_pc.lat = griddata['ey_pc'][:,:,:,4]

            ex_pu.X   = griddata['ex_pu'][:,:,:,0]
            ex_pu.Y   = griddata['ex_pu'][:,:,:,1]
            ex_pu.Z   = griddata['ex_pu'][:,:,:,2]
            ex_pu.lon = griddata['ex_pu'][:,:,:,3]
            ex_pu.lat = griddata['ex_pu'][:,:,:,4]

            ey_pu.X   = griddata['ey_pu'][:,:,:,0]
            ey_pu.Y   = griddata['ey_pu'][:,:,:,1]
            ey_pu.Z   = griddata['ey_pu'][:,:,:,2]
            ey_pu.lon = griddata['ey_pu'][:,:,:,3]
            ey_pu.lat = griddata['ey_pu'][:,:,:,4]

            ex_pv.X   = griddata['ex_pv'][:,:,:,0]
            ex_pv.Y   = griddata['ex_pv'][:,:,:,1]
            ex_pv.Z   = griddata['ex_pv'][:,:,:,2]
            ex_pv.lon = griddata['ex_pv'][:,:,:,3]
            ex_pv.lat = griddata['ex_pv'][:,:,:,4]

            ey_pv.X   = griddata['ey_pv'][:,:,:,0]
            ey_pv.Y   = griddata['ey_pv'][:,:,:,1]
            ey_pv.Z   = griddata['ey_pv'][:,:,:,2]
            ey_pv.lon = griddata['ey_pv'][:,:,:,3]
            ey_pv.lat = griddata['ey_pv'][:,:,:,4]

            # Geometric properties
            areas     = griddata['areas'][:,:,:]
            length_pu = griddata['length_pu'][:,:,:]
            length_pv = griddata['length_pv'][:,:,:]

            # Metric tensor/contravariant-covariant conversion tools
            metric_tensor_pc = griddata['metric_tensor_pc'][:,:,:]
            metric_tensor_pu = griddata['metric_tensor_pu'][:,:,:]
            metric_tensor_pv = griddata['metric_tensor_pv'][:,:,:]

            prod_ex_elon_pc          = griddata['prod_ex_elon_pc'][:,:,:]
            prod_ex_elat_pc          = griddata['prod_ex_elat_pc'][:,:,:]
            prod_ey_elon_pc          = griddata['prod_ey_elon_pc'][:,:,:]
            prod_ey_elat_pc          = griddata['prod_ey_elat_pc'][:,:,:]
            determinant_ll2contra_pc = griddata['determinant_ll2contra_pc'][:,:,:]

            prod_ex_elon_pu          = griddata['prod_ex_elon_pu'][:,:,:]
            prod_ex_elat_pu          = griddata['prod_ex_elat_pu'][:,:,:]
            prod_ey_elon_pu          = griddata['prod_ey_elon_pu'][:,:,:]
            prod_ey_elat_pu          = griddata['prod_ey_elat_pu'][:,:,:]
            determinant_ll2contra_pu = griddata['determinant_ll2contra_pu'][:,:,:]

            prod_ex_elon_pv          = griddata['prod_ex_elon_pv'][:,:,:]
            prod_ex_elat_pv          = griddata['prod_ex_elat_pv'][:,:,:]
            prod_ey_elon_pv          = griddata['prod_ey_elon_pv'][:,:,:]
            prod_ey_elat_pv          = griddata['prod_ey_elat_pv'][:,:,:]
            determinant_ll2contra_pv = griddata['determinant_ll2contra_pv'][:,:,:]

            Xu = griddata['Xu'][:,:,:]
            Yv = griddata['Yv'][:,:,:]

            # Attributes
            self.pc = pc
            self.vertices = vertices
            self.pu = pu
            self.pv = pv
            self.length_pu = length_pu
            self.length_pv = length_pv
            self.areas = areas
            self.metric_tensor_pc = metric_tensor_pc

            self.prod_ex_elon_pc = prod_ex_elon_pc
            self.prod_ex_elat_pc = prod_ex_elat_pc
            self.prod_ey_elon_pc = prod_ey_elon_pc
            self.prod_ey_elat_pc = prod_ey_elat_pc
            self.determinant_ll2contra_pc = determinant_ll2contra_pc

            self.prod_ex_elon_pu = prod_ex_elon_pu
            self.prod_ex_elat_pu = prod_ex_elat_pu
            self.prod_ey_elon_pu = prod_ey_elon_pu
            self.prod_ey_elat_pu = prod_ey_elat_pu
            self.determinant_ll2contra_pu = determinant_ll2contra_pu

            self.prod_ex_elon_pv = prod_ex_elon_pv
            self.prod_ex_elat_pv = prod_ex_elat_pv
            self.prod_ey_elon_pv = prod_ey_elon_pv
            self.prod_ey_elat_pv = prod_ey_elat_pv
            self.determinant_ll2contra_pv = determinant_ll2contra_pv

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
            ngl = 4
            ngr = 4

            ng = ngr + ngl
            self.ngl  = ngl
            self.ngr = ngr
            self.ng = ng

            # Interior indexes
            i0, iend = ngl, ngl+N
            j0, jend = ngl, ngl+N
            self.i0, self.iend = ngl, ngl+N
            self.j0, self.jend = ngl, ngl+N

            # Generate cell vertices
            x = np.linspace(x_min-ngl*dx, x_max + ngr*dx, N+1+ng) # vertices
            y = np.linspace(y_min-ngl*dx, y_max + ngr*dy, N+1+ng) # vertices
            if showonscreen==True:
                print("Generating cell vertices...")

            vertices = point(N+1+ng, N+1+ng)

            if transformation == "gnomonic_equiangular":
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = equiangular_gnomonic_map(x, y, N+1+ng, N+1+ng, self.R)

            elif transformation=="gnomonic_equidistant":
                vertices.X, vertices.Y, vertices.Z, vertices.lon, vertices.lat = equidistant_gnomonic_map(x, y, N+1+ng, N+1+ng, self.R)

            elif transformation=="conformal":
                vertices.X[i0:iend+1,j0:jend+1], vertices.Y[i0:iend+1,j0:jend+1], vertices.Z[i0:iend+1,j0:jend+1], vertices.lon[i0:iend+1,j0:jend+1], vertices.lat[i0:iend+1,j0:jend+1] = conformal_map(x[i0:iend+1], y[j0:jend+1], N+1, N+1)

            self.vertices = vertices

            # Generate cell pc
            if showonscreen==True:
                print("Generating cell pc...")

            pc = point(N+ng, N+ng)
            ex_pc   = point(N+ng, N+ng)
            ey_pc   = point(N+ng, N+ng)

            x = np.linspace(x_min+dx/2.0-ngl*dx, x_max-dx/2.0+ngr*dx, N+ng) # Centers
            y = np.linspace(y_min+dy/2.0-ngl*dy, y_max-dy/2.0+ngr*dy, N+ng) # Centers

            if transformation == "gnomonic_equiangular":
                pc.X, pc.Y, pc.Z, pc.lon, pc.lat = equiangular_gnomonic_map(x, y, N+ng, N+ng, self.R)
                ex_pc.X, ex_pc.Y, ex_pc.Z = equiangular_tg_xdir(x, y, N+ng, N+ng, self.R)
                ey_pc.X, ey_pc.Y, ey_pc.Z = equiangular_tg_ydir(x, y, N+ng, N+ng, self.R)

            elif transformation=="gnomonic_equidistant":
                pc.X, pc.Y, pc.Z, pc.lon, pc.lat = equidistant_gnomonic_map(x, y, N+ng, N+ng, self.R)
                ex_pc.X, ex_pc.Y, ex_pc.Z = equidistant_tg_xdir(x, y, N+ng, N+ng, self.R)
                ey_pc.X, ey_pc.Y, ey_pc.Z = equidistant_tg_ydir(x, y, N+ng, N+ng, self.R)
            elif transformation=="conformal":
                pc.X[i0:iend,j0:jend], pc.Y[i0:iend,j0:jend], pc.Z[i0:iend,j0:jend], pc.lon[i0:iend,j0:jend], pc.lat[i0:iend,j0:jend] \
                = conformal_map(x[i0:iend], y[j0:jend], N, N)

            self.pc = pc
            self.ex_pc = ex_pc
            self.ey_pc = ey_pc

            # Metric tensor on pc
            metric_tensor_pc = np.zeros((N+ng, N+ng, nbfaces))
            metric_tensor_pc[:,:,0] = metric_tensor(x, y, self.R, transformation)
            for p in range(1, nbfaces): metric_tensor_pc[:,:,p] = metric_tensor_pc[:,:,0]
            self.metric_tensor_pc = metric_tensor_pc

            # Generate cell edges in x direction
            if showonscreen==True:
                print("Generating cell edges and tangent vectors in x direction...")

            pu    = point(N+1+ng, N+ng)
            ex_pu = point(N+1+ng, N+ng)
            ey_pu = point(N+1+ng, N+ng)

            x = np.linspace(x_min-ngl*dx, x_max+ngr*dx, N+1+ng) # Edges
            y = np.linspace(y_min+dy/2.0-ngl*dy, y_max-dy/2.0+ngr*dy, N+ng) # Centers

            # edges
            Xu, Yu = np.meshgrid(x, y,indexing='ij')
            if transformation == "gnomonic_equiangular":
                pu.X, pu.Y, pu.Z, pu.lon, pu.lat = equiangular_gnomonic_map(x, y, N+1+ng, N+ng, self.R)
                ex_pu.X, ex_pu.Y, ex_pu.Z = equiangular_tg_xdir(x, y, N+1+ng, N+ng, self.R)
                ey_pu.X, ey_pu.Y, ey_pu.Z = equiangular_tg_ydir(x, y, N+1+ng, N+ng, self.R)
                #ex_pu2 = point(N+1+ng, N+ng)
                #ey_pu2 = point(N+1+ng, N+ng)
                #ex_pu2, ey_pu2 = tg_vector_geodesic_pu_midpoints(pu, vertices, N, ng)
            elif transformation=="gnomonic_equidistant":
                pu.X, pu.Y, pu.Z, pu.lon, pu.lat = equidistant_gnomonic_map(x, y, N+1+ng, N+ng, self.R)
                ex_pu.X, ex_pu.Y, ex_pu.Z = equidistant_tg_xdir(x, y, N+1+ng, N+ng, self.R)
                ey_pu.X, ey_pu.Y, ey_pu.Z = equidistant_tg_ydir(x, y, N+1+ng, N+ng, self.R)
                #ex_pu2 = point(N+1+ng, N+ng)
                #ey_pu2 = point(N+1+ng, N+ng)
                #ex_pu2, ey_pu2 = tg_vector_geodesic_pu_midpoints(pu, vertices, N, ng)

            elif transformation=="conformal":
                pu.X[i0:iend+1,j0:jend], pu.Y[i0:iend+1,j0:jend], pu.Z[i0:iend+1,j0:jend], pu.lon[i0:iend+1,j0:jend], pu.lat[i0:iend+1,j0:jend] \
                = conformal_map(x[i0:iend+1], y[j0:jend], N+1, N)

            self.pu = pu

            #ex_pu.X = ex_pu.X#/norm
            #ex_pu.Y = ex_pu.Y#/norm
            #ex_pu.Z = ex_pu.Z#/norm
            #ey_pu.X = ey_pu.X#/norm
            #ey_pu.Y = ey_pu.Y#/norm
            #ey_pu.Z = ey_pu.Z#/norm

            #print(np.amax(abs(ex_pu2.X-ex_pu.X)))
            #print(np.amax(abs(ex_pu2.Y-ex_pu.Y)))
            #print(np.amax(abs(ex_pu2.Z-ex_pu.Z)))
            #print(np.amax(abs(ey_pu2.X-ey_pu.X)))
            #print(np.amax(abs(ey_pu2.Y-ey_pu.Y)))
            #print(np.amax(abs(ey_pu2.Z-ey_pu.Z)))

            # Metric tensor on edges in x direction
            metric_tensor_pu = np.zeros((N+1+ng, N+ng, nbfaces))
            metric_tensor_pu[:,:,0] = metric_tensor(x, y, self.R, transformation)
            for p in range(1, nbfaces): metric_tensor_pu[:,:,p] = metric_tensor_pu[:,:,0]
            self.metric_tensor_pu = metric_tensor_pu

            # Generate cell edges in y direction
            if showonscreen==True:
                print("Generating cell edges and tangent vectors in y direction...")

            pv = point(N+ng, N+ng+1)
            ex_pv = point(N+ng, N+ng+1)
            ey_pv = point(N+ng, N+ng+1)

            x = np.linspace(x_min+dx/2.0-ngl*dx, x_max-dx/2.0+ngr*dx, N+ng) # Centers
            y = np.linspace(y_min-ngl*dx, y_max + ngr*dy, N+1+ng) # Edges

            # edges
            Xv, Yv = np.meshgrid(x, y,indexing='ij')

            if transformation == "gnomonic_equiangular":
                pv.X, pv.Y, pv.Z, pv.lon, pv.lat = equiangular_gnomonic_map(x, y, N+ng, N+ng+1, self.R)
                ex_pv.X, ex_pv.Y, ex_pv.Z = equiangular_tg_xdir(x, y, N+ng, N+ng+1, self.R)
                ey_pv.X, ey_pv.Y, ey_pv.Z = equiangular_tg_ydir(x, y, N+ng, N+ng+1, self.R)
                #ex_pv2 = point(N+ng, N+1+ng)
                #ey_pv2 = point(N+ng, N+1+ng)
                #ex_pv2, ey_pv2 = tg_vector_geodesic_pv_midpoints(pv, vertices, N, ng)
            elif transformation=="gnomonic_equidistant":
                pv.X, pv.Y, pv.Z, pv.lon, pv.lat = equidistant_gnomonic_map(x, y, N+ng, N+ng+1, self.R)
                ex_pv.X, ex_pv.Y, ex_pv.Z = equidistant_tg_xdir(x, y, N+ng, N+ng+1, self.R)
                ey_pv.X, ey_pv.Y, ey_pv.Z = equidistant_tg_ydir(x, y, N+ng, N+ng+1, self.R)
                #ex_pv2 = point(N+ng, N+1+ng)
                #ey_pv2 = point(N+ng, N+1+ng)
                #ex_pv2, ey_pv2 = tg_vector_geodesic_pv_midpoints(pv, vertices, N, ng)
            elif transformation=="conformal":
                pv.X[i0:iend,j0:jend+1], pv.Y[i0:iend,j0:jend+1], pv.Z[i0:iend,j0:jend+1], pv.lon[i0:iend,j0:jend+1], pv.lat[i0:iend,j0:jend+1] \
                = conformal_map(x[i0:iend], y[j0:jend+1], N, N+1)
            self.pv = pv

            #ex_pv.X = ex_pv.X#/norm
            #ex_pv.Y = ex_pv.Y#/norm
            #ex_pv.Z = ex_pv.Z#/norm
            #ey_pv.X = ey_pv.X#/norm
            #ey_pv.Y = ey_pv.Y#/norm
            #ey_pv.Z = ey_pv.Z#/norm
            #print(np.amax(abs(ex_pv2.X-ex_pv.X)))
            #print(np.amax(abs(ex_pv2.Y-ex_pv.Y)))
            #print(np.amax(abs(ex_pv2.Z-ex_pv.Z)))
            #print(np.amax(abs(ey_pv2.X-ey_pv.X)))
            #print(np.amax(abs(ey_pv2.Y-ey_pv.Y)))
            #print(np.amax(abs(ey_pv2.Z-ey_pv.Z)))

            # Metric tensor on edges in y direction
            metric_tensor_pv = np.zeros((N+ng, N+ng+1, nbfaces))
            metric_tensor_pv[:,:,0] = metric_tensor(x, y, self.R, transformation)
            for p in range(1, nbfaces): metric_tensor_pv[:,:,p] = metric_tensor_pv[:,:,0]
            self.metric_tensor_pv = metric_tensor_pv

            # Compute cell lenghts
            if showonscreen==True:
                print("Computing cell lengths...")

            # Compute the geodesic distance of cell edges in x direction
            # Given points
            p1 = [vertices.X[0:N+ng,:,:], vertices.Y[0:N+ng  ,:,:], vertices.Z[0:N+ng  ,:,:]]
            p2 = [vertices.X[1:N+ng+1,:,:], vertices.Y[1:N+ng+1,:,:], vertices.Z[1:N+ng+1,:,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+ng)*(N+ng+1)*nbfaces))
            p2 = np.reshape(p2,(3,(N+ng)*(N+ng+1)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1, p2)
            d = np.reshape(d,(N+ng,N+ng+1,nbfaces))
            length_x = d[:,:,:]

            # Compute the geodesic distance of cell edges in y direction
            # Given points
            p1 = [vertices.X[:,0:N+ng  ,:], vertices.Y[:,0:N+ng  ,:], vertices.Z[:,0:N+ng  ,:]]
            p2 = [vertices.X[:,1:N+ng+1,:], vertices.Y[:,1:N+ng+1,:], vertices.Z[:,1:N+ng+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+ng)*(N+ng+1)*nbfaces))
            p2 = np.reshape(p2,(3,(N+ng)*(N+ng+1)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+ng+1,N+ng,nbfaces))
            length_y = d[:,:,:]

            # Cell diagonal length
            # Given points
            p1 = [vertices.X[0:N+ng,1:N+ng+1,:], vertices.Y[0:N+ng,1:N+ng+1,:], vertices.Z[0:N+ng,1:N+ng+1,:]]
            p2 = [vertices.X[1:N+ng+1,0:N+ng,:], vertices.Y[1:N+ng+1,0:N+ng,:], vertices.Z[1:N+ng+1,0:N+ng,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+ng)*(N+ng)*nbfaces))
            p2 = np.reshape(p2,(3,(N+ng)*(N+ng)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+ng,N+ng,nbfaces))
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
            a = length_x[:,0:N+ng,:] # AB
            b = length_y[0:N+ng,:,:] # AD
            c = length_diag            # DB
            areas = sphgeo.tri_area(a, b, c)

            # Compute the area of triangle DCB
            a = length_x[:,1:N+ng+1,:] # DC
            b = length_y[1:N+ng+1,:,:] # CB
            c = length_diag              # DB
            areas = areas + sphgeo.tri_area(a, b, c)

            # Compute areas
            if showonscreen==True:
                print("Computing cell areas...")
            self.areas = (self.R**2)*areas

            # Cell pu length
            # Given points
            p1 = [pu.X[0:N+ng,0:N+ng,:]  , pu.Y[0:N+ng,0:N+ng,:]  , pu.Z[0:N+ng,0:N+ng,:]]
            p2 = [pu.X[1:N+ng+1,0:N+ng,:], pu.Y[1:N+ng+1,0:N+ng,:], pu.Z[1:N+ng+1,0:N+ng,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+ng)*(N+ng)*nbfaces))
            p2 = np.reshape(p2,(3,(N+ng)*(N+ng)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+ng,N+ng,nbfaces))

            self.length_pu = self.R*d
            #print(np.amin(self.length_pu),np.amax(self.length_pu))

            # Cell pv length
            # Given points
            p1 = [pv.X[0:N+ng,0:N+ng,:]  , pv.Y[0:N+ng,0:N+ng,:]  , pv.Z[0:N+ng,0:N+ng,:]]
            p2 = [pv.X[0:N+ng,1:N+ng+1,:], pv.Y[0:N+ng,1:N+ng+1,:], pv.Z[0:N+ng,1:N+ng+1,:]]

            # Reshape
            p1 = np.reshape(p1,(3,(N+ng)*(N+ng)*nbfaces))
            p2 = np.reshape(p2,(3,(N+ng)*(N+ng)*nbfaces))

            # Compute arclen
            d = sphgeo.arclen(p1,p2)
            d = np.reshape(d,(N+ng,N+ng,nbfaces))

            self.length_pv = self.R*d
            #print(np.amin(self.length_pv),np.amax(self.length_pv))

            # Generate tangent vectors
            if showonscreen==True:
                print("Generating latlon tangent vectors...")

            # Lat-lon tangent unit vectors
            elon_pc = sphgeo.tangent_geo_lon(pc.lon)
            elat_pc = sphgeo.tangent_geo_lat(pc.lon, pc.lat)

            # CS map tangent unit vectors at edges points
            # latlon coordinates
            ex_pc.lon = ex_pc.X[:,:,:]*elon_pc[:,:,:,0] + ex_pc.Y[:,:,:]*elon_pc[:,:,:,1] + ex_pc.Z[:,:,:]*elon_pc[:,:,:,2]
            ex_pc.lat = ex_pc.X[:,:,:]*elat_pc[:,:,:,0] + ex_pc.Y[:,:,:]*elat_pc[:,:,:,1] + ex_pc.Z[:,:,:]*elat_pc[:,:,:,2]

            ey_pc.lon = ey_pc.X[:,:,:]*elon_pc[:,:,:,0] + ey_pc.Y[:,:,:]*elon_pc[:,:,:,1] + ey_pc.Z[:,:,:]*elon_pc[:,:,:,2]
            ey_pc.lat = ey_pc.X[:,:,:]*elat_pc[:,:,:,0] + ey_pc.Y[:,:,:]*elat_pc[:,:,:,1] + ey_pc.Z[:,:,:]*elat_pc[:,:,:,2]

            self.ex_pc = ex_pc
            self.ey_pc = ey_pc
 
            # Lat-lon tangent unit vectors
            elon_pu = sphgeo.tangent_geo_lon(pu.lon)
            elat_pu = sphgeo.tangent_geo_lat(pu.lon, pu.lat)
            elon_pv = sphgeo.tangent_geo_lon(pv.lon)
            elat_pv = sphgeo.tangent_geo_lat(pv.lon, pv.lat)

            # CS map tangent unit vectors at edges points
            # latlon coordinates
            ex_pu.lon = ex_pu.X[:,:,:]*elon_pu[:,:,:,0] + ex_pu.Y[:,:,:]*elon_pu[:,:,:,1] + ex_pu.Z[:,:,:]*elon_pu[:,:,:,2]
            ex_pu.lat = ex_pu.X[:,:,:]*elat_pu[:,:,:,0] + ex_pu.Y[:,:,:]*elat_pu[:,:,:,1] + ex_pu.Z[:,:,:]*elat_pu[:,:,:,2]

            ex_pv.lon = ex_pv.X[:,:,:]*elon_pv[:,:,:,0] + ex_pv.Y[:,:,:]*elon_pv[:,:,:,1] + ex_pv.Z[:,:,:]*elon_pv[:,:,:,2]
            ex_pv.lat = ex_pv.X[:,:,:]*elat_pv[:,:,:,0] + ex_pv.Y[:,:,:]*elat_pv[:,:,:,1] + ex_pv.Z[:,:,:]*elat_pv[:,:,:,2]

            ey_pu.lon = ey_pu.X[:,:,:]*elon_pu[:,:,:,0] + ey_pu.Y[:,:,:]*elon_pu[:,:,:,1] + ey_pu.Z[:,:,:]*elon_pu[:,:,:,2]
            ey_pu.lat = ey_pu.X[:,:,:]*elat_pu[:,:,:,0] + ey_pu.Y[:,:,:]*elat_pu[:,:,:,1] + ey_pu.Z[:,:,:]*elat_pu[:,:,:,2]

            ey_pv.lon = ey_pv.X[:,:,:]*elon_pv[:,:,:,0] + ey_pv.Y[:,:,:]*elon_pv[:,:,:,1] + ey_pv.Z[:,:,:]*elon_pv[:,:,:,2]
            ey_pv.lat = ey_pv.X[:,:,:]*elat_pv[:,:,:,0] + ey_pv.Y[:,:,:]*elat_pv[:,:,:,1] + ey_pv.Z[:,:,:]*elat_pv[:,:,:,2]

            self.ex_pu = ex_pu
            self.ey_pu = ey_pu
            self.ex_pv = ex_pv
            self.ey_pv = ey_pv

            # Latlon/Contravariant conversion
            if showonscreen==True:
                print("Generating latlon/contravariant conversion variables... \n")

            # Latlon/Contravariant conversion at pc
            prod_ex_elon_pc = ex_pc.X*elon_pc[:,:,:,0] + ex_pc.Y*elon_pc[:,:,:,1] + ex_pc.Z*elon_pc[:,:,:,2]
            prod_ex_elat_pc = ex_pc.X*elat_pc[:,:,:,0] + ex_pc.Y*elat_pc[:,:,:,1] + ex_pc.Z*elat_pc[:,:,:,2]
            prod_ey_elon_pc = ey_pc.X*elon_pc[:,:,:,0] + ey_pc.Y*elon_pc[:,:,:,1] + ey_pc.Z*elon_pc[:,:,:,2]
            prod_ey_elat_pc = ey_pc.X*elat_pc[:,:,:,0] + ey_pc.Y*elat_pc[:,:,:,1] + ey_pc.Z*elat_pc[:,:,:,2]
            determinant_ll2contra_pc = prod_ex_elon_pc*prod_ey_elat_pc - prod_ey_elon_pc*prod_ex_elat_pc

            self.prod_ex_elon_pc = prod_ex_elon_pc
            self.prod_ex_elat_pc = prod_ex_elat_pc
            self.prod_ey_elon_pc = prod_ey_elon_pc
            self.prod_ey_elat_pc = prod_ey_elat_pc
            self.determinant_ll2contra_pc = determinant_ll2contra_pc

            # Latlon/Contravariant conversion at pu
            prod_ex_elon_pu = ex_pu.X*elon_pu[:,:,:,0] + ex_pu.Y*elon_pu[:,:,:,1] + ex_pu.Z*elon_pu[:,:,:,2]
            prod_ex_elat_pu = ex_pu.X*elat_pu[:,:,:,0] + ex_pu.Y*elat_pu[:,:,:,1] + ex_pu.Z*elat_pu[:,:,:,2]
            prod_ey_elon_pu = ey_pu.X*elon_pu[:,:,:,0] + ey_pu.Y*elon_pu[:,:,:,1] + ey_pu.Z*elon_pu[:,:,:,2]
            prod_ey_elat_pu = ey_pu.X*elat_pu[:,:,:,0] + ey_pu.Y*elat_pu[:,:,:,1] + ey_pu.Z*elat_pu[:,:,:,2]
            determinant_ll2contra_pu = prod_ex_elon_pu*prod_ey_elat_pu - prod_ey_elon_pu*prod_ex_elat_pu

            self.prod_ex_elon_pu = prod_ex_elon_pu
            self.prod_ex_elat_pu = prod_ex_elat_pu
            self.prod_ey_elon_pu = prod_ey_elon_pu
            self.prod_ey_elat_pu = prod_ey_elat_pu
            self.determinant_ll2contra_pu = determinant_ll2contra_pu

            # Latlon/Contravariant conversion at pv
            prod_ex_elon_pv = ex_pv.X*elon_pv[:,:,:,0] + ex_pv.Y*elon_pv[:,:,:,1] + ex_pv.Z*elon_pv[:,:,:,2]
            prod_ex_elat_pv = ex_pv.X*elat_pv[:,:,:,0] + ex_pv.Y*elat_pv[:,:,:,1] + ex_pv.Z*elat_pv[:,:,:,2]
            prod_ey_elon_pv = ey_pv.X*elon_pv[:,:,:,0] + ey_pv.Y*elon_pv[:,:,:,1] + ey_pv.Z*elon_pv[:,:,:,2]
            prod_ey_elat_pv = ey_pv.X*elat_pv[:,:,:,0] + ey_pv.Y*elat_pv[:,:,:,1] + ey_pv.Z*elat_pv[:,:,:,2]
            determinant_ll2contra_pv = prod_ex_elon_pv*prod_ey_elat_pv - prod_ey_elon_pv*prod_ex_elat_pv

            self.prod_ex_elon_pv = prod_ex_elon_pv
            self.prod_ex_elat_pv = prod_ex_elat_pv
            self.prod_ey_elon_pv = prod_ey_elon_pv
            self.prod_ey_elat_pv = prod_ey_elat_pv
            self.determinant_ll2contra_pv = determinant_ll2contra_pv
            #ex_pu, ey_pu
            #ex_pv, ey_pv
            #self.elon_pu, self.elat_pu
            #self.elon_pv, self.elat_pv

            Xu = np.repeat(Xu[:, :, np.newaxis], 6, axis=2)
            Yv = np.repeat(Yv[:, :, np.newaxis], 6, axis=2)
            self.Xu, self.Yv = Xu, Yv

            # Finish time counting
            elapsed_time = time.time() - start_time

            if showonscreen==True:
                # Print some grid properties
                print("\nMin  edge length (km)  : ","{:.2e}".format(np.amin(self.length_pu[i0:iend+1,j0:jend,:])*erad/10**3))
                print("Max  edge length (km)  : ","{:.2e}".format(np.amax(self.length_pu[i0:iend+1,j0:jend,:])*erad/10**3))
                print("Mean edge length (km)  : ","{:.2e}".format(np.mean(self.length_pu[i0:iend+1,j0:jend,:])*erad/10**3))
                print("Ratio max/min length   : ","{:.2e}".format(np.amax(self.length_pu[i0:iend+1,j0:jend,:])/np.amin(self.length_pu[i0:iend+1,j0:jend+1,:])))

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
        ng = cs_grid.ng

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
class velocity:
    def __init__(self, cs_grid, pos):
        N = cs_grid.N
        ng = cs_grid.ng

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
        elif pos == 'pc': # Velocity at pc
            self.ulon = np.zeros((N+ng, N+ng, nbfaces))
            self.vlat = np.zeros((N+ng, N+ng, nbfaces))
            self.ucontra = np.zeros((N+ng, N+ng, nbfaces))
            self.vcontra = np.zeros((N+ng, N+ng, nbfaces))
            self.vcontra_averaged = np.zeros((N+ng, N+ng, nbfaces)) # used for departure point
            self.vcontra_old      = np.zeros((N+ng, N+ng, nbfaces)) # used for departure point
        else:
            print('ERROR in  velocity class: invalid position, ', pos)

