####################################################################################
# 
# This module contains all the spherical geometry routines 
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

# Imports
import numpy as np
from constants import pi, nbfaces

####################################################################################
# Convert from spherical (lat,lon) to cartesian coordinates (x,y,z) 
# on the unit sphere.
# Inputs: latitude (lat), longitude (lon)
####################################################################################
def sph2cart(lon, lat):
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)   
    return x, y, z

####################################################################################
# Convert from cartesian (x,y,z) to spherical coordinates (lat,lon)
# on the unit sphere.
# Outputs: latitude (lat), longitude (lon)
####################################################################################
def cart2sph(X, Y, Z):
    hypotxy = np.hypot(X, Y)
    lat = np.arctan2(Z, hypotxy)
    lon = np.arctan2(Y, X)
    return lon, lat

####################################################################################
# Compute the geodesic distance of the unit sphere points p1=(x1,y1,z1) and 
# p2=(x2,y2,z2) given in cartesian coordinates. First, it computes the inner product
# r = <p1+p2,p1+p2> = <p1,p1> + 2*<p1,p2> + <p2,p2> = 2 + 2*<p1,p2>
# Then, the distance is given by d = 2*arctan(sqrt((4-r)/r)).
####################################################################################
def arclen(p1, p2):
    # <p1+p2,p1+p2>
    r = p1+p2
    r = r[0,:]*r[0,:] + r[1,:]*r[1,:] + r[2,:]*r[2,:]
    d = 2.0*np.arctan(np.sqrt((4-r)/r))   
    #print(r)
    return d

####################################################################################
# Compute the area of a geodesic triangle (on the unit sphere) with lengths a, b 
# and c using L'Huilier's Theorem
####################################################################################
def tri_area(a, b, c):
    s = (a + b + c)/2  # Semiperimeter
    tmp = np.tan(s/2)*np.tan((s-a)/2)*np.tan((s-b)/2)*np.tan((s-c)/2)
    area = 4.0*np.arctan(np.sqrt(tmp))
    return area

####################################################################################
# Compute the area of a geodesic quadrilateral (on the unit sphere) with angles 
# a1, a2, a3 and a4 using the spherical excess formula
####################################################################################
def quad_area(a1, a2, a3, a4):
    area = a1 + a2 + a3 + a4 - 2*pi
    return area

####################################################################################
# Generate the unit tangent (R^3) vector in the geographic coordinates in longitude direction
# lon is a (N,M,P) array (P = cs panel)
####################################################################################
def tangent_geo_lon(lon):
    e_lon =  np.zeros((np.shape(lon)[0], np.shape(lon)[1], np.shape(lon)[2], 3))
    e_lon[:,:,:,0] = -np.sin(lon)
    e_lon[:,:,:,1] =  np.cos(lon)
    e_lon[:,:,:,2] =  0.0
    return e_lon

####################################################################################
# Generate the unit tangent (R^3) vector in the geographic coordinates in latitude direction
# lon and lat are (N,M,P) arrays (P = cs panel)
####################################################################################
def tangent_geo_lat(lon, lat):
    e_lat =  np.zeros((np.shape(lon)[0], np.shape(lon)[1], np.shape(lon)[2], 3))
    e_lat[:,:,:,0] = -np.sin(lat)*np.cos(lon)
    e_lat[:,:,:,1] = -np.sin(lat)*np.sin(lon)
    e_lat[:,:,:,2] =  np.cos(lat)
    return e_lat

####################################################################################
# Given a point P in the sphere of radius 1, this routine returns the projection
# of Q at the tangent space at P.
# Returns the X, Y and Z components of the projection
####################################################################################
def tangent_projection(P, Q):
    proj_vec = np.zeros((np.shape(P)[0], np.shape(P)[1], np.shape(P)[2], 3))
    c = P[:,:,:,0]*Q[:,:,:,0] + P[:,:,:,1]*Q[:,:,:,1] + P[:,:,:,2]*Q[:,:,:,2]
    proj_vec[:,:,:,0] = c*P[:,:,:,0] - Q[:,:,:,0]
    proj_vec[:,:,:,1] = c*P[:,:,:,1] - Q[:,:,:,1]
    proj_vec[:,:,:,2] = c*P[:,:,:,2] - Q[:,:,:,2]

    # normalization
    #norm = np.sqrt(proj_vec[:,:,:,0]**2 + proj_vec[:,:,:,1]**2 + proj_vec[:,:,:,1]**2)
    #proj_vec[:,:,:,0] = proj_vec[:,:,:,0]/norm
    #proj_vec[:,:,:,1] = proj_vec[:,:,:,1]/norm
    #proj_vec[:,:,:,2] = proj_vec[:,:,:,2]/norm
    return proj_vec[:,:,:,0], proj_vec[:,:,:,1], proj_vec[:,:,:,2]

####################################################################################
# Given a spherical triangle with lengths a, b and c, this routine computes the
# angle C (opposite to c) using the spherical law of cossines
####################################################################################  
def tri_angle(a, b, c):
    angle = np.arccos((np.cos(c) - np.cos(a)*np.cos(b))/(np.sin(a)*np.sin(b)))
    return angle

####################################################################################
# Convert a latlon vector to contravariant vector
####################################################################################
def latlon_to_contravariant(u_lon, v_lat, ex_lon, ex_lat, ey_lon, ey_lat, det):
    ucontra =  ey_lat*u_lon - ey_lon*v_lat
    vcontra = -ex_lat*u_lon + ex_lon*v_lat
    ucontra =  ucontra/det
    vcontra =  vcontra/det
    return ucontra, vcontra

####################################################################################
# Convert a contravariant vector to latlon vector
####################################################################################
def contravariant_to_latlon(ucontra, vcontra, ex_lon, ex_lat, ey_lon, ey_lat):
    u_lon =  ex_lon*ucontra + ey_lon*vcontra
    v_lat =  ex_lat*ucontra + ey_lat*vcontra
    return u_lon, v_lat

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
# Compute unit tangent vectors at edges (in x dir) midpoints using the projection
# on the geodesic tangent line at P. Works only for geodesic grids.
# Returns the tangent vectors in both x and y directions of the CS local coordinates.
# Inputs:
# - edx (edges midpoints in x direction)
# - vertices (cell vertices)
# - N (cells per axis on each CS panel)
# - nghost (number of ghost cells)
####################################################################################
def tg_vector_geodesic_edx_midpoints(edx, vertices, N, nghost):
    # Unit tangent vectors in x dir
    tg_ex_edx = point(N+1+nghost, N+nghost)
    P = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))
    Q = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))

    # Points where we are going to compute the tangent vector
    P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edx.X, edx.Y, edx.Z
    
    # Vectors to be projected at the tangent line at P.
    # The tangent line considers the geodesic connecting edge mipoints in x dir.
    Q[0:N+nghost,:,:,0], Q[0:N+nghost,:,:,1], Q[0:N+nghost,:,:,2] = edx.X[0:N+nghost,:,:]-edx.X[1:N+nghost+1,:,:], edx.Y[0:N+nghost,:,:]-edx.Y[1:N+nghost+1,:,:], edx.Z[0:N+nghost,:,:]-edx.Z[1:N+nghost+1,:,:]
    Q[N+nghost,:,:,0], Q[N+nghost,:,:,1], Q[N+nghost,:,:,2] = edx.X[N+nghost-1,:,:]-edx.X[N+nghost,:,:], edx.Y[N+nghost-1,:,:]-edx.Y[N+nghost,:,:], edx.Z[N+nghost-1,:,:]-edx.Z[N+nghost,:,:]
    
    # Compute the projection on the tangent line at P     
    tg_ex_edx.X, tg_ex_edx.Y, tg_ex_edx.Z = tangent_projection(P, Q)

    # Normalization
    norm = tg_ex_edx.X*tg_ex_edx.X + tg_ex_edx.Y*tg_ex_edx.Y + tg_ex_edx.Z*tg_ex_edx.Z
    norm = np.sqrt(norm)
    tg_ex_edx.X = tg_ex_edx.X/norm
    tg_ex_edx.Y = tg_ex_edx.Y/norm
    tg_ex_edx.Z = tg_ex_edx.Z/norm

    # Unit tangent vectors in y dir
    tg_ey_edx = point(N+1+nghost, N+nghost)
    P = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))
    Q = np.zeros((N+1+nghost, N+nghost, nbfaces, 3))

    # Points where we are going to compute the tangent vector
    P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edx.X, edx.Y, edx.Z
    
    # Vectors to be projected at the tangent line at P.
    # The tangent line considers the geodesic connecting edx midpoints and vertices (y dir). 
    Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2] = edx.X[:,:,:]-vertices.X[:,1:N+nghost+1,:], edx.Y[:,:,:]-vertices.Y[:,1:N+nghost+1,:], edx.Z[:,:,:]-vertices.Z[:,1:N+nghost+1,:]

    # Compute the projection on the tangent line at P
    tg_ey_edx.X, tg_ey_edx.Y, tg_ey_edx.Z = tangent_projection(P, Q)
 
    # Normalization
    norm = tg_ey_edx.X*tg_ey_edx.X + tg_ey_edx.Y*tg_ey_edx.Y + tg_ey_edx.Z*tg_ey_edx.Z
    norm = np.sqrt(norm)
    tg_ey_edx.X = tg_ey_edx.X/norm
    tg_ey_edx.Y = tg_ey_edx.Y/norm
    tg_ey_edx.Z = tg_ey_edx.Z/norm
    return tg_ex_edx, tg_ey_edx

####################################################################################
# Compute unit tangent vectors at edges (in y dir) midpoints using the projection
# on the geodesic tangent line at P. Works only for geodesic grids.
# Returns the tangent vectors in both x and y directions of the CS local coordinates.
# Inputs:
# - edy (edges midpoints in y direction)
# - vertices (cell vertices)
# - N (cells per axis on each CS panel)
# - nghost (number of ghost cells)
####################################################################################
def tg_vector_geodesic_edy_midpoints(edy, vertices, N, nghost):
    # Unit tangent vectors in x dir
    tg_ex_edy = point(N+nghost, N+nghost+1)
    P = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))
    Q = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))

    # Points where we are going to compute the tangent vector
    P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edy.X, edy.Y, edy.Z
 
    # Vectors to be projected at the tangent line at P.
    # The tangent line considers the geodesic connecting edge mipoints and vercites in x dir.
    Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2] = edy.X-vertices.X[1:N+nghost+1,:,:], edy.Y-vertices.Y[1:N+nghost+1,:,:], edy.Z-vertices.Z[1:N+nghost+1,:,:]

    # Compute the projection on the tangent line at P     
    tg_ex_edy.X, tg_ex_edy.Y, tg_ex_edy.Z = tangent_projection(P, Q)

    # Normalization
    norm = tg_ex_edy.X*tg_ex_edy.X + tg_ex_edy.Y*tg_ex_edy.Y + tg_ex_edy.Z*tg_ex_edy.Z
    norm = np.sqrt(norm)
    tg_ex_edy.X = tg_ex_edy.X/norm
    tg_ex_edy.Y = tg_ex_edy.Y/norm
    tg_ex_edy.Z = tg_ex_edy.Z/norm

    # Unit tangent vectors in y dir
    tg_ey_edy = point(N+1+nghost, N+nghost)
    P = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))
    Q = np.zeros((N+nghost, N+nghost+1, nbfaces, 3))

    # Points where we are going to compute the tangent vector
    P[:,:,:,0], P[:,:,:,1] ,P[:,:,:,2] = edy.X, edy.Y, edy.Z

    # Vectors to be projected at the tangent line at P.
    # The tangent line considers the geodesic connecting edx midpoints and vertices (x dir). 
    Q[:,0:N+nghost,:,0], Q[:,0:N+nghost,:,1], Q[:,0:N+nghost,:,2] = edy.X[:,0:N+nghost,:]-edy.X[:,1:N+nghost+1,:], edy.Y[:,0:N+nghost,:]-edy.Y[:,1:N+nghost+1,:], edy.Z[:,0:N+nghost,:]-edy.Z[:,1:N+nghost+1,:]
    Q[:,N+nghost,:,0], Q[:,N+nghost,:,1], Q[:,N+nghost,:,2] = edy.X[:,N+nghost-1,:]-edy.X[:,N+nghost,:], edy.Y[:,N+nghost-1,:]-edy.Y[:,N+nghost,:], edy.Z[:,N+nghost-1,:]-edy.Z[:,N+nghost,:]

    # Compute the projection on the tangent line at P
    tg_ey_edy.X, tg_ey_edy.Y, tg_ey_edy.Z = tangent_projection(P, Q)

    # Normalization
    norm = tg_ey_edy.X*tg_ey_edy.X + tg_ey_edy.Y*tg_ey_edy.Y + tg_ey_edy.Z*tg_ey_edy.Z
    norm = np.sqrt(norm)
    tg_ey_edy.X = tg_ey_edy.X/norm
    tg_ey_edy.Y = tg_ey_edy.Y/norm
    tg_ey_edy.Z = tg_ey_edy.Z/norm
    return tg_ex_edy, tg_ey_edy
