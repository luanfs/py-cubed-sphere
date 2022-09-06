####################################################################################
# 
# This module contains all the spherical geometry routines 
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

# Imports
import numpy as np
from constants import pi

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
#
#
####################################################################################
def quad_areas(x, y):
    N = np.shape(x)[0]
    print(N)
    area = np.zeros((N-1,N-1,6))
    for i in range(0,N-1):
        for j in range(0,N-1):
            x0 = x[i,j]
            xf = x[i+1,j]
            y0 = y[i,j]
            yf = y[i,j+1]
            # Equidistant CS
            a = 1.0/np.sqrt(3.0)
            r1 = (a*np.sqrt(a**2 + xf**2 + yf**2))
            r2 = (a*np.sqrt(a**2 + x0**2 + yf**2))
            r3 = (a*np.sqrt(a**2 + xf**2 + y0**2))
            r4 = (a*np.sqrt(a**2 + x0**2 + y0**2))
            #print(x0,xf,r1)
            area[i,j,:] = np.arctan(xf*yf/r1)
            area[i,j,:] = area[i,j,:] - np.arctan(x0*yf/r2)
            area[i,j,:] = area[i,j,:] - np.arctan(xf*y0/r3)
            area[i,j,:] = area[i,j,:] + np.arctan(x0*y0/r4)
            area[i,j,:] = area[i,j,:]
    return area

####################################################################################
# Based on the routine INSIDETR from iModel (https://github.com/luanfs/iModel/blob/master/src/smeshpack.f90)
# Checks if 'p' is inside the geodesical triangle formed by p1, p2, p3
# The vertices of the triangle must be given ccwisely
# The algorithm checks if the point left of each edge
####################################################################################
def inside_triangle(p, p1, p2, p3):
    insidetr = False
    A = np.zeros((3,3))

    # For every edge
    # Check if point is at left of edge
    A[0:3,0] = p[0:3] 
    A[0:3,1] = p1[0:3] 
    A[0:3,2] = p2[0:3] 
    left = np.linalg.det(A)
    if(left < 0):
        # Point not in triangle
        return insidetr

    A[0:3,0] = p[0:3] 
    A[0:3,1] = p2[0:3] 
    A[0:3,2] = p3[0:3] 
    left = np.linalg.det(A)
    if(left < 0):
        # Point not in triangle
        return insidetr

    A[0:3,0] = p[0:3] 
    A[0:3,1] = p3[0:3] 
    A[0:3,2] = p1[0:3]
    left = np.linalg.det(A)
    if(left < 0):
        # Point not in triangle
        return insidetr

    # If left >=0  to all edge, then the point
    # is inside the triangle, or on the edge
    insidetr = True

    return insidetr

####################################################################################
# Checks if 'p' is inside the geodesical quadrilateral formed by p1, p2, p3, p4
# The vertices of the quadrilateral must be given ccwisely
####################################################################################
def inside_quadrilateral(p, p1, p2, p3, p4):
    # Check if p is inside the geodesical triangle formed by p1, p2 and p3
    inside_tr1 = inside_triangle(p, p1, p2, p3)

    if inside_tr1 == True:
        return inside_tr1
    else:
        # Check if p is inside the geodesical triangle formed by p2, p4 and p3
        inside_tr2 = inside_triangle(p, p2, p4, p3)
        inside_quad = np.logical_or(inside_tr1, inside_tr2)
        return inside_quad
