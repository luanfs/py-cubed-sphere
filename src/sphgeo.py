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
# Compute
####################################################################################
def latlon_to_contravariant(u_lon, v_lat, ex_lon, ex_lat, ey_lon, ey_lat, det):
    ucontra =  ey_lat*u_lon - ey_lon*v_lat
    vcontra = -ex_lat*u_lon + ex_lon*v_lat
    ucontra =  ucontra/det
    vcontra =  vcontra/det
    return ucontra, vcontra
