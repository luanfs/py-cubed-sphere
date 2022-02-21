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
# Inputs: latitude (ϕ), longitude (λ)
####################################################################################
def sph2cart(λ, ϕ):
   x = np.cos(ϕ)*np.cos(λ)
   y = np.cos(ϕ)*np.sin(λ)
   z = np.sin(ϕ)   
   return x, y, z

####################################################################################
# Convert from cartesian (x,y,z) to spherical coordinates (lat,lon)
# on the unit sphere.
# Outputs: latitude (ϕ), longitude (λ)
####################################################################################
def cart2sph(x, y, z):
   hypotxy = np.hypot(x, y)
   ϕ = np.arctan2(z, hypotxy)
   λ = np.arctan2(y, x)
   return λ, ϕ

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
# α1, α2, α3 and α4 using the spherical excess formula
####################################################################################
def quad_area(α1, α2, α3, α4):
   area = α1 + α2 + α3 + α4 - 2*pi
   return area

####################################################################################
# Given a spherical triangle with lengths a, b and c, this routine computes the
# angle C (opposite to c) using the spherical law of cossines
####################################################################################  
def tri_angle(a, b, c):
   angle = np.arccos((np.cos(c) - np.cos(a)*np.cos(b))/(np.sin(a)*np.sin(b)))
   return angle
