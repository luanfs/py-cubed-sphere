####################################################################################
# 
# Module for cubed-sphere transformations.
#
# Based on "C. Ronchi, R. Iacono, P.S. Paolucci, The “Cubed Sphere”:
# A New Method for the Solution of Partial Differential Equations 
# in Spherical Geometry."
#
# It includes equiangular and equidistant gnomonic mappings.
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from constants import*
import sphgeo
import time

# The panels are indexed as below following Ronchi et al (96).
#      +---+
#      | 4 |
#  +---+---+---+---+
#  | 3 | 0 | 1 | 2 |
#  +---+---+---+---+
#      | 5 |
#      +---+
   
####################################################################################
#
# This routine computes the Gnomonic mapping based on the equiangular projection
# defined by Ronchi et al (96) for each panel.
# - ξ, η are the angular variables defined in [-pi/4,pi/4].
# - The projection is applied on the points (ξ,η)
# - N is the number of cells along a coordinate axis
# - Returns the Cartesian (x,y,z) and spherical (lon, lat) coordinates of the
# projected points.
#
####################################################################################
def equiangular_gnomonic_map(ξ, η, N):
   # Cartesian coordinates of the projected points
   x = np.zeros((N, N, nbfaces))
   y = np.zeros((N, N, nbfaces))
   z = np.zeros((N, N, nbfaces))      
   
   # Creates a grid in [-pi/4,pi/4]x[-pi/4,pi/4] using
   # the given values of ξ and η
   ξ, η = np.meshgrid(ξ, η, indexing='ij')
   
   # Auxiliary variables
   X = np.tan(ξ)
   Y = np.tan(η)
   δ2   = 1.0 + X**2 + Y**2
   δ    = np.sqrt(δ2)
   invδ = 1.0/δ
   Xoδ  = invδ*X
   Yoδ  = invδ*Y    
    
   # Compute the Cartesian coordinates for each panel
   # with the aid of the auxiliary variables 

   # Panel 0
   x[:,:,0] = invδ
   y[:,:,0] = Xoδ  # x*X
   z[:,:,0] = Yoδ  # x*Y

   # Panel 1
   y[:,:,1] =  invδ
   x[:,:,1] = -Xoδ #-y*X
   z[:,:,1] =  Yoδ # y*Y

   # Panel 2
   x[:,:,2] = -invδ
   y[:,:,2] = -Xoδ # x*X
   z[:,:,2] =  Yoδ #-x*Y
   
   # Panel 3
   y[:,:,3] = -invδ
   x[:,:,3] =  Xoδ #-y*X
   z[:,:,3] =  Yoδ #-y*Y       
   
   # Panel 4
   z[:,:,4] =  invδ
   x[:,:,4] = -Yoδ #-z*Y
   y[:,:,4] =  Xoδ # z*X
   
   # Panel 5
   z[:,:,5] = -invδ
   x[:,:,5] =  Yoδ #-z*Y
   y[:,:,5] =  Xoδ #-z*X         
               
   # Convert to spherical coordinates
   lon, lat = sphgeo.cart2sph(x, y, z)

   return x, y, z, lon, lat

####################################################################################
# This routine computes the Gnomonic mapping based on the equidistant projection
# defined by Rancic et al (96) for each panel ()
# - x1, x2 are the variables defined in [-a, a].
# - The projection is applied on the points (x1,x2)
# - N is the number of cells along a coordinate axis
# - Returns the Cartesian (x,y,z) and spherical (lon, lat) coordinates of the
# projected points.
#
# References: 
# - Rancic, M., Purser, R.J. and Mesinger, F. (1996), A global shallow-water model using an expanded
#  spherical cube: Gnomonic versus conformal coordinates. Q.J.R. Meteorol. Soc., 122: 959-982. 
#  https://doi.org/10.1002/qj.49712253209
# - Nair, R. D., Thomas, S. J., & Loft, R. D. (2005). A Discontinuous Galerkin Transport Scheme on the
# Cubed Sphere, Monthly Weather Review, 133(4), 814-828. Retrieved Feb 7, 2022, 
# from https://journals.ametsoc.org/view/journals/mwre/133/4/mwr2890.1.xml
#
####################################################################################
def equidistant_gnomonic_map(x1, x2, N):
   # Half length of the cube
   a = 1.0/np.sqrt(3.0)
   
   # Cartesian coordinates of the projected points
   x = np.zeros((N, N, nbfaces))
   y = np.zeros((N, N, nbfaces))
   z = np.zeros((N, N, nbfaces))      
   
   # Creates a grid in [-a,a]x[-a,a] using
   # the given values of x1 and x2
   x1, x2 = np.meshgrid(x1, x2, indexing='ij')
   
   # Auxiliary variables
   r2   = a**2 + x1**2 + x2**2
   r    = np.sqrt(r2)
   invr = 1.0/r
   x1or = invr*x1
   x2or = invr*x2
   aor  = invr*a   
    
   # Compute the Cartesian coordinates for each panel
   # with the aid of the auxiliary variables 

   # Panel 0
   x[:,:,0] =  aor
   y[:,:,0] =  x1or
   z[:,:,0] =  x2or

   # Panel 1
   x[:,:,1] = -x1or
   y[:,:,1] =  aor
   z[:,:,1] =  x2or

   # Panel 2
   x[:,:,2] = -aor
   y[:,:,2] = -x1or
   z[:,:,2] =  x2or
   
   # Panel 3
   x[:,:,3] =  x1or
   y[:,:,3] = -aor
   z[:,:,3] =  x2or      
   
   # Panel 4
   x[:,:,4] = -x2or
   y[:,:,4] =  x1or
   z[:,:,4] =  aor    
   
   # Panel 5
   x[:,:,5] =  x2or
   y[:,:,5] =  x1or
   z[:,:,5] = -aor         
              
   # Convert to spherical coordinates
   lon, lat = sphgeo.cart2sph(x, y, z)

   return x, y, z, lon, lat
