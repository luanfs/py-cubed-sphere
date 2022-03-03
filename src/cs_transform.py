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
import matplotlib.pyplot as plt

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
# - ξ, η are 1d arrays storing the angular variables defined in [-pi/4,pi/4].
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
# Given a panel, this routine computes the inverse of the equiangular gnomonic map
####################################################################################
def inverse_equiangular_gnomonic_map(x, y, z, panel):
   if panel == 0:
      X = y/x
      Y = z/x
   elif panel == 1:
      X = -x/y
      Y =  z/y
   elif panel == 2:
      X =  y/x
      Y = -z/x
   elif panel == 3:
      X = -x/y
      Y = -z/y
   elif panel == 4:
      X =  y/z
      Y = -x/z
   elif panel == 5:
      X = -y/z
      Y = -x/z
   else:
      print("ERROR: invalid panel.")
      exit()

   ξ, η = np.arctan(X), np.arctan(Y)

   return  ξ, η

####################################################################################
# This routine computes the Gnomonic mapping based on the equidistant projection
# defined by Rancic et al (96) for each panel
# - x1, x2 are 1d arrays storing the variables defined in [-a, a].
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

####################################################################################
# Given a panel, this routine computes the inverse of the equidistant gnomonic map
####################################################################################
def inverse_equidistant_gnomonic_map(x, y, z, panel):
   # Half length of the cube
   a = 1.0/np.sqrt(3.0)
   if panel == 0:
      r = a/x
      ξ = y*r
      η = z*r
   elif panel == 1:
      r =  a/y
      ξ = -x*r
      η =  z*r
   elif panel == 2:
      r = -a/x
      ξ = -y*r
      η =  z*r
   elif panel == 3:
      r = -a/y
      ξ =  x*r
      η =  z*r
   elif panel == 4:
      r =  a/z
      ξ =  y*r
      η = -x*r
   elif panel == 5:
      r = -a/z
      ξ =  y*r
      η =  x*r
   else:
      print("ERROR: invalid panel.")
      exit()

   return ξ, η
      
####################################################################################
# Compute the conformal map from Rancic et al(1996).
#
# This routine is a python translation from the Julia code provided by
# https://github.com/CliMA/CubedSphere.jl and the matlab 
# script from http://wwwcvs.mitgcm.org/viewvc/MITgcm/MITgcm_contrib/high_res_cube/matlab-grid-generator/map_xy2xyz.m?view=markup
#
# Reference: 
# - Rancic, M., Purser, R.J. and Mesinger, F. (1996), A global shallow-water model using an expanded
#  spherical cube: Gnomonic versus conformal coordinates. Q.J.R. Meteorol. Soc., 122: 959-982. 
#  https://doi.org/10.1002/qj.49712253209
#################################################################################### 
def conformal_map(ξ, η, N):
   ξ, η = np.meshgrid(ξ, η, indexing='ij')

   ξc = abs(ξ)
   ηc = abs(η)

   kξ  =  ξ < 0
   kη  =  η < 0
   kξη = ηc > ξc
 
   #plt.plot(ξ, η,'x')
   #plt.show()  
   X = ξc
   Y = ηc
   #plt.plot(ξc, ηc,'x')
   #plt.show()     
   ξc = 1 - ξc
   ηc = 1 - ηc

   ξc[kξη] = 1 - Y[kξη]
   ηc[kξη] = 1 - X[kξη]	
   #plt.plot(ξc, ηc,'x')
   #plt.show()     
   z = ((ξc+1j*ηc)/2.0)**4
   W = WofZ(z)

   thrd = 1.0/3.0
   i3 = 1j**thrd
   ra = np.sqrt(3.0)-1.0
   cb = 1j-1.0
   cc = ra*cb/2.0
	
   W = i3*(W*1j)**thrd
   W = (W - ra)/(cb + cc*W)

   X = W.real
   Y = W.imag
   H = 2.0/(1+X**2+Y**2)
   X = X*H
   Y = Y*H
   Z = H-1

   #T = X

   X[kξη], Y[kξη] =  Y[kξη], X[kξη]
   #Y[kxy] =  T[kxy]
   Y[kη]  = -Y[kη]
   X[kξ]  = -X[kξ]

   # Fix truncation for x = 0 or y = 0 
   X[ξ==0]=0
   Y[η==0]=0
   
   #print(np.amax(abs(X**2+Y**2+Z**2-1)))

   # Cartesian coordinates of the projected points
   x = np.zeros((N, N, nbfaces))
   y = np.zeros((N, N, nbfaces))
   z = np.zeros((N, N, nbfaces))
  
   # Panel 4
   x[:,:,4] = np.transpose(np.flipud(X))
   y[:,:,4] = np.transpose(np.flipud(Y))
   z[:,:,4] = np.transpose(np.flipud(Z))

   # Panel 5
   x[:,:,5] = -x[:,:,4]
   y[:,:,5] =  y[:,:,4]
   z[:,:,5] = -z[:,:,4]

   # Panel 3
   x[:,:,3] =  y[:,:,4]
   y[:,:,3] = -z[:,:,4]
   z[:,:,3] = -x[:,:,4]

   # Panel 2
   x[:,:,2] = -z[:,:,4]
   y[:,:,2] = -y[:,:,4]
   z[:,:,2] = -x[:,:,4]     

   # Panel 1
   x[:,:,1] = -y[:,:,4]
   y[:,:,1] =  z[:,:,4]
   z[:,:,1] = -x[:,:,4]
   
   # Panel 0
   x[:,:,0] =  z[:,:,4]
   y[:,:,0] =  y[:,:,4]
   z[:,:,0] = -x[:,:,4]

   # Convert to spherical coordinates
   lon, lat = sphgeo.cart2sph(x, y, z)
   return x, y, z, lon, lat
 
####################################################################################
# Failed attempt to compute the conformal map inverse
# Given a panel, this routine computes the inverse of the conformal map
# The points given in the arrays x,y,z are assumed to lie in the same panel
####################################################################################
def inverse_conformal_map(x, y, z, panel):
   # Maps back to panel 4
   if panel == 0:
      Z =  x
      Y =  y
      X = -z
   elif panel == 1:
      Y = -x
      Z =  y
      X = -z
   elif panel == 2:
      Z = -x
      Y = -y
      X = -z
   elif panel == 3:
      Y =  x
      Z = -y
      X = -z
   elif panel == 4:
      X =  x
      Y =  y
      Z =  z 
   elif panel == 5:
      X = -x
      Y =  y
      Z = -z
   else:
      print("ERROR: invalid panel.")
      exit()      

   X = np.ndarray.flatten(X)
   Y = np.ndarray.flatten(Y)
   Z = np.ndarray.flatten(Z)      
   P = np.zeros((np.shape(X)[0],3))
   P[:,0] = abs(X)
   P[:,1] = abs(Y)
   P[:,2] = abs(Z)
   PM = P.max(axis=1)
   mask = np.logical_and(PM == abs(Z), Z>0)
   #print(mask)
   #for i in range(np.shape(x)[0]):
   #   print(i,mask[i])
   #   if mask[i] == False:
   #      exit()
   # Compute the inverse in panel 4   
   #X[X<0] = -X[X<0]
   #Y[Y<0] = -Y[Y<0]

   kXY = X>Y
   X[kXY], Y[kXY] =  Y[kXY], X[kXY]
   kX = X<0
   kY = Y<0
   Y[kY]  = -Y[kY]
   X[kX]  = -X[kX]

   H  = Z + 1.0
   Xs = X/H
   Ys = Y/H
   im = 1j
   ω  = Xs + im*Ys
   ra = np.sqrt(3.0) - 1.0
   cb = -1.0 + im
   cc = ra * cb / 2.0
   ω0 = (ω*cb + ra)/(1-ω*cc)
   #print(np.amax(abs(ω0)))
   #W0 = im*ω0**(3*im)
   W0 = -ω0**3
   #print(np.amax(abs(W0)))
   Z1  = ZofW(W0)
   #print(np.amax(abs(Z1)))
   z1  = (Z1**0.25)*2
   ξ, η = z1.real, z1.imag   #x1, y1 = reim(z)

   kη = η < 0
   η[kη] = -η[kη]
   kξη = abs(η) <= abs(ξ)
   η = 1.0 - abs(η)
   ξ = 1.0 - abs(ξ)
   #T =  ξ[kξη]
   #ξ[kξη] = η[kξη] # !kxy && ( x1 = 1 - abs(yy) )
   #η[kξη] = T # !kxy && ( y1 = 1 - abs(xx) )

   #xf = ξ
   #yf = η
   
   #xf[X<Y] = η[X<Y]   # ( X < Y ) && ( xf = y1  )
   #yf[X<Y] = ξ[X<Y]   # ( X < Y ) && ( yf = x1  )
   #ξ = xf
   #η = yf
   

   #X = ξc
   #Y = ηc	
   
   #ξc = 1 - ξc
   #ηc = 1 - ηc

   #kξ  =  ξ < 0
   #kη  =  η < 0
   #kξη = ηc > ξc
   
   
   #ξc = abs(ξ)
   #ηc = abs(η)

   #print(np.amax(abs(ξ)), np.amax(abs(η)))
   #plt.plot(ξ,η,'+')
   #plt.show()
   
   return ξ, η

####################################################################################
# Evaluates the Taylor series W(z) (equation A.3a from Rancic et al 1996)
####################################################################################
def WofZ(z):
   A = A_coeffs()
   w = np.zeros(np.shape(z))
   for j in range(0,30):
      w = w + A[j]*z**(j+1)
   return w
   
####################################################################################
# Evaluates the Taylor series Z(w) (equation A.4a from Rancic et al 1996)
####################################################################################
def ZofW(w):
   B = B_coeffs()
   z = np.zeros(np.shape(w))
   for j in range(0,30):
      z = z + B[j]*w**(j+1)
   return z

####################################################################################
# A coefficients given in table B1 from Rancic et al 1996
####################################################################################
def A_coeffs():
   A = [
    +1.47713062600964,
    -0.38183510510174,
    -0.05573058001191,
    -0.00895883606818,
    -0.00791315785221,
    -0.00486625437708,
    -0.00329251751279,
    -0.00235481488325,
    -0.00175870527475,
    -0.00135681133278,
    -0.00107459847699,
    -0.00086944475948,
    -0.00071607115121,
    -0.00059867100093,
    -0.00050699063239,
    -0.00043415191279,
    -0.00037541003286,
    -0.00032741060100,
    -0.00028773091482,
    -0.00025458777519,
    -0.00022664642371,
    -0.00020289261022,
    -0.00018254510830,
    -0.00016499474461,
    -0.00014976117168,
    -0.00013646173946,
    -0.00012478875823,
    -0.00011449267279,
    -0.00010536946150,
    -0.00009725109376]
   return A

####################################################################################
# B coefficients given in table B1 from Rancic et al 1996
####################################################################################
def B_coeffs():
   B = [0.67698822171341,
      0.11847295533659,
      0.05317179075349,
      0.02965811274764,
      0.01912447871071,
      0.01342566129383,
      0.00998873721022,
      0.00774869352561,
      0.00620347278164,
      0.00509011141874,
      0.00425981415542,
      0.00362309163280,
      0.00312341651697,
      0.00272361113245,
      0.00239838233411,
      0.00213002038153,
      0.00190581436893,
      0.00171644267546,
      0.00155493871562,
      0.00141600812949,
      0.00129556691848,
      0.00119042232809,
      0.00109804804853,
      0.00101642312253,
      0.00094391466713,
      0.00087919127990,
      0.00082115825576,
      0.00076890854394,
      0.00072168520663,
      0.00067885239089]
   return B

####################################################################################
# Linear search of the points (xll, yll, zll) in a given panel 
####################################################################################
def linear_search(xll, yll, zll, cs_grid, panel):
   print('Using linear search, this may take a while')
   x = cs_grid.vertices.x[:,:,panel]
   y = cs_grid.vertices.y[:,:,panel]
   z = cs_grid.vertices.z[:,:,panel]
   N = cs_grid.N
   
   # Points
   p  = np.zeros(3)
   p1 = np.zeros(3)
   p2 = np.zeros(3)
   p3 = np.zeros(3)
   p4 = np.zeros(3)
   indexes =  np.zeros((2,len(xll)), dtype=np.int32)

   for k in range(0,len(xll)):
      p[0:3] = xll[k], yll[k], zll[k]
      # Linear search for p
      for i in range(0,N):
         for j in range(0,N):
            p1[0:3] = x[i  , j]  , y[i  , j]  , z[i  , j]
            p2[0:3] = x[i+1, j]  , y[i+1, j]  , z[i+1, j]
            p3[0:3] = x[i  , j+1], y[i  , j+1], z[i  , j+1]   
            p4[0:3] = x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]   
            inside = sphgeo.inside_quadrilateral(p, p1, p2, p3, p4)           
            if inside == True:
               indexes[0,k], indexes[1,k] = i, j
               break
         if inside == True:
            indexes[0,k], indexes[1,k] = i, j
            break
      if inside == False:
         print('Error in linear search.')
         exit()
   return indexes

####################################################################################
# Binary search of the points (xll, yll, zll) in a given panel 
####################################################################################
def binary_search(xll, yll, zll, cs_grid, panel):
   #print('Using linear search, this may take a while')
   x = cs_grid.vertices.x[:,:,panel]
   y = cs_grid.vertices.y[:,:,panel]
   z = cs_grid.vertices.z[:,:,panel]
   N = cs_grid.N

   # Points
   p  = np.zeros(3)
   p1 = np.zeros(3)
   p2 = np.zeros(3)
   p3 = np.zeros(3)
   p4 = np.zeros(3)
   indexes =  np.zeros((2,len(xll)), dtype=np.int32)

   for k in range(0,len(xll)):
      print('Binary search: searching for point',k,'from',len(xll),'in panel',panel)
      p[0:3] = xll[k], yll[k], zll[k]
      leftx  = 0
      rightx = N
      lefty  = 0
      righty = N

      while abs(leftx-rightx)>1 or abs(lefty - righty)>1:
         if(leftx < rightx):
            middlex = int((leftx+rightx)/2)
         if(lefty < righty):
            middley = int((lefty+righty)/2)

         p1[0:3] = x[leftx, lefty],   y[leftx, lefty]  , z[leftx, lefty]
         p2[0:3] = x[rightx, lefty],  y[rightx, lefty] , z[rightx, lefty]
         p3[0:3] = x[leftx, righty],  y[leftx, righty] , z[leftx, righty]         
         p4[0:3] = x[rightx, righty], y[rightx, righty], z[rightx, righty]
         inside = sphgeo.inside_quadrilateral(p, p1, p2, p3, p4)

         if(inside==False):
            print('Error 1')
            exit()

         ix = [leftx, middlex, rightx]
         jy = [lefty, middley, righty]        

         for i in range(0,2):
            for j in range(0,2):
               A = ix[i]  , jy[j]
               B = ix[i+1], jy[j]
               C = ix[i]  , jy[j+1]
               D = ix[i+1], jy[j+1]
               
               p1[0:3] = x[A], y[A], z[A]
               p2[0:3] = x[B], y[B], z[B]
               p3[0:3] = x[C], y[C], z[C]         
               p4[0:3] = x[D], y[D], z[D]
               inside = sphgeo.inside_quadrilateral(p, p1, p2, p3, p4)
               if inside == True:
                  leftx_old, rightx_old = leftx, rightx
                  lefty_old, righty_old = lefty, righty
                  leftx , lefty  = A
                  rightx, righty = D
                  break
            if inside == True:
               break
         
         # Linear search
         if inside == False:
            for i in range(leftx_old, rightx_old):
               for j in range(lefty_old, righty_old):
                  p1[0:3] = x[i  , j]  , y[i  , j]  , z[i  , j]
                  p2[0:3] = x[i+1, j]  , y[i+1, j]  , z[i+1, j]
                  p3[0:3] = x[i  , j+1], y[i  , j+1], z[i  , j+1]   
                  p4[0:3] = x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]   
                  inside = sphgeo.inside_quadrilateral(p, p1, p2, p3, p4)           
                  if inside == True:
                     leftx , lefty  = i, j
                     rightx, righty = i, j
                     break
               if inside == True:
                  leftx , lefty  = i, j
                  rightx, righty = i, j
                  break
            #print(inside,leftx,rightx)
            if inside == False:
               leftx_old  = 0
               lefty_old  = 0
               rightx_old = N
               righty_old = N
               #exit()
      indexes[0,k] = leftx
      indexes[1,k] = lefty
   return indexes
