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
# - x, y are 1d arrays storing the angular variables defined in [-pi/4,pi/4].
# - The projection is applied on the points (x, y)
# - N is the number of cells along a coordinate axis
# - Returns the Cartesian (X,Y,Z) and spherical (lon, lat) coordinates of the
# projected points.
#
####################################################################################
def equiangular_gnomonic_map(x, y, N, M, R):
    # Cartesian coordinates of the projected points
    X = np.zeros((N, M, nbfaces))
    Y = np.zeros((N, M, nbfaces))
    Z = np.zeros((N, M, nbfaces))      
   
    # Creates a grid in [-pi/4,pi/4]x[-pi/4,pi/4] using
    # the given values of x and y
    x, y = np.meshgrid(x, y, indexing='ij')
   
    # Auxiliary variables
    tanx = np.tan(x)
    tany = np.tan(y)
    D2   = 1.0 + tanx**2 + tany**2
    D    = np.sqrt(D2)
    invD = 1.0/D
    XoD  = invD*tanx
    YoD  = invD*tany    
    
    # Compute the Cartesian coordinates for each panel
    # with the aid of the auxiliary variables 

    # Panel 0
    X[:,:,0] = invD
    Y[:,:,0] = XoD  # x*X
    Z[:,:,0] = YoD  # x*Y

    # Panel 1
    Y[:,:,1] =  invD
    X[:,:,1] = -XoD #-y*X
    Z[:,:,1] =  YoD # y*Y

    # Panel 2
    X[:,:,2] = -invD
    Y[:,:,2] = -XoD # x*X
    Z[:,:,2] =  YoD #-x*Y
   
   # Panel 3
    Y[:,:,3] = -invD
    X[:,:,3] =  XoD #-y*X
    Z[:,:,3] =  YoD #-y*Y       
   
    # Panel 4
    Z[:,:,4] =  invD
    X[:,:,4] = -YoD #-z*Y
    Y[:,:,4] =  XoD # z*X
   
    # Panel 5
    Z[:,:,5] = -invD
    X[:,:,5] =  YoD #-z*Y
    Y[:,:,5] =  XoD #-z*X
               
    # Convert to spherical coordinates
    lon, lat = sphgeo.cart2sph(X, Y, Z)

    return X, Y, Z, lon, lat

####################################################################################
# Given a panel, this routine computes the inverse of the equiangular gnomonic map
####################################################################################
def inverse_equiangular_gnomonic_map(X, Y, Z, panel):
    if panel == 0:
        tanx = Y/X
        tany = Z/X
    elif panel == 1:
        tanx = -X/Y
        tany =  Z/Y
    elif panel == 2:
        tanx =  Y/X
        tany = -Z/X
    elif panel == 3:
        tanx = -X/Y
        tany = -Z/Y
    elif panel == 4:
        tanx =  Y/Z
        tany = -X/Z
    elif panel == 5:
        tanx = -Y/Z
        tany = -X/Z
    else:
        print("ERROR: invalid panel.")
        exit()

    x, y = np.arctan(tanx), np.arctan(tany)

    return  x, y

####################################################################################
# This routine computes the Gnomonic mapping based on the equidistant projection
# defined by Rancic et al (96) for each panel
# - x, y are 1d arrays storing the variables defined in [-a, a].
# - The projection is applied on the points (x,y)
# - N is the number of cells along a coordinate axis
# - Returns the Cartesian (X,Y,Z) and spherical (lon, lat) coordinates of the
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
def equidistant_gnomonic_map(x, y, N, M, R):
    # Half length of the cube
    a = R/np.sqrt(3.0)

    # Cartesian coordinates of the projected points
    X = np.zeros((N, M, nbfaces))
    Y = np.zeros((N, M, nbfaces))
    Z = np.zeros((N, M, nbfaces))      
   
    # Creates a grid in [-a,a]x[-a,a] using
    # the given values of x and y
    x, y = np.meshgrid(x, y, indexing='ij')
   
    # Auxiliary variables
    r2   = a**2 + x**2 + y**2
    r    = np.sqrt(r2)
    invr = 1.0/r
    xor  = invr*x
    yor  = invr*y
    aor  = invr*a   
    
    # Compute the Cartesian coordinates for each panel
    # with the aid of the auxiliary variables 

    # Panel 0
    X[:,:,0] =  aor
    Y[:,:,0] =  xor
    Z[:,:,0] =  yor

    # Panel 1
    X[:,:,1] = -xor
    Y[:,:,1] =  aor
    Z[:,:,1] =  yor

    # Panel 2
    X[:,:,2] = -aor
    Y[:,:,2] = -xor
    Z[:,:,2] =  yor
   
    # Panel 3
    X[:,:,3] =  xor
    Y[:,:,3] = -aor
    Z[:,:,3] =  yor      

    # Panel 4
    X[:,:,4] = -yor
    Y[:,:,4] =  xor
    Z[:,:,4] =  aor

    # Panel 5
    X[:,:,5] =  yor
    Y[:,:,5] =  xor
    Z[:,:,5] = -aor         

    # Convert to spherical coordinates
    lon, lat = sphgeo.cart2sph(X, Y, Z)

    return X, Y, Z, lon, lat

####################################################################################
# Given a panel, this routine computes the inverse of the equidistant gnomonic map
####################################################################################
def inverse_equidistant_gnomonic_map(X, Y, Z, panel):
    # Half length of the cube
    a = 1.0/np.sqrt(3.0)
    if panel == 0:
        r = a/X
        x = Y*r
        y = Z*r
    elif panel == 1:
        r =  a/Y
        x = -X*r
        y =  Z*r
    elif panel == 2:
        r = -a/X
        x = -Y*r
        y =  Z*r
    elif panel == 3:
        r = -a/Y
        x =  X*r
        y =  Z*r
    elif panel == 4:
        r =  a/Z
        x =  Y*r
        y = -X*r
    elif panel == 5:
        r = -a/Z
        x =  Y*r
        y =  X*r
    else:
        print("ERROR: invalid panel.")
        exit()

    return x, y


####################################################################################
# Tangent vector of the equidistant gnomonic mapping in x dir
####################################################################################
def equidistant_tg_xdir(x, y, N, M, R):
    # Half length of the cube
    a = R/np.sqrt(3.0)

    # Cartesian coordinates of the tangent vector
    X = np.zeros((N, M, nbfaces))
    Y = np.zeros((N, M, nbfaces))
    Z = np.zeros((N, M, nbfaces))      
   
    # Creates a grid in [-a,a]x[-a,a] using
    # the given values of x and y
    x, y = np.meshgrid(x, y, indexing='ij')
   
    # Auxiliary variables
    r2   = a**2 + x**2 + y**2
    r32  = np.sqrt(r2)**3
    invr = R/r32
    ax = a*x
    xy = x*y
    y2 = y*y
    a2 = a*a

    # Compute the Cartesian coordinates of the tangent vector for each panel
    # with the aid of the auxiliary variables 

    # Panel 0
    X[:,:,0] = -ax*invr
    Y[:,:,0] =  (a2 + y2)*invr
    Z[:,:,0] = -xy*invr

    # Panel 1
    X[:,:,1] = -(a2 + y2)*invr
    Y[:,:,1] = -ax*invr
    Z[:,:,1] = -xy*invr

    # Panel 2
    X[:,:,2] =  ax*invr
    Y[:,:,2] = -(a2 + y2)*invr
    Z[:,:,2] = -xy*invr
   
    # Panel 3
    X[:,:,3] =  (a2 + y2)*invr
    Y[:,:,3] =  ax*invr
    Z[:,:,3] = -xy*invr 

    # Panel 4
    X[:,:,4] =  xy*invr
    Y[:,:,4] =  (a2 + y2)*invr
    Z[:,:,4] = -ax*invr

    # Panel 5
    X[:,:,5] = -xy*invr
    Y[:,:,5] =  (a2 + y2)*invr
    Z[:,:,5] =  ax*invr

    return X, Y, Z

####################################################################################
# Tangent vector of the equidistant gnomonic mapping in y dir
####################################################################################
def equidistant_tg_ydir(x, y, N, M, R):
    # Half length of the cube
    a = R/np.sqrt(3.0)

    # Cartesian coordinates of the tangent vector
    X = np.zeros((N, M, nbfaces))
    Y = np.zeros((N, M, nbfaces))
    Z = np.zeros((N, M, nbfaces))      
   
    # Creates a grid in [-a,a]x[-a,a] using
    # the given values of x and y
    x, y = np.meshgrid(x, y, indexing='ij')
   
    # Auxiliary variables
    r2   = a**2 + x**2 + y**2
    r32  = np.sqrt(r2)**3
    invr = R/r32
    ay = a*y
    xy = x*y
    x2 = x*x
    a2 = a*a
    
    # Compute the Cartesian coordinates of the tangent vector for each panel
    # with the aid of the auxiliary variables 

    # Panel 0
    X[:,:,0] = -ay*invr
    Y[:,:,0] = -xy*invr
    Z[:,:,0] = (a2 + x2)*invr

    # Panel 1
    X[:,:,1] =  xy*invr
    Y[:,:,1] = -ay*invr
    Z[:,:,1] =  (a2 + x2)*invr

    # Panel 2
    X[:,:,2] =  ay*invr
    Y[:,:,2] =  xy*invr
    Z[:,:,2] = (a2 + x2)*invr
   
    # Panel 3
    X[:,:,3] = -xy*invr
    Y[:,:,3] =  ay*invr
    Z[:,:,3] = (a2 + x2)*invr 

    # Panel 4
    X[:,:,4] = -(a2 + x2)*invr
    Y[:,:,4] = -xy*invr
    Z[:,:,4] = -ay *invr

    # Panel 5
    X[:,:,5] =  (a2 + x2)*invr
    Y[:,:,5] = -xy*invr
    Z[:,:,5] =  ay*invr

    return X, Y, Z

####################################################################################
# Tangent vector of the equiangular gnomonic mapping in x dir
####################################################################################
def equiangular_tg_xdir(x, y, N, M, R):
    # Half length of the cube
    a = R/np.sqrt(3.0)

    X, Y, Z = equidistant_tg_xdir(a*np.tan(x), a*np.tan(y), N, M, R)
    
    x, y = np.meshgrid(x, y, indexing='ij')
    
    cosx = np.cos(x)
    cos2x = cosx*cosx
    
    for p in range(0, nbfaces): 
        X[:,:,p], Y[:,:,p], Z[:,:,p] = a*X[:,:,p]/cos2x, a*Y[:,:,p]/cos2x, a*Z[:,:,p]/cos2x

    return X, Y, Z

####################################################################################
# Tangent vector of the equiangular gnomonic mapping in y dir
####################################################################################
def equiangular_tg_ydir(x, y, N, M, R):
    # Half length of the cube
    a = R/np.sqrt(3.0)

    X, Y, Z = equidistant_tg_ydir(a*np.tan(x), a*np.tan(y), N, M, R)

    x, y = np.meshgrid(x, y, indexing='ij')
 
    cosy = np.cos(y)
    cos2y = cosy*cosy
    for p in range(0, nbfaces): 
        X[:,:,p], Y[:,:,p], Z[:,:,p] = a*X[:,:,p]/cos2y, a*Y[:,:,p]/cos2y, a*Z[:,:,p]/cos2y

    return X, Y, Z

####################################################################################
# Compute the metric tensor of the cubed-sphere mapping
# Inputs: 1d arrays x and y containing the cube face coordinates, the cube   
# projection and sphere radius R.
# Output: metric tensor G on the grid (x,y)
# Reference: Rancic et al 1996
####################################################################################
def metric_tensor(x, y, R, projection):
    # Half of cube length
    a = R/np.sqrt(3.0)
    x, y = np.meshgrid(x, y, indexing='ij')
    #print(x)
    if projection == 'gnomonic_equidistant':
        r = np.sqrt(a*a + x*x + y*y)
        G = R*R*a/r**3
    elif projection == 'gnomonic_equiangular' or 'conformal':
        tanx, tany = np.tan(x), np.tan(y)
        r = np.sqrt(1 + tanx*tanx + tany*tany)
        G = R*R/r**3
        G = G/(np.cos(x)*np.cos(y))**2
    else:
        print("Error in metric_tensor from sphgeo.py: Invalid map projection")
        exit()
    return G

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
def conformal_map(ξ, η, N, M):
    #print(ξ)
    swap = ξ
    ξ = -η
    η = swap
    ξ, η = np.meshgrid(ξ, η)
    #print(np.shape(ξ))
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
    #print(np.shape(X))
    # Cartesian coordinates of the projected points
    x = np.zeros((N, M, nbfaces))
    y = np.zeros((N, M, nbfaces))
    z = np.zeros((N, M, nbfaces))

    # Panel 4
    x[:,:,4] = X
    y[:,:,4] = Y
    z[:,:,4] = Z

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
