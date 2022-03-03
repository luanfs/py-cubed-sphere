####################################################################################
# 
# Module for plotting routines.
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from cs_datastruct import cubed_sphere
from constants import*
from sphgeo import*

####################################################################################
# This routine plots the cubed-sphere grid.
####################################################################################
fig_format = 'pdf' # Figure format
def plot_grid(grid, map_projection):
   # Figure resolution
   dpi = 100

   # Color of each cubed panel
   colors = ('blue','red','blue','red','green','green')

   print("--------------------------------------------------------")
   print('Plotting '+grid.name+' cubed-sphere grid using '+map_projection+' projection...')   
   if map_projection == "mercator":
      plateCr = ccrs.PlateCarree()
      plt.figure(figsize=(1832/dpi, 977/dpi), dpi=dpi)
   elif map_projection == 'sphere':
      plateCr = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)   
      plt.figure(figsize=(800/dpi, 800/dpi), dpi=dpi)
   else:
      print('ERROR: Invalid map projection.')
      exit()

   plateCr._threshold = plateCr._threshold/10.
   ax = plt.axes(projection=plateCr)
   ax.stock_img()
  
   for p in range(0, nbfaces):
      # Vertices
      lon = grid.vertices.lon[:,:,p]*rad2deg
      lat = grid.vertices.lat[:,:,p]*rad2deg

      # Centers
      #lonc = grid.center.lon[:,:,p]*rad2deg
      #latc = grid.center.lat[:,:,p]*rad2deg

      # Plot vertices
      A_lon, A_lat = lon[0:grid.N, 0:grid.N], lat[0:grid.N, 0:grid.N]
      A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)
      
      B_lon, B_lat = lon[1:grid.N+1, 0:grid.N], lat[1:grid.N+1, 0:grid.N]
      B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)
      
      C_lon, C_lat = lon[1:grid.N+1, 1:grid.N+1], lat[1:grid.N+1, 1:grid.N+1]
      C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

      D_lon, D_lat = lon[0:grid.N  , 1:grid.N+1], lat[0:grid.N  , 1:grid.N+1]
      D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

      plt.plot([A_lon, B_lon], [A_lat, B_lat],linewidth=1, color=colors[p], transform=ccrs.Geodetic())
      plt.plot([B_lon, C_lon], [B_lat, C_lat],linewidth=1, color=colors[p], transform=ccrs.Geodetic())
      plt.plot([C_lon, D_lon], [C_lat, D_lat],linewidth=1, color=colors[p], transform=ccrs.Geodetic())
      plt.plot([D_lon, A_lon], [D_lat, A_lat],linewidth=1, color=colors[p], transform=ccrs.Geodetic())
 
      # Plot centers
      #if map_projection == 'mercator':
      #   center_lon, center_lat = lonc, latc
      #   center_lon, center_lat = np.ndarray.flatten(center_lon),np.ndarray.flatten(center_lat)
      #   print(np.shape(center_lon))
      #   plt.plot(center_lon, center_lat, marker='+',color = 'black')

   if map_projection == 'mercator':
      ax.gridlines(draw_labels=True)
      
   # Save the figure
   #plt.show()
   plt.savefig(graphdir+grid.name+"_"+map_projection+'.'+fig_format, format=fig_format)
   print('Figure has been saved in '+graphdir+grid.name+"_"+map_projection+'.'+fig_format)
   print("--------------------------------------------------------\n")    
   plt.close()

####################################################################################
# This routine plots the scalar field "field" given in the latlon_grid
####################################################################################
def plot_scalar_field(field, name, cs_grid, latlon_grid, map_projection):
   print("Plotting scalar field",name,"...")

   # Figure quality
   dpi = 100
   
   if map_projection == "mercator":
      plateCr = ccrs.PlateCarree()
      plt.figure(figsize=(1832/dpi, 977/dpi), dpi=dpi)
   elif map_projection == "sphere":
      plateCr = ccrs.Orthographic(central_longitude=-60.0, central_latitude=0.0)      
      plt.figure(figsize=(800/dpi, 800/dpi), dpi=dpi)
   else:
      print('ERROR: Invalid projection.')
      exit()

   plateCr._threshold = plateCr._threshold/10.
   ax = plt.axes(projection=plateCr)
   ax.stock_img()

   # Plot auxiliary cubed-sphere grid
   if cs_grid.N <= 10:
     cs_grid_aux = cs_grid
   else:
     cs_grid_aux = cubed_sphere(1, cs_grid.projection, False, False)
   
   for p in range(0, nbfaces):
      lons = cs_grid_aux.vertices.lon[:,:,p]*rad2deg
      lats = cs_grid_aux.vertices.lat[:,:,p]*rad2deg

      # Plot vertices
      A_lon, A_lat = lons[0:cs_grid_aux.N, 0:cs_grid_aux.N], lats[0:cs_grid_aux.N, 0:cs_grid_aux.N]
      A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)
      
      B_lon, B_lat = lons[1:cs_grid_aux.N+1, 0:cs_grid_aux.N], lats[1:cs_grid_aux.N+1, 0:cs_grid_aux.N]
      B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)
      
      C_lon, C_lat = lons[1:cs_grid_aux.N+1, 1:cs_grid_aux.N+1], lats[1:cs_grid_aux.N+1, 1:cs_grid_aux.N+1]
      C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

      D_lon, D_lat = lons[0:cs_grid_aux.N  , 1:cs_grid_aux.N+1], lats[0:cs_grid_aux.N  , 1:cs_grid_aux.N+1]
      D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

      plt.plot([A_lon, B_lon], [A_lat, B_lat],linewidth=1, color='black', transform=ccrs.Geodetic())
      plt.plot([B_lon, C_lon], [B_lat, C_lat],linewidth=1, color='black', transform=ccrs.Geodetic())
      plt.plot([C_lon, D_lon], [C_lat, D_lat],linewidth=1, color='black', transform=ccrs.Geodetic())
      plt.plot([D_lon, A_lon], [D_lat, A_lat],linewidth=1, color='black', transform=ccrs.Geodetic())

   ax.coastlines()

   if map_projection == 'mercator':
      ax.gridlines(draw_labels=True)

   # Plot the scalar field
   plt.contourf(latlon_grid.lon*rad2deg,latlon_grid.lat*rad2deg,field,cmap='jet', transform=ccrs.PlateCarree())

   # Plot colorbar
   if map_projection == 'mercator':
      plt.colorbar(orientation='horizontal',fraction=0.046, pad=0.04)
   elif map_projection == 'sphere':
      plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04)   

   # Save the figure
   plt.savefig(graphdir+cs_grid.name+"_"+name+"_"+map_projection+'.'+fig_format, format=fig_format)   

   print('Figure has been saved in '+graphdir+cs_grid.name+"_"+name+"_"+map_projection+'.'+fig_format+"\n")
   plt.close()  

####################################################################################
# Create a netcdf file using the fields given in the list fields_ll
####################################################################################  
def open_netcdf4(fields_cs, ts, name):
   # Open a netcdf file
   print("--------------------------------------------------------")
   print("Creating netcdf file "+datadir+name+".nc")
   data = nc.Dataset(datadir+name+".nc", mode='w', format='NETCDF4_CLASSIC')

   # Name
   data.title = name
   
   # Size of lat-lon grid
   m, n = Nlon, Nlat

   # Number of time steps
   nt = len(ts)

   # Create dimensions (horizontal, time,...)   
   time = data.createDimension('time', nt)
   lat  = data.createDimension('lat' , n)
   lon  = data.createDimension('lon' , m)

   # Create variables (horizontal, time,...)   
   times = data.createVariable('time' ,'f8',('time',))
   lats  = data.createVariable('lat'  ,'f8',('lat',))
   lons  = data.createVariable('lon'  ,'f8',('lon',))

   # Create field variables using the fields given in the list
   variables = [None]*len(fields_cs)
   for l in range(0, len(fields_cs)):
      variables[l] = data.createVariable(fields_cs[l].name, 'f8', ('lon','lat','time'))
      #print(fields_cs[l].name)

   # Values
   times[:] = ts    
   lats[:]  = np.linspace( -90.0,  90.0, n)
   lons[:]  = np.linspace(-180.0, 180.0, m)

   print("Done.")
   print("--------------------------------------------------------\n")
   return data
   

####################################################################################
# Create a netcdf file using the fields given in the list fields_ll
####################################################################################  
def save_grid_netcdf4(grid):
   # Open a netcdf file
   print("--------------------------------------------------------")
   print("Creating grid netcdf file "+griddir+grid.name+".nc")
   griddata = nc.Dataset(grid.netcdfdata_filename, mode='w', format='NETCDF4_CLASSIC')

   # Name
   griddata.title = grid.name 
   
   # Cell in each panel axis
   n = grid.N

   # Create dimensions
   # Panels 
   panel = griddata.createDimension('panel', nbfaces)

   # Panel xy coordinates
   ix = griddata.createDimension('ix', n+1)
   jy = griddata.createDimension('jy', n+1)

   ix2 = griddata.createDimension('ix2', n)
   jy2 = griddata.createDimension('jy2', n)
   
   # R3 dimension + S2 (sphere in R3) dimension (x,y,z + lat,lon coordinates)
   coorddim = griddata.createDimension('coorddim', 5)

   # Number of edges in a cell
   ed = griddata.createDimension('ed', 4)

   # Create variables
   vertices = griddata.createVariable('vertices', 'f8', ('ix' , 'jy' , 'panel', 'coorddim'))
   centers  = griddata.createVariable('centers',  'f8', ('ix2', 'jy2', 'panel', 'coorddim'))

   # Geometric properties
   areas           = griddata.createVariable('areas'          , 'f8', ('ix2', 'jy2', 'panel'))
   length_x        = griddata.createVariable('length_x'       , 'f8', ('ix2', 'jy' , 'panel'))
   length_y        = griddata.createVariable('length_y'       , 'f8', ('ix' , 'jy2', 'panel'))
   length_diag     = griddata.createVariable('length_diag'    , 'f8', ('ix2', 'jy2', 'panel'))
   length_antidiag = griddata.createVariable('length_antidiag', 'f8', ('ix2', 'jy2', 'panel'))
   angles          = griddata.createVariable('angles'         , 'f8', ('ix2', 'jy2', 'panel', 'ed'))   

   # Values attribution
   vertices[:,:,:,0] = grid.vertices.x
   vertices[:,:,:,1] = grid.vertices.y
   vertices[:,:,:,2] = grid.vertices.z
   vertices[:,:,:,3] = grid.vertices.lon
   vertices[:,:,:,4] = grid.vertices.lat

   centers[:,:,:,0] = grid.centers.x
   centers[:,:,:,1] = grid.centers.y
   centers[:,:,:,2] = grid.centers.z
   centers[:,:,:,3] = grid.centers.lon
   centers[:,:,:,4] = grid.centers.lat

   angles[:,:,:,0] = grid.angles[:,:,:,0]
   angles[:,:,:,1] = grid.angles[:,:,:,1]
   angles[:,:,:,2] = grid.angles[:,:,:,2]
   angles[:,:,:,3] = grid.angles[:,:,:,3]

   areas[:,:,:]           = grid.areas[:,:,:]
   length_x[:,:,:]        = grid.length_x[:,:,:]
   length_y[:,:,:]        = grid.length_y[:,:,:]
   length_diag[:,:,:]     = grid.length_diag[:,:,:]
   length_antidiag[:,:,:] = grid.length_antidiag[:,:,:]

   griddata.close()
   print("Done.")
   print("--------------------------------------------------------\n")
