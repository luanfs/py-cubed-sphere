##########################################################
#
# Script for plotting netcdf files
#
# Luan da Fonseca Santos (luan.santos@usp.br)
# February 2022
##########################################################

# Source code directory
srcdir = "src/"

import sys
sys.path.append(srcdir)

# Imports
from configuration import get_parameters
from cs_datastruct import cubed_sphere
from constants     import graphdir, datadir, rad2deg
import os.path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4 as nc

##########################################################
# Open a netcdf file plot its scalar fields
##########################################################
fig_format = 'pdf' # Figure format

def plot_netcdf4(filename, transformation, map_projection):
   # Check if the file exists
   if not os.path.isfile(filename):
      print('ERROR: could not find the file',filename)
      exit()

   # Open the netcdf4 file
   dataset = nc.Dataset(filename,'r')
   lats = dataset['lat'][:]
   lons = dataset['lon'][:]
   data = []

   # Create a CS mesh
   N = 1
   cs_grid = cubed_sphere(N, transformation)

   # Get the variables
   for var in dataset.variables.values():
      data.append(var)

   print("--------------------------------------------------------\n")
   print("Plotting fields from ", dataset.title, "netcdf file.")
   
   # Figure quality
   dpi = 100
  
   # Plot the fields
   for l in range(2,len(data)):
      # Map projection
      if map_projection == "mercator":
         plateCr = ccrs.PlateCarree()
         plt.figure(figsize=(1832/dpi, 977/dpi), dpi=dpi)
      elif map_projection == "sphere":
         plateCr = ccrs.Orthographic(central_longitude=-60.0, central_latitude=0.0)      
         plt.figure(figsize=(800/dpi, 800/dpi), dpi=dpi)
      else:
         print("ERROR: invalid map_projection")
         exit()

      plateCr._threshold = plateCr._threshold/10.
      ax = plt.axes(projection=plateCr)
      ax.stock_img()
      print("\nPlotting scalar field",data[l].name,"...")

      for p in range(0,6):
         lon = cs_grid.vertices.lon[:,:,p]*rad2deg
         lat = cs_grid.vertices.lat[:,:,p]*rad2deg
         for i in range(0,cs_grid.N):
            for j in range(0,cs_grid.N):
               A_lon, A_lat = lon[i  ,j  ], lat[i  ,j  ]
               B_lon, B_lat = lon[i+1,j  ], lat[i+1,j  ]
               C_lon, C_lat = lon[i+1,j+1], lat[i+1,j+1]
               D_lon, D_lat = lon[i  ,j+1], lat[i  ,j+1]
               plt.plot([A_lon, B_lon], [A_lat, B_lat], linewidth=1, color='black', transform=ccrs.Geodetic())
               plt.plot([B_lon, C_lon], [B_lat, C_lat], linewidth=1, color='black', transform=ccrs.Geodetic())
               plt.plot([C_lon, D_lon], [C_lat, D_lat], linewidth=1, color='black', transform=ccrs.Geodetic())
               plt.plot([D_lon, A_lon], [D_lat, A_lat], linewidth=1, color='black', transform=ccrs.Geodetic())

      ax.coastlines()
      if map_projection == 'mercator':
         ax.gridlines(draw_labels=True)
    
      # Plot the scalar field
      plt.contourf(lons,lats,data[l][:,:].T, cmap='jet', transform=ccrs.PlateCarree())

      # Plot colorbar
      if map_projection == 'mercator':
         plt.colorbar(orientation='horizontal',fraction=0.046, pad=0.04)
      elif map_projection == 'sphere':
         plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04)   

      # Save the figure
      plt.savefig(graphdir+dataset.title+"_"+data[l].name+"_"+map_projection+'.'+fig_format, format=fig_format)   
      print('Figure has been saved in '+graphdir+dataset.title+"_"+data[l].name+"_"+ map_projection+'.'+fig_format+"")
      #plt.show()
      plt.close()
   
   dataset.close()
   print("--------------------------------------------------------")      

###############################################################
print("Script for plotting netcdf files")
# Get the parameters
N, transformation, showonscreen, test_case, map_projection = get_parameters()

# Name test case
if test_case == 2:
   tc = "grid_quality"
# Name of netcdf4 file to be opened
filename = datadir+transformation+"_"+str(N)+"_"+tc+".nc"

# Read and plot netcdf4 file
plot_netcdf4(filename, transformation, map_projection)
