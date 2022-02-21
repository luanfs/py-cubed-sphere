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
import numpy as np

##########################################################
# Open a netcdf file plot its scalar fields
##########################################################
fig_format = 'png' # Figure format

def plot_netcdf4(filename, transformation, map_projection):
   # Check if the file exists
   if not os.path.isfile(filename):
      print('ERROR: could not find the file',filename)
      exit()

   # Open the netcdf4 file
   dataset = nc.Dataset(filename,'r')
   lats = dataset['lat'][:]
   lons = dataset['lon'][:]
   time = dataset['time'][:]   
   data = []

   # Create a CS mesh
   N = 1
   cs_grid = cubed_sphere(N, transformation,'False')
   
   # Get the variables
   for var in dataset.variables.values():
      data.append(var)

   print("--------------------------------------------------------\n")
   print("Plotting fields from ", dataset.title, "netcdf file.")

   # Figure quality
   dpi = 100

   # Plot the fields
   for l in range(3, len(data)):
      for t in range(0, len(time)):
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
         
         if len(time)>1:
            print("\nPlotting scalar field",data[l].name,"at time ",time[t],"..." )
         else:
            print("\nPlotting scalar field",data[l].name,"..." ) 
                        
         # Plot cubed-sphere
         for p in range(0, 6):
            lon = cs_grid.vertices.lon[:,:,p]*rad2deg
            lat = cs_grid.vertices.lat[:,:,p]*rad2deg

            # Plot vertices
            A_lon, A_lat = lon[0:cs_grid.N, 0:cs_grid.N], lat[0:cs_grid.N, 0:cs_grid.N]
            A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)
      
            B_lon, B_lat = lon[1:cs_grid.N+1, 0:cs_grid.N], lat[1:cs_grid.N+1, 0:cs_grid.N]
            B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)
      
            C_lon, C_lat = lon[1:cs_grid.N+1, 1:cs_grid.N+1], lat[1:cs_grid.N+1, 1:cs_grid.N+1]
            C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

            D_lon, D_lat = lon[0:cs_grid.N  , 1:cs_grid.N+1], lat[0:cs_grid.N  , 1:cs_grid.N+1]
            D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

            plt.plot([A_lon, B_lon], [A_lat, B_lat],linewidth=1, color='black', transform=ccrs.Geodetic())
            plt.plot([B_lon, C_lon], [B_lat, C_lat],linewidth=1, color='black', transform=ccrs.Geodetic())
            plt.plot([C_lon, D_lon], [C_lat, D_lat],linewidth=1, color='black', transform=ccrs.Geodetic())
            plt.plot([D_lon, A_lon], [D_lat, A_lat],linewidth=1, color='black', transform=ccrs.Geodetic())

         ax.coastlines()
         if map_projection == 'mercator':
            ax.gridlines(draw_labels=True)
    
         # Plot the scalar field
         plt.contourf(lons,lats,data[l][:,:,t].T, cmap='jet', transform=ccrs.PlateCarree())

         # Plot colorbar
         if map_projection == 'mercator':
            plt.colorbar(orientation='horizontal',fraction=0.046, pad=0.04)
         elif map_projection == 'sphere':
            plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04)   

         # Save the figure
         if len(time)>1:
            name = dataset.title+"_"+data[l].name+"_t"+str(int(time[t]))+"_"+map_projection+'.'+fig_format
         else:
            name = dataset.title+"_"+data[l].name+"_"+map_projection+'.'+fig_format 
         plt.savefig(graphdir+name, format=fig_format)   
         print('Figure has been saved in '+graphdir+name)
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
elif test_case == 3:
   tc = "divergence"
   
# Name of netcdf4 file to be opened
filename = datadir+transformation+"_"+str(N)+"_"+tc+".nc"

# Read and plot netcdf4 file
plot_netcdf4(filename, transformation, map_projection)
