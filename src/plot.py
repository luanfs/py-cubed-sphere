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
# Figure format
#fig_format = 'eps'
fig_format = 'png'
def plot_grid(grid, map_projection):
    # Figure resolution
    dpi = 100

    # Interior cells index (we are ignoring ghost cells)
    i0   = grid.i0
    iend = grid.iend
    j0   = grid.j0
    jend = grid.jend
    Nt = grid.nghost + grid.N

    # Plot ghost cells?
    plot_gc = False

    # Plot C grid points?
    plot_cgrid = False

    # Color of each cubed panel
    colors = ('blue','red','blue','red','green','green')
    colors = ('black','black','black','black','black','black')

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
        lonc = grid.centers.lon[:,:,p]*rad2deg
        latc = grid.centers.lat[:,:,p]*rad2deg

        # Edges in x direction
        lon_edx = grid.edx.lon[:,:,p]*rad2deg
        lat_edx = grid.edx.lat[:,:,p]*rad2deg

        # Edges in y direction
        lon_edy = grid.edy.lon[:,:,p]*rad2deg
        lat_edy = grid.edy.lat[:,:,p]*rad2deg

        # Plot vertices geodesic
        A_lon, A_lat = lon[i0:iend, j0:jend], lat[i0:iend, j0:jend]
        A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)

        B_lon, B_lat = lon[i0+1:iend+1, j0:jend], lat[i0+1:iend+1, j0:jend]
        B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

        C_lon, C_lat = lon[i0+1:iend+1, j0+1:jend+1], lat[i0+1:iend+1, j0+1:jend+1]
        C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

        D_lon, D_lat = lon[i0:iend, j0+1:jend+1], lat[i0:iend, j0+1:jend+1]
        D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

        plt.plot([A_lon, B_lon], [A_lat, B_lat], '-', linewidth=0.75, color=colors[p], transform=ccrs.Geodetic())
        plt.plot([B_lon, C_lon], [B_lat, C_lat], '-', linewidth=0.75, color=colors[p], transform=ccrs.Geodetic())
        plt.plot([C_lon, D_lon], [C_lat, D_lat], '-', linewidth=0.75, color=colors[p], transform=ccrs.Geodetic())
        plt.plot([D_lon, A_lon], [D_lat, A_lat], '-', linewidth=0.75, color=colors[p], transform=ccrs.Geodetic())

        if plot_gc:
            if p==0:
                # Plot centers geodesics
                A_lon, A_lat = lonc[0:Nt-1, 0:Nt-1], latc[0:Nt-1, 0:Nt-1]
                A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)

                B_lon, B_lat = lonc[1:Nt, 0:Nt-1], latc[1:Nt, 0:Nt-1]
                B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

                C_lon, C_lat = lonc[1:Nt, 1:Nt], latc[1:Nt, 1:Nt]
                C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

                D_lon, D_lat = lonc[0:Nt-1, 1:Nt], latc[0:Nt-1, 1:Nt]
                D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

                plt.plot([A_lon, B_lon], [A_lat, B_lat], '--', linewidth=0.75, color='yellow', transform=ccrs.Geodetic())
                plt.plot([B_lon, C_lon], [B_lat, C_lat], '--', linewidth=0.75, color='yellow', transform=ccrs.Geodetic())
                plt.plot([C_lon, D_lon], [C_lat, D_lat], '--', linewidth=0.75, color='yellow', transform=ccrs.Geodetic())
                plt.plot([D_lon, A_lon], [D_lat, A_lat], '--', linewidth=0.75, color='yellow', transform=ccrs.Geodetic())

                # Plot centers
                A_lon, A_lat = lonc[:,:], latc[:,:]
                for i in range(0, Nt):
                    for j in range(0, Nt):
                        if (i>=i0 and i<iend) and (j>=j0 and j<jend):
                           plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'green', transform=ccrs.Geodetic())
                        elif i==j or i+j==grid.N+grid.nghost-1:
                           plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'magenta', transform=ccrs.Geodetic())
                        else:
                           plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'yellow', transform=ccrs.Geodetic())

            elif p==1 or p==3:
                # Plot centers geodesics
                A_lon, A_lat = lonc[0:Nt-1, 0:Nt-1], latc[0:Nt-1, 0:Nt-1]
                A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)

                B_lon, B_lat = lonc[1:Nt, 0:Nt-1], latc[1:Nt, 0:Nt-1]
                B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

                C_lon, C_lat = lonc[1:Nt, 1:Nt], latc[1:Nt, 1:Nt]
                C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

                D_lon, D_lat = lonc[0:Nt-1, 1:Nt], latc[0:Nt-1, 1:Nt]
                D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

                # Plot centers
                A_lon, A_lat = lonc[:,:], latc[:,:]
                for i in range(0, Nt):
                    for j in range(0, Nt):
                        if p==1 and (i>=i0 and i<i0+grid.nghost_right) and (j<j0 or j>=jend):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'cyan', transform=ccrs.Geodetic())
                        if p==3 and (i>=iend-grid.nghost_right and i<iend) and (j<j0 or j>=jend):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'cyan', transform=ccrs.Geodetic())

            if p != 0:
                # Plot centers
                A_lon, A_lat = lonc[:,:], latc[:,:]
                for i in range(i0, iend):
                    for j in range(j0, jend ):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'black', transform=ccrs.Geodetic())

            plt.xlim(-80,80)
            plt.ylim(-80,80)

        elif plot_cgrid:
            if p==0:
                A_lon, A_lat = lonc[:,:], latc[:,:]
                for i in range(i0, iend):
                    for j in range(j0, jend):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'black', transform=ccrs.Geodetic())

                A_lon, A_lat = lon_edx[:,:], lat_edx[:,:]
                for i in range(i0, iend+1):
                    for j in range(j0, jend):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'blue', transform=ccrs.Geodetic())

                A_lon, A_lat = lon_edy[:,:], lat_edy[:,:]
                for i in range(i0, iend):
                    for j in range(j0, jend+1):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'red', transform=ccrs.Geodetic())


                plt.xlim(-50,50)
                plt.ylim(-50,50)


    if map_projection == 'mercator':
        ax.gridlines(draw_labels=True)
    # Save the figure
    plt.savefig(graphdir+grid.name+"_"+map_projection+'.'+fig_format, format=fig_format)
    print('Figure has been saved in '+graphdir+grid.name+"_"+map_projection+'.'+fig_format)
    print("--------------------------------------------------------\n")
    plt.close()

####################################################################################
# This routine plots the scalar field "field" given in the latlon_grid
####################################################################################
def plot_scalar_field(field, name, cs_grid, latlon_grid, map_projection, \
                      colormap=None, qmin=None, qmax=None, filename=None):
    print("Plotting scalar field",name,"...")

    # Figure quality
    dpi = 100

    # Map projection
    if map_projection == "mercator":
        plateCr = ccrs.PlateCarree()
        plt.figure(figsize=(1832/dpi, 977/dpi), dpi=dpi)
    elif map_projection == "sphere":
        plateCr = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)
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

    # Interior cells index (we are ignoring ghost cells)
    i0   = cs_grid_aux.i0
    iend = cs_grid_aux.iend
    j0   = cs_grid_aux.j0
    jend = cs_grid_aux.jend

    # Plot CS grid
    for p in range(0, nbfaces):
        lons = cs_grid_aux.vertices.lon[:,:,p]*rad2deg
        lats = cs_grid_aux.vertices.lat[:,:,p]*rad2deg

        # Plot vertices
        A_lon, A_lat = lons[i0:iend, j0:jend], lats[i0:iend, j0:jend]
        A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)

        B_lon, B_lat = lons[i0+1:iend+1, j0:jend], lats[i0+1:iend+1, j0:jend]
        B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

        C_lon, C_lat = lons[i0+1:iend+1, j0+1:jend+1], lats[i0+1:iend+1, j0+1:jend+1]
        C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

        D_lon, D_lat = lons[i0:iend, j0+1:jend+1], lats[i0:iend, j0+1:jend+1]
        D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

        plt.plot([A_lon, B_lon], [A_lat, B_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())
        plt.plot([B_lon, C_lon], [B_lat, C_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())
        plt.plot([C_lon, D_lon], [C_lat, D_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())
        plt.plot([D_lon, A_lon], [D_lat, A_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())

    ax.coastlines()


    if map_projection == 'mercator':
        ax.gridlines(draw_labels=True)

    # check if colormap was given
    if not colormap:
        colormap = 'jet'

    # check if colorbar range was given
    if not qmin or not qmax:
        qmin = np.amin(field)
        qmax = np.amax(field)

    # add title
    if filename:
        plt.title(filename)

    # Plot the scalar field
    plt.contourf(latlon_grid.lon*rad2deg, latlon_grid.lat*rad2deg, field, cmap=colormap, levels = np.linspace(qmin, qmax, 101), transform=ccrs.PlateCarree())

    # Plot colorbar
    plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04,  format='%.1e')

    # Save the figure
    plt.savefig(graphdir+cs_grid.name+"_"+name+"_"+map_projection+'.'+fig_format, format=fig_format)

    print('Figure has been saved in '+graphdir+cs_grid.name+"_"+name+"_"+map_projection+'.'+fig_format+"\n")
    plt.close()

####################################################################################
# This routine plots the scalar field "field" and given in the latlon_grid
# and vector field (ulon, vlat) at edges midpoints on the cubed sphere
####################################################################################
def plot_scalar_and_vector_field(field, ulon_edx, vlat_edx, ulon_edy, vlat_edy, \
                                 name, title, cs_grid, latlon_grid, map_projection, \
                                 colormap, qmin, qmax):
    print("Plotting scalar field",name,"...")

    # Figure quality
    dpi = 100

    # Map projection
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

    # Interior cells index (we are ignoring ghost cells)
    i0   = cs_grid_aux.i0
    iend = cs_grid_aux.iend
    j0   = cs_grid_aux.j0
    jend = cs_grid_aux.jend

    # Plot CS grid
    for p in range(0, nbfaces):
        lons = cs_grid_aux.vertices.lon[:,:,p]*rad2deg
        lats = cs_grid_aux.vertices.lat[:,:,p]*rad2deg

        # Plot vertices
        A_lon, A_lat = lons[i0:iend, j0:jend], lats[i0:iend, j0:jend]
        A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)

        B_lon, B_lat = lons[i0+1:iend+1, j0:jend], lats[i0+1:iend+1, j0:jend]
        B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

        C_lon, C_lat = lons[i0+1:iend+1, j0+1:jend+1], lats[i0+1:iend+1, j0+1:jend+1]
        C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

        D_lon, D_lat = lons[i0:iend, j0+1:jend+1], lats[i0:iend, j0+1:jend+1]
        D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

        #plt.plot([A_lon, B_lon], [A_lat, B_lat], linewidth=0.5, color='black', transform=ccrs.Geodetic())
        #plt.plot([B_lon, C_lon], [B_lat, C_lat], linewidth=0.5, color='black', transform=ccrs.Geodetic())
        #plt.plot([C_lon, D_lon], [C_lat, D_lat], linewidth=0.5, color='black', transform=ccrs.Geodetic())
        #plt.plot([D_lon, A_lon], [D_lat, A_lat], linewidth=0.5, color='black', transform=ccrs.Geodetic())


    if map_projection == 'mercator':
        ax.gridlines(draw_labels=True)

    # Plot the scalar field
    plt.contourf(latlon_grid.lon*rad2deg, latlon_grid.lat*rad2deg, field, cmap=colormap, levels = np.linspace(qmin, qmax, 101), transform=ccrs.PlateCarree())
    #plt.contourf(latlon_grid.lon*rad2deg, latlon_grid.lat*rad2deg, field, cmap=colormap, levels = 101, transform=ccrs.PlateCarree())

    plt.title(title)

    # Plot colorbar
    plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04, format='%.1e')

    #ax.coastlines()

    # Plot vector field at edges midpoints
    if map_projection == 'mercator':
        i0   = cs_grid.i0
        iend = cs_grid.iend
        j0   = cs_grid.j0
        jend = cs_grid.jend
        N = cs_grid.N
        if cs_grid.N>10:
            step = int(N/5)
        else:
            step = 1

        for p in range(0, nbfaces):
            # Edges in x direction
            lon_edx = cs_grid.edx.lon[i0:iend+1:step, j0:jend:step,p]*rad2deg
            lat_edx = cs_grid.edx.lat[i0:iend+1:step, j0:jend:step,p]*rad2deg
            lon_edx, lat_edx = np.ndarray.flatten(lon_edx), np.ndarray.flatten(lat_edx)

            # Edges in y direction
            lon_edy = cs_grid.edy.lon[i0:iend:step, j0:jend+1:step,p]*rad2deg
            lat_edy = cs_grid.edy.lat[i0:iend:step, j0:jend+1:step,p]*rad2deg
            lon_edy, lat_edy = np.ndarray.flatten(lon_edy), np.ndarray.flatten(lat_edy)

            # Vector field at edges in x direction
            vec_edx_lon = ulon_edx[0:N+1:step,0:N:step,p]
            vec_edx_lat = vlat_edx[0:N+1:step,0:N:step,p]
            vec_edx_lat, vec_edx_lon = np.ndarray.flatten(vec_edx_lat), np.ndarray.flatten(vec_edx_lon)

            # Vector field at edges in y direction
            vec_edy_lon = ulon_edy[0:N:step,0:N+1:step,p]
            vec_edy_lat = vlat_edy[0:N:step,0:N+1:step,p]
            vec_edy_lat, vec_edy_lon = np.ndarray.flatten(vec_edy_lat), np.ndarray.flatten(vec_edy_lon)

            # Plot tangent vector at edge points in x dir
            plt.quiver(lon_edx, lat_edx, vec_edx_lon, vec_edx_lat, width = 0.001)
            #plt.quiver(lon_edy, lat_edy, vec_edy_lon, vec_edy_lat, width = 0.001)

    # Save the figure
    #ax.coastlines()
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
    n = grid.N+grid.nghost

    # Grid spacing
    dx  = griddata.createVariable('dx','f8')
    dy  = griddata.createVariable('dy','f8')
    dx[:] = grid.dx
    dy[:] = grid.dy

    # Some integers
    nghost          = griddata.createVariable('nghost'   ,'i4')
    nghost_left     = griddata.createVariable('nghost_left'   ,'i4')
    nghost_right    = griddata.createVariable('nghost_right'   ,'i4')
    N     = griddata.createVariable('N'   ,'i4')
    i0    = griddata.createVariable('i0'  ,'i4')
    iend  = griddata.createVariable('iend','i4')
    j0    = griddata.createVariable('j0'  ,'i4')
    jend  = griddata.createVariable('jend','i4')

    nghost[:] = grid.nghost
    nghost_left[:]  = grid.nghost_left
    nghost_right[:] = grid.nghost_right
    N[:]    = grid.N
    i0[:]   = grid.i0
    iend[:] = grid.iend
    j0[:]   = grid.j0
    jend[:] = grid.jend

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

    # R3 dimension
    r3dim = griddata.createDimension('r3dim', 3)

    # Number of edges in a cell
    ed = griddata.createDimension('ed', 4)

    # Create variables
    vertices = griddata.createVariable('vertices', 'f8', ('ix' , 'jy' , 'panel', 'coorddim'))
    centers  = griddata.createVariable('centers' , 'f8', ('ix2', 'jy2', 'panel', 'coorddim'))
    edx      = griddata.createVariable('edx'     , 'f8', ('ix' , 'jy2', 'panel', 'coorddim'))
    edy      = griddata.createVariable('edy'     , 'f8', ('ix2', 'jy' , 'panel', 'coorddim'))

    # Geometric properties
    areas                   = griddata.createVariable('areas'                , 'f8', ('ix2', 'jy2', 'panel'))
    length_edx              = griddata.createVariable('length_edx'           , 'f8', ('ix2', 'jy2', 'panel'))
    length_edy              = griddata.createVariable('length_edy'           , 'f8', ('ix2', 'jy2', 'panel'))

    metric_tensor_centers   = griddata.createVariable('metric_tensor_centers', 'f8', ('ix2', 'jy2', 'panel'))

    prod_ex_elon_edx          = griddata.createVariable('prod_ex_elon_edx'    , 'f8', ('ix', 'jy2' , 'panel'))
    prod_ex_elat_edx          = griddata.createVariable('prod_ex_elat_edx'    , 'f8', ('ix', 'jy2' , 'panel'))
    prod_ey_elon_edx          = griddata.createVariable('prod_ey_elon_edx'    , 'f8', ('ix', 'jy2' , 'panel'))
    prod_ey_elat_edx          = griddata.createVariable('prod_ey_elat_edx'    , 'f8', ('ix', 'jy2' , 'panel'))
    determinant_ll2contra_edx = griddata.createVariable('determinant_ll2contra_edx', 'f8', ('ix', 'jy2' , 'panel'))

    prod_ex_elon_edy          = griddata.createVariable('prod_ex_elon_edy'    , 'f8', ('ix2', 'jy' , 'panel'))
    prod_ex_elat_edy          = griddata.createVariable('prod_ex_elat_edy'    , 'f8', ('ix2', 'jy' , 'panel'))
    prod_ey_elon_edy          = griddata.createVariable('prod_ey_elon_edy'    , 'f8', ('ix2', 'jy' , 'panel'))
    prod_ey_elat_edy          = griddata.createVariable('prod_ey_elat_edy'    , 'f8', ('ix2', 'jy' , 'panel'))
    determinant_ll2contra_edy = griddata.createVariable('determinant_ll2contra_edy', 'f8', ('ix2', 'jy' , 'panel'))


    Xu   = griddata.createVariable('Xu'    , 'f8', ('ix', 'jy2' , 'panel'))
    Yv   = griddata.createVariable('Yv'    , 'f8', ('ix2', 'jy' , 'panel'))
    # Values attribution
    vertices[:,:,:,0] = grid.vertices.X
    vertices[:,:,:,1] = grid.vertices.Y
    vertices[:,:,:,2] = grid.vertices.Z
    vertices[:,:,:,3] = grid.vertices.lon
    vertices[:,:,:,4] = grid.vertices.lat

    centers[:,:,:,0] = grid.centers.X
    centers[:,:,:,1] = grid.centers.Y
    centers[:,:,:,2] = grid.centers.Z
    centers[:,:,:,3] = grid.centers.lon
    centers[:,:,:,4] = grid.centers.lat

    edx[:,:,:,0] = grid.edx.X
    edx[:,:,:,1] = grid.edx.Y
    edx[:,:,:,2] = grid.edx.Z
    edx[:,:,:,3] = grid.edx.lon
    edx[:,:,:,4] = grid.edx.lat

    edy[:,:,:,0] = grid.edy.X
    edy[:,:,:,1] = grid.edy.Y
    edy[:,:,:,2] = grid.edy.Z
    edy[:,:,:,3] = grid.edy.lon
    edy[:,:,:,4] = grid.edy.lat

    areas[:,:,:]           = grid.areas[:,:,:]
    length_edx[:,:,:]      = grid.length_edx[:,:,:]
    length_edy[:,:,:]      = grid.length_edy[:,:,:]

    metric_tensor_centers[:,:,:] = grid.metric_tensor_centers[:,:,:]

    prod_ex_elon_edx[:,:,:]          = grid.prod_ex_elon_edx[:,:,:]
    prod_ex_elat_edx[:,:,:]          = grid.prod_ex_elat_edx[:,:,:]
    prod_ey_elon_edx[:,:,:]          = grid.prod_ey_elon_edx[:,:,:]
    prod_ey_elat_edx[:,:,:]          = grid.prod_ey_elat_edx[:,:,:]
    determinant_ll2contra_edx[:,:,:] = grid.determinant_ll2contra_edx[:,:,:]

    prod_ex_elon_edy[:,:,:]          = grid.prod_ex_elon_edy[:,:,:]
    prod_ex_elat_edy[:,:,:]          = grid.prod_ex_elat_edy[:,:,:]
    prod_ey_elon_edy[:,:,:]          = grid.prod_ey_elon_edy[:,:,:]
    prod_ey_elat_edy[:,:,:]          = grid.prod_ey_elat_edy[:,:,:]
    determinant_ll2contra_edy[:,:,:] = grid.determinant_ll2contra_edy[:,:,:]


    Xu[:,:,:] = grid.Xu[:,:,:]
    Yv[:,:,:] = grid.Yv[:,:,:]

    griddata.close()
    print("Done.")
    print("--------------------------------------------------------\n")

####################################################################################
# Create a netcdf file for the latlon 2 cubed sphere indexes
####################################################################################
def ll2cs_netcdf(i, j, panel_list, cs_grid):
    # Open a netcdf file
    print("--------------------------------------------------------")
    filename = griddir+cs_grid.name+'_ll2cs'
    print("Creating grid netcdf file "+filename+".nc")

    data = nc.Dataset(filename+".nc", mode='w', format='NETCDF4_CLASSIC')

    # Name
    data.title = cs_grid.name+'_ll2cs'

    Nlon, Nlat = np.shape(i)

    # Create dimensions
    # Latlon coordinates
    lon = data.createDimension('lon', Nlon)
    lat = data.createDimension('lat', Nlat)

    # Create variables
    I = data.createVariable('i',  'i4', ('lon','lat'))
    J = data.createVariable('j',  'i4', ('lon','lat'))
    P = data.createVariable('panel',  'i4', ('lon','lat'))

    # Values
    I[:,:] = i[:,:]
    J[:,:] = j[:,:]
    P[:,:] = panel_list[:,:]

    data.close()
    print("Done.")
    print("--------------------------------------------------------\n")
