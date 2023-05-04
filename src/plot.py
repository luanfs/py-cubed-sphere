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
    Nt = grid.ng + grid.N

    # Plot ghost cells?
    plot_gc = False

    # Plot C grid points?
    plot_cgrid = False

    # Plot tangent vectors?
    plot_tg = False

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
        lon_pu = grid.pu.lon[:,:,p]*rad2deg
        lat_pu = grid.pu.lat[:,:,p]*rad2deg

        # Edges in y direction
        lon_pv = grid.pv.lon[:,:,p]*rad2deg
        lat_pv = grid.pv.lat[:,:,p]*rad2deg

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
                        elif i==j or i+j==grid.N+grid.ng-1:
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
                        if p==1 and (i>=i0 and i<i0+grid.ngr) and (j<j0 or j>=jend):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'cyan', transform=ccrs.Geodetic())
                        if p==3 and (i>=iend-grid.ngr and i<iend) and (j<j0 or j>=jend):
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

                A_lon, A_lat = lon_pu[:,:], lat_pu[:,:]
                for i in range(i0, iend+1):
                    for j in range(j0, jend):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'blue', transform=ccrs.Geodetic())

                A_lon, A_lat = lon_pv[:,:], lat_pv[:,:]
                for i in range(i0, iend):
                    for j in range(j0, jend+1):
                            plt.plot(A_lon[i,j], A_lat[i,j], marker='o',color = 'red', transform=ccrs.Geodetic())


                plt.xlim(-50,50)
                plt.ylim(-50,50)

        elif plot_tg:
            # Edges in x direction
            lon_pu = grid.pu.lon[i0:iend+1,j0:jend,p]*rad2deg
            lat_pu = grid.pu.lat[i0:iend+1,j0:jend,p]*rad2deg

            # Edges in y direction
            lon_pv = grid.pv.lon[i0:iend,j0:jend+1,p]*rad2deg
            lat_pv = grid.pv.lat[i0:iend,j0:jend+1,p]*rad2deg

            # Tangent vector at edges in x direction
            vec_tgx_pu_lat = grid.ex_pu.lat[i0:iend+1,j0:jend,p]
            vec_tgx_pu_lon = grid.ex_pu.lon[i0:iend+1,j0:jend,p]
            vec_tgy_pu_lat = grid.ey_pu.lat[i0:iend+1,j0:jend,p]
            vec_tgy_pu_lon = grid.ey_pu.lon[i0:iend+1,j0:jend,p]

            # Tangent vector at edges in y direction
            vec_tgx_pv_lat = grid.ex_pv.lat[i0:iend,j0:jend+1,p]
            vec_tgx_pv_lon = grid.ex_pv.lon[i0:iend,j0:jend+1,p]
            vec_tgy_pv_lat = grid.ey_pv.lat[i0:iend,j0:jend+1,p]
            vec_tgy_pv_lon = grid.ey_pv.lon[i0:iend,j0:jend+1,p]

            # Plot tangent vector at edge points in x dir
            plt.quiver(lon_pu, lat_pu, vec_tgx_pu_lon, vec_tgx_pu_lat, width = 0.001)
            #plt.quiver(lon_pu, lat_pu, vec_tgy_pu_lon, vec_tgy_pu_lat, width = 0.001)
            #plt.quiver(lon_pv, lat_pv, vec_tgx_pv_lon, vec_tgx_pv_lat, width = 0.001)
            #plt.quiver(lon_pv, lat_pv, vec_tgy_pv_lon, vec_tgy_pv_lat, width = 0.001)

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

        #plt.plot([A_lon, B_lon], [A_lat, B_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())
        #plt.plot([B_lon, C_lon], [B_lat, C_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())
        #plt.plot([C_lon, D_lon], [C_lat, D_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())
        #plt.plot([D_lon, A_lon], [D_lat, A_lat],linewidth=0.2, color='black', transform=ccrs.Geodetic())

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

    plt.contourf(latlon_grid.lon*rad2deg, latlon_grid.lat*rad2deg, field, cmap=colormap,  levels = np.linspace(qmin, qmax, 101))
    ax.projection = ccrs.PlateCarree()
    #plt.show()
    #exit()
    # Plot the scalar field
    #plt.contourf(latlon_grid.lon*rad2deg, latlon_grid.lat*rad2deg, field, cmap=colormap, levels = np.linspace(qmin, qmax, 101), transform=ccrs.PlateCarree())

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
def plot_scalar_and_vector_field(field, ulon_pu, vlat_pu, ulon_pv, vlat_pv, \
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
            lon_pu = cs_grid.pu.lon[i0:iend+1:step, j0:jend:step,p]*rad2deg
            lat_pu = cs_grid.pu.lat[i0:iend+1:step, j0:jend:step,p]*rad2deg
            lon_pu, lat_pu = np.ndarray.flatten(lon_pu), np.ndarray.flatten(lat_pu)

            # Edges in y direction
            lon_pv = cs_grid.pv.lon[i0:iend:step, j0:jend+1:step,p]*rad2deg
            lat_pv = cs_grid.pv.lat[i0:iend:step, j0:jend+1:step,p]*rad2deg
            lon_pv, lat_pv = np.ndarray.flatten(lon_pv), np.ndarray.flatten(lat_pv)

            # Vector field at edges in x direction
            vec_pu_lon = ulon_pu[0:N+1:step,0:N:step,p]
            vec_pu_lat = vlat_pu[0:N+1:step,0:N:step,p]
            vec_pu_lat, vec_pu_lon = np.ndarray.flatten(vec_pu_lat), np.ndarray.flatten(vec_pu_lon)

            # Vector field at edges in y direction
            vec_pv_lon = ulon_pv[0:N:step,0:N+1:step,p]
            vec_pv_lat = vlat_pv[0:N:step,0:N+1:step,p]
            vec_pv_lat, vec_pv_lon = np.ndarray.flatten(vec_pv_lat), np.ndarray.flatten(vec_pv_lon)

            # Plot tangent vector at edge points in x dir
            plt.quiver(lon_pu, lat_pu, vec_pu_lon, vec_pu_lat, width = 0.001)
            #plt.quiver(lon_pv, lat_pv, vec_pv_lon, vec_pv_lat, width = 0.001)

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
    n = grid.N+grid.ng

    # Grid spacing
    dx  = griddata.createVariable('dx','f8')
    dy  = griddata.createVariable('dy','f8')
    dx[:] = grid.dx
    dy[:] = grid.dy

    # Some integers
    ng    = griddata.createVariable('ng'  ,'i4')
    ngl   = griddata.createVariable('ngl' ,'i4')
    ngr   = griddata.createVariable('ngr' ,'i4')
    N     = griddata.createVariable('N'   ,'i4')
    i0    = griddata.createVariable('i0'  ,'i4')
    iend  = griddata.createVariable('iend','i4')
    j0    = griddata.createVariable('j0'  ,'i4')
    jend  = griddata.createVariable('jend','i4')

    ng[:]   = grid.ng
    ngl[:]  = grid.ngl
    ngr[:]  = grid.ngr
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
    pu = griddata.createVariable('pu'     , 'f8', ('ix' , 'jy2', 'panel', 'coorddim'))
    pv = griddata.createVariable('pv'     , 'f8', ('ix2', 'jy' , 'panel', 'coorddim'))
    ex_pc = griddata.createVariable('ex_pc', 'f8', ('ix2' , 'jy2', 'panel', 'coorddim'))
    ey_pc = griddata.createVariable('ey_pc', 'f8', ('ix2' , 'jy2', 'panel', 'coorddim'))
    ex_pu = griddata.createVariable('ex_pu', 'f8', ('ix' , 'jy2', 'panel', 'coorddim'))
    ey_pu = griddata.createVariable('ey_pu', 'f8', ('ix' , 'jy2', 'panel', 'coorddim'))
    ex_pv = griddata.createVariable('ex_pv', 'f8', ('ix2', 'jy' , 'panel', 'coorddim'))
    ey_pv = griddata.createVariable('ey_pv', 'f8', ('ix2', 'jy' , 'panel', 'coorddim'))

    # Geometric properties
    areas                   = griddata.createVariable('areas'                , 'f8', ('ix2', 'jy2', 'panel'))
    length_pu               = griddata.createVariable('length_pu'           , 'f8', ('ix2', 'jy2', 'panel'))
    length_pv               = griddata.createVariable('length_pv'           , 'f8', ('ix2', 'jy2', 'panel'))

    prod_ex_elon_pc          = griddata.createVariable('prod_ex_elon_pc'    , 'f8', ('ix2', 'jy2' , 'panel'))
    prod_ex_elat_pc          = griddata.createVariable('prod_ex_elat_pc'    , 'f8', ('ix2', 'jy2' , 'panel'))
    prod_ey_elon_pc          = griddata.createVariable('prod_ey_elon_pc'    , 'f8', ('ix2', 'jy2' , 'panel'))
    prod_ey_elat_pc          = griddata.createVariable('prod_ey_elat_pc'    , 'f8', ('ix2', 'jy2' , 'panel'))
    determinant_ll2contra_pc = griddata.createVariable('determinant_ll2contra_pc', 'f8', ('ix2', 'jy2' , 'panel'))
    metric_tensor_centers   = griddata.createVariable('metric_tensor_centers', 'f8', ('ix2', 'jy2', 'panel'))

    prod_ex_elon_pu          = griddata.createVariable('prod_ex_elon_pu'    , 'f8', ('ix', 'jy2' , 'panel'))
    prod_ex_elat_pu          = griddata.createVariable('prod_ex_elat_pu'    , 'f8', ('ix', 'jy2' , 'panel'))
    prod_ey_elon_pu          = griddata.createVariable('prod_ey_elon_pu'    , 'f8', ('ix', 'jy2' , 'panel'))
    prod_ey_elat_pu          = griddata.createVariable('prod_ey_elat_pu'    , 'f8', ('ix', 'jy2' , 'panel'))
    determinant_ll2contra_pu = griddata.createVariable('determinant_ll2contra_pu', 'f8', ('ix', 'jy2' , 'panel'))
    metric_tensor_pu = griddata.createVariable('metric_tensor_pu', 'f8', ('ix', 'jy2' , 'panel'))

    prod_ex_elon_pv          = griddata.createVariable('prod_ex_elon_pv'    , 'f8', ('ix2', 'jy' , 'panel'))
    prod_ex_elat_pv          = griddata.createVariable('prod_ex_elat_pv'    , 'f8', ('ix2', 'jy' , 'panel'))
    prod_ey_elon_pv          = griddata.createVariable('prod_ey_elon_pv'    , 'f8', ('ix2', 'jy' , 'panel'))
    prod_ey_elat_pv          = griddata.createVariable('prod_ey_elat_pv'    , 'f8', ('ix2', 'jy' , 'panel'))
    determinant_ll2contra_pv = griddata.createVariable('determinant_ll2contra_pv', 'f8', ('ix2', 'jy' , 'panel'))
    metric_tensor_pv = griddata.createVariable('metric_tensor_pv', 'f8', ('ix2', 'jy' , 'panel'))

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

    pu[:,:,:,0] = grid.pu.X
    pu[:,:,:,1] = grid.pu.Y
    pu[:,:,:,2] = grid.pu.Z
    pu[:,:,:,3] = grid.pu.lon
    pu[:,:,:,4] = grid.pu.lat

    pv[:,:,:,0] = grid.pv.X
    pv[:,:,:,1] = grid.pv.Y
    pv[:,:,:,2] = grid.pv.Z
    pv[:,:,:,3] = grid.pv.lon
    pv[:,:,:,4] = grid.pv.lat

    ex_pc[:,:,:,0] = grid.ex_pc.X
    ex_pc[:,:,:,1] = grid.ex_pc.Y
    ex_pc[:,:,:,2] = grid.ex_pc.Z
    ex_pc[:,:,:,3] = grid.ex_pc.lon
    ex_pc[:,:,:,4] = grid.ex_pc.lat

    ey_pc[:,:,:,0] = grid.ey_pc.X
    ey_pc[:,:,:,1] = grid.ey_pc.Y
    ey_pc[:,:,:,2] = grid.ey_pc.Z
    ey_pc[:,:,:,3] = grid.ey_pc.lon
    ey_pc[:,:,:,4] = grid.ey_pc.lat

    ex_pu[:,:,:,0] = grid.ex_pu.X
    ex_pu[:,:,:,1] = grid.ex_pu.Y
    ex_pu[:,:,:,2] = grid.ex_pu.Z
    ex_pu[:,:,:,3] = grid.ex_pu.lon
    ex_pu[:,:,:,4] = grid.ex_pu.lat

    ey_pu[:,:,:,0] = grid.ey_pu.X
    ey_pu[:,:,:,1] = grid.ey_pu.Y
    ey_pu[:,:,:,2] = grid.ey_pu.Z
    ey_pu[:,:,:,3] = grid.ey_pu.lon
    ey_pu[:,:,:,4] = grid.ey_pu.lat

    ex_pv[:,:,:,0] = grid.ex_pv.X
    ex_pv[:,:,:,1] = grid.ex_pv.Y
    ex_pv[:,:,:,2] = grid.ex_pv.Z
    ex_pv[:,:,:,3] = grid.ex_pv.lon
    ex_pv[:,:,:,4] = grid.ex_pv.lat

    ey_pv[:,:,:,0] = grid.ey_pv.X
    ey_pv[:,:,:,1] = grid.ey_pv.Y
    ey_pv[:,:,:,2] = grid.ey_pv.Z
    ey_pv[:,:,:,3] = grid.ey_pv.lon
    ey_pv[:,:,:,4] = grid.ey_pv.lat

    areas[:,:,:]     = grid.areas[:,:,:]
    length_pu[:,:,:] = grid.length_pu[:,:,:]
    length_pv[:,:,:] = grid.length_pv[:,:,:]

    metric_tensor_centers[:,:,:] = grid.metric_tensor_centers[:,:,:]
    metric_tensor_pu[:,:,:] = grid.metric_tensor_pu[:,:,:]
    metric_tensor_pv[:,:,:] = grid.metric_tensor_pv[:,:,:]

    prod_ex_elon_pc[:,:,:] = grid.prod_ex_elon_pc[:,:,:]
    prod_ex_elat_pc[:,:,:] = grid.prod_ex_elat_pc[:,:,:]
    prod_ey_elon_pc[:,:,:] = grid.prod_ey_elon_pc[:,:,:]
    prod_ey_elat_pc[:,:,:] = grid.prod_ey_elat_pc[:,:,:]
    determinant_ll2contra_pc[:,:,:] = grid.determinant_ll2contra_pc[:,:,:]

    prod_ex_elon_pu[:,:,:] = grid.prod_ex_elon_pu[:,:,:]
    prod_ex_elat_pu[:,:,:] = grid.prod_ex_elat_pu[:,:,:]
    prod_ey_elon_pu[:,:,:] = grid.prod_ey_elon_pu[:,:,:]
    prod_ey_elat_pu[:,:,:] = grid.prod_ey_elat_pu[:,:,:]
    determinant_ll2contra_pu[:,:,:] = grid.determinant_ll2contra_pu[:,:,:]

    prod_ex_elon_pv[:,:,:] = grid.prod_ex_elon_pv[:,:,:]
    prod_ex_elat_pv[:,:,:] = grid.prod_ex_elat_pv[:,:,:]
    prod_ey_elon_pv[:,:,:] = grid.prod_ey_elon_pv[:,:,:]
    prod_ey_elat_pv[:,:,:] = grid.prod_ey_elat_pv[:,:,:]
    determinant_ll2contra_pv[:,:,:] = grid.determinant_ll2contra_pv[:,:,:]

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
