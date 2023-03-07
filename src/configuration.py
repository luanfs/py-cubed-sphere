####################################################################################
#
# This module contains all the routines that get the needed
# parameters from the /par directory.
#
# Luan da Fonseca Santos - January 2022
# (luan.santos@usp.br)
####################################################################################

from constants import*
import os.path

def get_parameters():
   # The standard grid file configuration.par must exist in par/ directory
   file_path = pardir+"configuration.par"

   if os.path.exists(file_path): # The file exists
      # Open the grid file
      confpar = open(file_path, "r")

      # Read the grid file lines
      confpar.readline()
      confpar.readline()
      N = confpar.readline()
      confpar.readline()
      tranformation = confpar.readline()
      confpar.readline()
      gridload = confpar.readline()
      confpar.readline()
      showonscreen = confpar.readline()
      confpar.readline()
      test_case = confpar.readline()
      confpar.readline()
      map_projection = confpar.readline()
      confpar.readline()

      # Close the file
      confpar.close()

      # Convert from str to int
      N              = int(N)
      showonscreen   = bool(int(showonscreen))  # Also converted to boolean
      gridload       = bool(int(gridload))      # Also converted to boolean
      tranformation  = int(tranformation)
      test_case      = int(test_case)
      map_projection = int(map_projection)

      # Name the transformation
      if tranformation == 1:
         transf = "gnomonic_equidistant"
      elif tranformation == 2:
         transf = "gnomonic_equiangular"
      #elif tranformation == 3:
      #   transf = "conformal"
      else:
         print("ERROR: invalid transformation")
         exit()

      # Name the map projection
      if map_projection == 1:
         map = "mercator"
      elif map_projection == 2:
         map = "sphere"
      else:
         print("ERROR: invalid map projection")
         exit()

      # Print the parameters on the screen
      print("\n--------------------------------------------------------")
      print("Parameters from file", file_path,"\n")
      print("Number of cells along a coordinate axis: ", N)
      print("Show process on the screen: ", showonscreen)
      print("Loadable grid: ", gridload)
      print("Transformation:", transf)
      print("Test case to be done: ", test_case)
      print("Map projection: ", map)
      print("--------------------------------------------------------\n")

   else:   # The file does not exist
      print("ERROR in get_grid_parameters: file mesh.par not found in /par.")
      exit()

   return N, transf, showonscreen, gridload, test_case, map


####################################################################################
# Get parameters for advection test case
####################################################################################
def get_advection_parameters():
    # The standard grid file configuration.par must exist in par/ directory
    file_path = pardir+"advection.par"

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        Tf = confpar.readline()
        confpar.readline()
        dt = confpar.readline()
        confpar.readline()
        ic = confpar.readline()
        confpar.readline()
        vf = confpar.readline()
        confpar.readline()
        tc = confpar.readline()
        confpar.readline()
        recon = confpar.readline()
        confpar.readline()
        opsplit = confpar.readline()
        confpar.readline()
        degree = confpar.readline()
        confpar.readline()

        # Close the file
        confpar.close()

        # Convert from str to int
        Tf = float(Tf)
        dt = float(dt)
        ic = int(ic)
        vf = int(vf)
        tc = int(tc)
        recon = int(recon)
        opsplit = int(opsplit)
        degree = int(degree)

        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Total time of integration: ", Tf)
        print("Time step: ", dt)
        print("Initial condition: ", ic)
        print("Vector field: ", vf)
        print("Adv test case: ", tc)
        print("Reconstruction scheme: ", recon)
        print("Splitting scheme: ", opsplit)
        print("Ghost cells interpolation degree: ", degree)
        print("--------------------------------------------------------\n")

    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file "+ filename +" not found in /par.")
        exit()
    return dt, Tf, tc, ic, vf, recon, opsplit, degree


####################################################################################
# Get parameters for interpolation test case
####################################################################################
def get_interpolation_parameters():
    # The standard grid file configuration.par must exist in par/ directory
    file_path = pardir+"interpolation.par"

    if os.path.exists(file_path): # The file exists
        # Open the grid file
        confpar = open(file_path, "r")

        # Read the grid file lines
        confpar.readline()
        confpar.readline()
        ic = confpar.readline()
        confpar.readline()
        #interpol_degree = confpar.readline()
        #confpar.readline()

        # Close the file
        confpar.close()

        # Convert from str to int
        ic = int(ic)
        #degree = int(interpol_degree)
        #Print the parameters on the screen
        print("\n--------------------------------------------------------")
        print("Parameters from file", file_path,"\n")
        print("Scalar field: ", ic)
        #print("Ghost cells interpolation degree: ", degree)
        print("--------------------------------------------------------\n")

    else:   # The file does not exist
        print("ERROR in get_grid_parameters: file "+ filename +" not found in /par.")
        exit()
    return ic#, degree
