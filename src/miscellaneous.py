####################################################################################
# 
# Module for miscellaneous routines
#
# Luan da Fonseca Santos - February 2022
# (luan.santos@usp.br)
####################################################################################


import os
from constants import datadir, graphdir

####################################################################################
# Create a folder
# Reference: https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
####################################################################################
def createFolder(dir):
   try:
      if not os.path.exists(dir):
         os.makedirs(dir)
   except OSError:
      print ('Error: Creating directory. '+ dir)

####################################################################################
# Create the needed directories
####################################################################################       
def createDirs():
   print("--------------------------------------------------------")
   print("Cubed sphere grid generator code by Luan Santos - 2022\n")
   # Check directory data does not exist   
   if not os.path.exists(datadir):
      print('Creating directory ',datadir)
      createFolder(datadir) 

   # Check directory data does not exist
   if not os.path.exists(graphdir):
      print('Creating directory ',graphdir)
      createFolder(graphdir)
   print("--------------------------------------------------------")      
