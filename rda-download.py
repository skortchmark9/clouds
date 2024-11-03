#!/usr/bin/env python
""" 
Python script to download selected files from rda.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os
from urllib.request import build_opener

opener = build_opener()

filelist = [
   # Cloud water mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QCLOUD_200010.nc',
   # Ice mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QICE_200010.nc',
  # Rain water mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QRAIN_200010.nc'
]

for file in filelist:
    ofile = 'data/' + os.path.basename(file)
    sys.stdout.write("downloading " + ofile + " ... ")
    sys.stdout.flush()
    infile = opener.open(file)
    outfile = open(ofile, "wb")
    outfile.write(infile.read())
    outfile.close()
    sys.stdout.write("done\n")
