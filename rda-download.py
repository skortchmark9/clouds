#!/usr/bin/env python
""" 
Python script to download selected files from rda.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

filelist = [
   # Ice mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QICE_200010.nc',
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QICE_200011.nc',

   # Cloud water mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QCLOUD_200010.nc',
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QCLOUD_200011.nc',

  # Graupel mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QGRAUP_200010.nc',
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QGRAUP_200011.nc',

  # Rain water mixing ratio (kg kg-1)
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QRAIN_200010.nc',
  'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QRAIN_200011.nc',

] + [
  # Heights (GPH) are stored in separate files for each day
  f'https://data.rda.ucar.edu/d612000/CTRL3D/2000/wrf3d_d01_CTRL_Z_200010{str(i).zfill(2)}.nc'
  for i in range(1, 32)
] + [
  # Snow is stored in separate files for each day
  f'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QSNOW_200010{str(i).zfill(2)}.nc'
  for i in range(1, 32)
] + [
  # Heights (GPH) are stored in separate files for each day
  f'https://data.rda.ucar.edu/d612000/CTRL3D/2000/wrf3d_d01_CTRL_Z_200011{str(i).zfill(2)}.nc'
  for i in range(1, 32)
] + [
  # Snow is stored in separate files for each day
  f'https://data.rda.ucar.edu/d612000/PGW3D/2000/wrf3d_d01_PGW_QSNOW_200011{str(i).zfill(2)}.nc'
  for i in range(1, 32)
]


def download_file(url, folder='data'):
    """Download a single file and save it to the specified folder."""
    filename = os.path.join(folder, os.path.basename(url))
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping.")
        return
    
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, "wb") as outfile:
            for chunk in response.iter_content(chunk_size=8192):
                outfile.write(chunk)
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        try:
            os.remove(filename)
        except OSError:
            pass

def batch_download_parallel(filelist, batch_size=5, delay=0.1):
    """Download files in parallel with controlled batch size and delay."""
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(download_file, url) for url in filelist]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % batch_size == 0:
                print(f"Batch {i // batch_size} complete. Waiting {delay} second before next batch.")
                time.sleep(delay)


# Create directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download files in parallel with controlled batch size
batch_size = 5  # Set your preferred batch size for parallel downloads
delay = 2       # Set delay between batches in seconds
batch_download_parallel(filelist, batch_size=batch_size, delay=delay)
