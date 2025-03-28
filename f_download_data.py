import sys
import requests
import geopandas as gpd
import earthaccess
import os
from pathlib import Path
import zipfile
from datetime import datetime
from IPython.display import clear_output
import time 

# function to select the passes, download the data and extract the files

path_to_shp = os.path.dirname(os.path.abspath(__file__))
one_back = os.path.abspath(os.path.join(path_to_shp, '..'))

def passes_to_download(Basin):

    '''this function allows the user either to select 
    the passes to download or to download all the passes in the selected basin'''

    to_basins = os.path.join(one_back, '0_data','External', 'Basins', Basin, 'shapes', 'Passes', 'Pass_codes.shp')
    Basin_shp = os.path.join(one_back, to_basins)
    Passes = []
    input_passes = input("Enter the pass(es) you want to download separated by commas (press Enter to finish): ")
    print(input_passes)
    if input_passes:
        Passes.extend(input_passes.split(','))
        Passes = [i.strip() for i in Passes]
    else:
        shp_to_gdf = gpd.read_file(Basin_shp)
        column = "pass"
        Passes = shp_to_gdf[column].tolist()
        Passes = list(set(Passes))
    return Passes

current_datetime = datetime.now() #Get the current date and time

def download_data(Basin, passes, continent, start_date, reach_node):

    '''this function downloads the data for the selected passes'''
    relapath = os.path.join(one_back, '0_data','External', 'Basins', Basin, 'shapes', reach_node)
    savepath = os.path.join(one_back, relapath)

    # Convert start_date to datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.now()

    temporal_range = (
        start_dt.strftime('%Y-%m-%dT%H:%M:%S'),
        end_dt.strftime('%Y-%m-%dT%H:%M:%S')
)


    attempts = 0

    while attempts < 3:
        try:
            # Attempt to download the data
            files_downloaded = 0
            for codes in passes:
                granule_name_patter = f'*{reach_node}*_*_{codes}_{continent}*' # Define the pattern to search for the granule
                results2 = earthaccess.search_data(
                    short_name='SWOT_L2_HR_RIVERSP_2.0',
                    temporal=temporal_range,
                    granule_name=granule_name_patter
                )
                if not results2:
                    continue

                for url in results2:
                    earthaccess.download(url, savepath) # Download the data to the specific path
                    clear_output(wait=True) # Clear the previous progress bar before displaying the new one
                    print("File downloaded successfully")
            break
        except Exception as e:
            print(f"Download attempt {attempts + 1} failed", e)
            attempts += 1
            time.sleep(1) # Wait for 1 second before trying again

    for item in os.listdir(savepath): # loop through items in dir
        if item.endswith(".zip"): # check for ".zip" extension
            zip_ref = zipfile.ZipFile(f"{savepath}/{item}") # create zipfile object
            zip_ref.extractall(savepath) # extract file to dir
            zip_ref.close() # close file

    if attempts >= 3:
            return print("Maximum number of download attempts reached. Please try again later")
    else:
        return print("All files downloaded successfully")

def unzip_files_in_directory(savepath):
    '''This function unzips all zip files in the given directory.'''
    
    for item in os.listdir(savepath):  # loop through items in dir
        if item.endswith(".zip"):  # check for ".zip" extension
            file_path = os.path.join(savepath, item)
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:  # create zipfile object
                    zip_ref.extractall(savepath)  # extract file to dir
                print(f"{item} unzipped successfully")
            except zipfile.BadZipFile:
                print(f"Error: {item} is not a valid zip file or is corrupted")
            except Exception as e:
                print(f"An unexpected error occurred while unzipping {item}: {e}")
