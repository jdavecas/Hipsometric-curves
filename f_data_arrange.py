import os
import geopandas as gpd
import re
import pandas as pd
import glob
import json
from shapely.geometry import Point
import tkinter as tk
from tkinter import filedialog


path_to_shp = os.path.dirname(os.path.abspath(__file__))
one_back = os.path.abspath(os.path.join(path_to_shp, '..'))

def get_directory_path(title="Select Directory"):
    """Opens a file dialog to select a directory and returns its path."""
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(title=title)
    return directory_path


def replace_missing(data):
    """
    Replaces values starting with '-99999' with None in a dictionary.

    Args:
        data: The dictionary containing data.

    Returns:
        A new dictionary with the replaced values.
    """
    pattern = r'^-99999\d*'  # Match the pattern for missing values

    updated_data = {}
    for key, nested_dict in data.items():
        updated_nested_dict = {}
        for var_name, values in nested_dict.items():
            updated_nested_dict[var_name] = [None if re.match(pattern, str(value)) else value for value in values]
        updated_data[key] = updated_nested_dict
    return updated_data

def process_shape(Basin, reach_node,r_n_id):
    """
    Process the shapefile data whether node or reach and return a dictionary with the data.

    Args:
        Basin: The name of the basin. (Atrato, Magdalena, Ohio, Po, or Tanana)
        reach_node: Indicates if the shapefiles to process are reaches or nodes. (Reach or Node)
        r_n_id: the first column of the shapefile that contains the reach or node id. (reach_id or node_id)

    Returns:
        A dictionary with the processed data
    """
    
    results = {}
    # Get 'Basins' directory interactively
    to_basins = get_directory_path(title=f"Select 'Basins' directory for {Basin}") 
    if not to_basins:  # Handle cancellation
        return {} 

    shapefiles = glob.glob(os.path.join(to_basins, '*.shp'))

    if reach_node == 'Reach':
        columns_to_take = ['reach_id','time_str', 'p_lat', 'p_lon', 'river_name','wse','wse_u','width','width_u','node_dist','xtrk_dist','reach_q','reach_q_b',
                    'dark_frac','ice_clim_f','ice_dyn_f','xovr_cal_q','p_width','p_wid_var','p_dist_out', 'p_length'] # Define the columns to take from the shapefile
    else:
        columns_to_take = ['node_id','time_str', 'lat', 'lon', 'river_name','wse','wse_u','width','width_u','node_dist','xtrk_dist','node_q','node_q_b',
                    'dark_frac','ice_clim_f','ice_dyn_f','xovr_cal_q','p_width','p_wid_var', 'p_dist_out'] # Define the columns to take from the shapefile
    
    for shp in shapefiles:
        gdf = gpd.read_file(shp)

        for _, row in gdf.iterrows(): 
            id = row[r_n_id] # Get the reach or node id (e.g. reach_id or node_id)
            if id not in results:
                results[id] = {}

            for col in columns_to_take:
                if col == r_n_id or col not in gdf.columns:
                    continue

                if col not in results[id]:
                    results[id][col] = []

                value = row[col]
                
                if pd.notnull(value):
                    results[id][col].append(value)

    # Apply the replace_missing function to clean the data
    results = replace_missing(results)

    return results

def export_to_json(results_dict):
    """
    Export the results dictionary to a JSON file with interactive file selection.
    """

    # Get output file path interactively
    output_file = filedialog.asksaveasfilename(defaultextension=".json", 
                                              filetypes=[("JSON files", "*.json")],
                                              title="Save results as JSON")
    if not output_file:  # Handle cancellation
        return

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)


def dict_to_shapefile(data_dict, reach_node):
    """
    Convert a dictionary of data to a shapefile, 
    and save it to the specified output path.

    Args:
        data_dict (dict): The dictionary containing the data of the selected basin or river

    Raises:
        ValueError: If lat-lon list lengths do not match
    """
    # Hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to choose where to save the shapefile
    output_file = filedialog.asksaveasfilename(defaultextension=".shp", 
                                               filetypes=[("Shapefile", "*.shp")],
                                               title="Save shapefile")
    if not output_file:  # Handle cancellation
        print("File saving canceled.")
        return

    # List to store all the data for each point
    all_data = []

    # Iterate over the dictionary and process each node
    for outer_key, values in data_dict.items():

        if reach_node == 'Reach':
            lat_list = values.get('p_lat', [])
            lon_list = values.get('p_lon', [])
        else:
            lat_list = values.get('lat', [])
            lon_list = values.get('lon', [])

        # Ensure lat_list and lon_list are lists even if they're scalar values
        if not isinstance(lat_list, list):
            lat_list = [lat_list]
        if not isinstance(lon_list, list):
            lon_list = [lon_list]

        # Make sure lat_list and lon_list have the same length
        if len(lat_list) != len(lon_list):
            raise ValueError(f"Mismatch in length of latitude and longitude lists for {outer_key}")

        # Iterate over the lat-lon pairs and create Point objects
        for i, (lat, lon) in enumerate(zip(lat_list, lon_list)):
            point_data = {}

            # Add 'reach_id' from the outer_key as a new field in the point_data dictionary

            if reach_node == 'Reach':
                point_data['reach_id'] = outer_key
            else:
                point_data['node_id'] = outer_key

            # Copy only scalar values or take the ith element from each list
            for key, value in values.items():
                if isinstance(value, list):
                    # Take the ith element from the list if it's a list
                    point_data[key] = value[i] if i < len(value) else None
                else:
                    # If it's scalar, just copy it
                    point_data[key] = value
            
            # Add latitude and longitude and create Point geometry
            if reach_node == 'Reach':
                point_data['p_lat'] = lat
                point_data['p_lon'] = lon
            else:
                point_data['lat'] = lat
                point_data['lon'] = lon
                
            # Create the geometry field    
            point_data['geometry'] = Point(lon, lat)

            # Append the point data to the list
            all_data.append(point_data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_data)

    # Convert the DataFrame into a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set the coordinate reference system (CRS), for example, WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)

    # Export the GeoDataFrame to the shapefile path chosen by the user
    gdf.to_file(output_file, driver='ESRI Shapefile')

    print(f"Shapefile successfully saved to {output_file}")