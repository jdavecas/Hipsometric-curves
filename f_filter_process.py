import json
import geopandas as gdp
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import csv
import sys

def get_file(title = "Select a file"):
    """
    Opens as window dialog to select a file and 
    returns the path to the selected file
    """
    root = tk.Tk()
    root.withdraw()
    filetypes = (
        ('JSON files', '*.json'),
        ('CSV files', '*.csv'),
        ('Parquet files', '*.parquet'),
        ('SHAPE files', '*.shp'),
        ('GPKG files', '*.gpkg'),
        ('All files', '*.*')
    )
    file_path = filedialog.askopenfilename(title = title, filetypes=filetypes)
    return file_path

def call_file():
    """
    Opens a window dialog to select a file and
    returns the path to the selected file

    """

    to_basin_path = get_file(title = "Select the file containing the file")
    if not to_basin_path:
        return {}
    
    if to_basin_path.endswith('.json'):
        with open(to_basin_path, 'r') as f:
            river_dict = json.load(f)
    elif to_basin_path.endswith('.csv'):
        with open(to_basin_path, 'r') as f:
            river_dict = pd.read_csv(f)
    elif to_basin_path.endswith('.shp'):
        river_dict = gdp.read_file(to_basin_path)
    elif to_basin_path.endswith('.parquet'):
        river_dict = pd.read_parquet(to_basin_path)
    elif to_basin_path.endswith('.gpkg'):
        river_dict = gdp.read_file(to_basin_path)
    else:
        return {}
    
    return river_dict


def valid_pairs(basin_data, reach_node,q_b_value,dark_value):

    """
    Removes any key-value pairs in the basin dictionary selected where the 'width' < 99 or None,  or 'wse' 
    values are None or negative, or 'node_reach_q_b' >= 16, and removes any keys where the inner dictionary is empty
    or has lists with fewer than 3 items.
    """
    results_list = []  # Initialize list to store results
    total_keys = len(basin_data.keys())
    deleted_count = 0
    large_lists = {}
    
    for outer_key in list(basin_data.keys()):
        inner_dict = basin_data[outer_key]
        width_list = inner_dict.get('width', [])
        wse_list = inner_dict.get('wse', [])
        dark_list = inner_dict.get('dark_frac',[])
        node_list = inner_dict.get('node_dist',[])

        if reach_node == 'Reach':
            bit_list = inner_dict.get('reach_q_b', [])
        else:
            bit_list = inner_dict.get('node_q_b', [])

        # Identify indices where 'width' or 'wse' are None or negative or 'node_reach_q_b' >= 16/8 (or 'reach_q_b' >= 16/8)
        indices_to_remove = [i for i, (w, ws, bl, dfr, nl) in enumerate(zip(width_list, wse_list, bit_list,dark_list,node_list)) if w is None or ws is None or w < 99 or ws < 0 
                         or bl >= q_b_value or dfr > dark_value or nl is None] 

        # Remove items in reverse order to avoid indexing issues
        for key in list(inner_dict.keys()):
            if isinstance(inner_dict[key], list):
                for i in sorted(indices_to_remove, reverse=True):
                    if i < len(inner_dict[key]):
                        inner_dict[key].pop(i)

    # If any list has fewer than 3 items, mark the outer dictionary for deletion
            if isinstance(inner_dict[key], list) and len(inner_dict[key]) < 3:
                del inner_dict[key]

    # Check if the inner dictionary is empty after removals or has lists with fewer than 3 items
        if not inner_dict or all(isinstance(value, list) and len(value) < 3 for value in inner_dict.values()):
            del basin_data[outer_key]
            deleted_count += 1
        else:
            # Identify and store lists with more than 20 items
            for key, value in inner_dict.items():
                if isinstance(value, list) and len(value) > 20:
                    large_lists[key] = value

    # Calculate remaining and deleted percentages
    remaining_count = total_keys - deleted_count
    remaining_percent = (remaining_count / total_keys) * 100
    deleted_percent = (deleted_count / total_keys) * 100

    # Add the result for this dataset to the list
    results_list.append([remaining_count, deleted_count, remaining_percent, deleted_percent])
    summary = pd.DataFrame(results_list, columns=['Remaining', 'Deleted', 'Remaining %', 'Deleted %'])

    # Print results
    print(f"{deleted_count} out of {total_keys} outer keys were deleted which is {deleted_count/total_keys*100:.2f}%")
    
    return basin_data, summary

def dynamic_threshold(cv, base_threshold):
    """
    Adjusts the threshold based on the coefficient of variation (CV).
    For high variability (CV > 1), the threshold is reduced moderately, but not too strictly.
    For low variability (CV < 1), the threshold is increased.
    """
    if cv >= 1:  # High variability -> stricter threshold
        return base_threshold * (1 - min(cv*0.6, 0.8))  # Cap reduction to avoid negative threshold
    else:  # Low variability -> more lenient threshold
        return base_threshold * (1 + cv)

def distance_based_filtering_auto_threshold(basin_data, reach_node, base_threshold=3):
    """
    Applies distance-based filtering using normalization for each node or reach in the dataset, adjusting 
    the threshold based on width variability (automated threshold based on coefficient of variation).
    
    Parameters:
    - basin_data: Dictionary containing river data for nodes or reaches.
    - reach_node: String to identify whether it is 'Reach' or 'Node'.
    - base_threshold: Base threshold for filtering measurements based on distance to the centroid.
    
    Returns:
    - filtered_basin_data: Filtered dictionary after applying distance-based normalization and filtering.
    - summary: DataFrame with remaining and deleted key counts and percentages.
    """
    results_list = []  # Initialize list to store results
    total_keys = len(basin_data.keys())
    deleted_count = 0
    total_removed_items = 0  # Initialize total removed items counter

    for outer_key in list(basin_data.keys()):
        inner_dict = basin_data[outer_key]
        width_list = inner_dict.get('width', [])
        dist_list = inner_dict.get('node_dist', [])  # Distance to centroid for nodes
        if reach_node == 'Reach':
            dist_list = inner_dict.get('reach_dist', [])  # Distance to centroid for reaches

        # Step 1: Calculate the median width and coefficient of variation (CV) for this node/reach
        if len(width_list) > 0:
            median_width = pd.Series(width_list).median()
            width_std = pd.Series(width_list).std()  # Width variability (standard deviation)
            coefficient_of_variation = width_std / median_width if median_width > 0 else 0
        else:
            median_width = None
            coefficient_of_variation = 0

        # Store median width and CV in the inner dictionary
        inner_dict['median_width'] = median_width
        inner_dict['cv'] = coefficient_of_variation

        # Step 2: Calculate dynamic threshold based on coefficient of variation
        dynamic_threshold_value = dynamic_threshold(coefficient_of_variation, base_threshold)

        # Store dynamic threshold in the inner dictionary
        inner_dict['dynamic_threshold'] = dynamic_threshold_value

        # Step 3: Normalize the node_dist or reach_dist by the median width
        if median_width and median_width > 0:
            normalized_dist_list = [dist / median_width if dist is not None else float('inf') for dist in dist_list]
        else:
            normalized_dist_list = [float('inf')] * len(dist_list)  # Mark as 'inf' if no valid width for filtering

        # Store normalized distances in the inner dictionary
        inner_dict['normalized_dist'] = normalized_dist_list

        # Step 4: Identify indices to remove based on the dynamic threshold
        indices_to_remove = [i for i, norm_dist in enumerate(normalized_dist_list) if norm_dist > dynamic_threshold_value]

        # Count the number of items being removed
        total_removed_items += len(indices_to_remove)

        # Step 5: Remove items in reverse order to avoid indexing issues
        for key in list(inner_dict.keys()):
            if isinstance(inner_dict[key], list):
                for i in sorted(indices_to_remove, reverse=True):
                    if i < len(inner_dict[key]):
                        inner_dict[key].pop(i)

        # Step 6: If any list has fewer than 3 items, mark the outer dictionary for deletion
        if all(isinstance(value, list) and len(value) < 3 for value in inner_dict.values()):
            del basin_data[outer_key]
            deleted_count += 1

    # Step 7: Calculate remaining and deleted percentages
    remaining_count = total_keys - deleted_count
    remaining_percent = (remaining_count / total_keys) * 100
    deleted_percent = (deleted_count / total_keys) * 100

    # Add the result for this dataset to the list
    results_list.append([remaining_count, deleted_count, remaining_percent, deleted_percent, total_removed_items])
    summary = pd.DataFrame(results_list, columns=['Remaining', 'Deleted', 'Remaining %', 'Deleted %', 'Total Removed Items'])

    # Print results
    print(f"{deleted_count} out of {total_keys} outer keys were deleted, which is {deleted_percent:.2f}%.")
    print(f"Total removed items across all nodes: {total_removed_items}")

    return basin_data, summary


def outliers(basin_data):
    """
    Removes outliers from the basin dictionary. Outliers are values more than 2 standard deviations 
    from the mean in 'width' and 'wse' lists. If an outlier is removed, all values in the same position
    in other lists are also removed. This is applied only to lists with more than 20 items.
    """
    total_removed = 0  # Initialize counter for removed outliers
    total_keys = len(basin_data.keys())  # Get the total number of keys before filtering
    deleted_count = 0  # Initialize deleted keys count

    for outer_key in list(basin_data.keys()):
        inner_dict = basin_data[outer_key]

        # Process only lists with more than 20 items
        for key in list(inner_dict.keys()):
            if isinstance(inner_dict[key], list) and len(inner_dict[key]) > 20:
                # Check if 'width' or 'wse' keys are in the dictionary
                if key in ['width', 'wse']:
                    data_list = inner_dict[key]
                    
                    # Calculate the mean and standard deviation
                    mean = np.mean(data_list)
                    std_dev = np.std(data_list)

                    # Identify outliers
                    outliers_indices = [i for i, value in enumerate(data_list) if abs(value - mean) > 2 * std_dev]

                    # Remove outliers in reverse order to avoid indexing issues
                    num_outliers = len(outliers_indices)
                    total_removed += num_outliers

                    # Sort outlier indices in reverse order
                    outliers_indices = sorted(outliers_indices, reverse=True)

                    # Remove outliers and corresponding values in other lists
                    for i in outliers_indices:
                        for k in inner_dict.keys():
                            if isinstance(inner_dict[k], list) and i < len(inner_dict[k]):
                                inner_dict[k].pop(i)

        # After outlier removal, check if there are fewer than 3 items in any list
        if all(isinstance(value, list) and len(value) < 3 for value in inner_dict.values()):
            del basin_data[outer_key]
            deleted_count += 1

    # Calculate the remaining and deleted counts
    remaining_count = total_keys - deleted_count
    remaining_percent = (remaining_count / total_keys) * 100
    deleted_percent = (deleted_count / total_keys) * 100

    # Create a summary DataFrame
    summary = pd.DataFrame([[remaining_count, deleted_count, remaining_percent, deleted_percent]], 
                           columns=['Remaining', 'Deleted', 'Remaining %', 'Deleted %'])

    # Print results
    print(f"{deleted_count} out of {total_keys} outer keys were deleted.")

    return basin_data, summary





def data(basin, ind):
    """
    Creates a DataFrame with the remaining and deleted keys from the 'basin' dictionary and their percentages

    Args:
        basin (dict): The filtered basin dictionary
        ind (str): The index names for the DataFrame

    Returns:
        DataFrame: A DataFrame with the remaining and deleted keys and their percentages
    """
    # Define the index and columns
    index = [ind]
    columns = ['Remaining', 'Deleted', 'Remaining %', 'Deleted %']
    
    # Create an empty DataFrame with the specified index and columns
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill the DataFrame with values from the 'basin' DataFrame
    
    df.loc[ind, 'Remaining'] = basin.loc[0, 'Remaining']
    df.loc[ind, 'Deleted'] = basin.loc[0, 'Deleted']
    df.loc[ind, 'Remaining %'] = basin.loc[0, 'Remaining %']
    df.loc[ind, 'Deleted %'] = basin.loc[0, 'Deleted %']
    return df

def export_dataframe(df, is_geodataframe=False):
    """
    Opens a dialog window to ask the user for a file path and saves the DataFrame (or GeoDataFrame) 
    as either a CSV, Shapefile, GeoPackage, or Parquet file.

    Args:
        df (DataFrame or GeoDataFrame): The DataFrame or GeoDataFrame to be saved.
        is_geodataframe (bool): If True, saves as a geospatial file (Shapefile, GeoPackage, or Parquet).
    """
    # Create a Tkinter root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Determine the file types based on whether it's a GeoDataFrame or not
    if is_geodataframe:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".gpkg",
            filetypes=[("GeoPackage", "*.gpkg"), ("Parquet", "*.parquet"), ("Shapefiles", "*.shp"), ("All files", "*.*")],
            title="Save the GeoDataFrame"
        )
    else:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save the DataFrame"
        )
    
    # If the user selected a file, save the DataFrame or GeoDataFrame to the chosen file
    if file_path:
        if is_geodataframe:
            if file_path.endswith(".shp"):
                df.to_file(file_path, driver='ESRI Shapefile')
                print(f"GeoDataFrame has been exported as a Shapefile to {file_path}")
            elif file_path.endswith(".gpkg"):
                df.to_file(file_path, driver="GPKG")
                print(f"GeoDataFrame has been exported as a GeoPackage to {file_path}")
            elif file_path.endswith(".parquet"):
                df.to_parquet(file_path)
                print(f"GeoDataFrame has been exported as Parquet to {file_path}")
        else:
            df.to_csv(file_path, index=True)
            print(f"DataFrame has been exported to {file_path}")
    else:
        print("Export canceled.")
