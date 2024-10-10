import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
import numpy as np
from shapely.ops import nearest_points

def merge_shapefiles():
    """Imports and merges all the shapefiles in a file

    Returns:
        A new geodaframe (shapefile) containing all the shapefiles merged
    """
    # Initialize tkinter
    root = Tk()
    root.withdraw()  # Close the root window

    # Open a dialog window to select the folder with shapefiles
    directory = askdirectory(title="Select Folder with Shapefiles")
    if not directory:
        print("No folder selected. Exiting...")
        return

    # List all shapefiles in the selected directory
    shapefiles = [f for f in os.listdir(directory) if f.endswith('.shp')]

    # Check if there are any shapefiles
    if not shapefiles:
        print("No shapefiles found in the selected folder.")
        return

    # Read and concatenate all shapefiles
    gdf_list = [gpd.read_file(os.path.join(directory, shapefile)) for shapefile in shapefiles]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    # Return the merged GeoDataFrame
    return merged_gdf

def process_geopackage(gpkg):
    """
    Processes a GeoDataFrame by grouping the data by crossSxnID,
    calculating the mean, max, min, and 2-sigma based on the width column,
    and creating a new GeoDataFrame with the same columns (excluding width)
    plus the calculated columns.

    Parameters:
    - gdf: GeoDataFrame, the input GeoDataFrame.

    Returns:
    - gdf_final: GeoDataFrame with calculated statistics.
    """

    # Function to calculate 2-sigma
    def calculate_2sigma(series):
        return np.std(series) * 2

    # Grouping by crossSxnID and calculating the statistics for the width column
    grouped = gpkg.groupby('crossSxnID').agg(
        median_width=('width', 'median'),
        max_width=('width', 'max'),
        min_width=('width', 'min'),
        sigma_2=('width', calculate_2sigma)
    ).reset_index()

    # Remove duplicates, keeping only the first occurrence of each crossSxnID
    gdf_clean = gpkg.drop_duplicates(subset='crossSxnID').drop(columns=['width'])

    # Merging the grouped statistics with the cleaned GeoDataFrame
    gdf_final = gdf_clean.merge(grouped, on='crossSxnID')

    return gdf_final

def merge_nearest_neighbors(rio_GLS, rio_shp_SWOT, threshold=250):
    # Ensure both GeoDataFrames are using the same CRS and project to meters
    rio_GLS = rio_GLS.to_crs(epsg=3395)  # Project to EPSG:3395 (World Mercator) for meters
    rio_shp_SWOT = rio_shp_SWOT.to_crs(epsg=3395)
    
    # Function to find the nearest rio_GLS point to each rio_shp_SWOT point
    def nearest(row, other_gdf, threshold):
        # Find nearest point in rio_GLS within the threshold distance
        nearest_geom = nearest_points(row.geometry, other_gdf.unary_union)[1]
        nearest_point = other_gdf.loc[other_gdf.geometry == nearest_geom]
        if nearest_point.geometry.distance(row.geometry).values[0] <= threshold:
            return nearest_point.index[0]  # Return index of the nearest row
        else:
            return None

    # Apply nearest function to find matching rows from rio_GLS
    rio_shp_SWOT['nearest_idx'] = rio_shp_SWOT.apply(nearest, other_gdf=rio_GLS, threshold=threshold, axis=1)

    # Drop rows where no nearest neighbor was found (outside threshold)
    rio_shp_SWOT = rio_shp_SWOT.dropna(subset=['nearest_idx'])

    # Merge the GeoDataFrames using the nearest neighbor index and adding suffix '_GLOW' for rio_GLS columns
    merged_gdf = rio_shp_SWOT.merge(rio_GLS, left_on='nearest_idx', right_index=True, suffixes=('_SWOT', '_GLOW'))

    # Ensure merged_gdf remains a GeoDataFrame by setting its geometry again
    merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry_SWOT')

    # Clean up unnecessary columns
    merged_gdf = merged_gdf.drop(columns=['geometry_GLOW', 'nearest_idx'])

    # Reproject back to original CRS (degrees)
    merged_gdf = merged_gdf.to_crs(epsg=4326)
    
    return merged_gdf
