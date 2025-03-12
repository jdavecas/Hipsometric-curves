"""Python helper functions.

Module Name: xs_api_pull.py
Author: Amanda Whaling (awhaling@usgs.gov)
Last Update: 08/30/2024
"""
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import overload
import math
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from eomaps import Maps
from shapely import wkb
import numpy as np

# Define conversion for transforming between meters and feet
m2ft = 3.2808399


def load_schema(schema_file):
    """Load a schema file from a path.

    Parameters
    ----------
    schema_file : str
        Path to schema JSON file.

    Returns
    -------
    dict
        Schema as a python dictionary.
    """
    try:
        with open(Path(schema_file), "r") as file:
            schema = json.load(file)
            return schema
    except IOError:
        msg = "Error opening schema file."
        print(msg)
        sys.exit(1)


def get_layer_schema(schema):
    """Extract layer types with patterns and create a data type mapping dictionary for each layer type.

    Parameters
    ----------
    schema : dict
        Schema as a python dictionary.

    Returns
    -------
    dict
        Dictionary with layer types as keys. Each value is a dictionary containing:
            - 'pattern': The pattern used for the layer name.
            - 'column_types': A dictionary mapping column names to their expected data types.
    """
    layer_schema = {}

    for layer_type, layer_details in schema["properties"].items():
        pattern = None
        column_types = {}

        # Check if patternProperties exist
        if "patternProperties" in layer_details:
            pattern_details = layer_details["patternProperties"]
            pattern = list(pattern_details.keys())[0]

            # Extract column properties from the patternProperties
            columns = pattern_details[pattern].get("properties", {})

            for col, details in columns.items():
                data_type = details.get("type")

                # Skip null type, but keep track of other types
                if isinstance(data_type, list):
                    data_type = [t for t in data_type if t != "null"]
                    if data_type:
                        data_type = data_type[0]  # Take the first non-null type

                column_types[col] = data_type

        layer_schema[layer_type] = {"pattern": pattern, "column_types": column_types}

    return layer_schema


def coerce_column_types(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Coerce the DataFrame columns to the specified data types from the schema.

    Parameters
    ----------
    df : pandas.DataFrame
        The file path to the GeoPackage.
    column_types : dict
        A dictionary of column names and the data type they should store.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with the columns coerced to the data type specified in the column_types.
    """
    for col, expected_type in column_types.items():
        if col in df.columns:
            if expected_type == "string":
                df[col] = df[col].astype(str)
            elif expected_type == "number":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif expected_type == "integer":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            print(f"Column {col} does not exist in the GeoPackage schema. Check the schema version.")

    return df


@overload
def read_gpkg(gpkg_path: str | Path, layer_names: str, schema_path: str | Path) -> pd.DataFrame | gpd.GeoDataFrame:
    ...


@overload
def read_gpkg(
    gpkg_path: str | Path, layer_names: list[str], schema_path: str | Path
) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    ...


def read_gpkg(
    gpkg_path: str | Path, layer_names: str | list[str], schema_path: str | Path
) -> pd.DataFrame | gpd.GeoDataFrame | dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """Read layers from a GeoPackage file and convert WKB geometries to GeoDataFrames if necessary.

    Parameters
    ----------
    gpkg_path : str or pathlib.Path
        The file path to the GeoPackage.
    layer_names : str or list of str
        The name(s) of the layers to read from the GeoPackage.
    schema_path : str or pathlib.Path
        The file path to the GeoPackage's schema file.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame or dict[str, pandas.DataFrame | geopandas.GeoDataFrame]
        A single pandas.DataFrame or geopandas.GeoDataFrame if only one layer is specified,
        or a dictionary of pandas.DataFrames and/or geopandas.GeoDataFrames if multiple layers are read.

    Raises
    ------
    ValueError
        If the specified layer does not exist, has an unsupported data type,
        or lacks a 'wkb' column for feature layers.

    Notes
    -----
    This function supports reading both attribute and feature layers from a GeoPackage.
    For feature layers, it converts WKB geometries to shapely geometries and creates a geopandas.GeoDataFrame.
    """
    gpkg_path = Path(gpkg_path)  # Convert to Path object if it's a string
    schema_path = Path(schema_path)  # Convert to Path object if it's a string
    single_layer = isinstance(layer_names, str)
    layer_names = [layer_names] if single_layer else layer_names
    data: dict[str, pd.DataFrame | gpd.GeoDataFrame] = {}
    schema = load_schema(schema_path)
    # Get the layer schema with patterns and column types
    layer_schema = get_layer_schema(schema)

    with sqlite3.connect(str(gpkg_path)) as conn:
        cursor = conn.cursor()

        for layer in layer_names:
            cursor.execute("SELECT data_type FROM gpkg_contents WHERE table_name=?", (layer,))
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"Layer {layer} does not exist in the GeoPackage")

            data_type = result[0]
            df = pd.read_sql_query(f'SELECT * FROM "{layer}"', conn)  # noqa: S608, B907

            # Determine the layer type based on the pattern in the schema
            for _layer_type, layer_info in layer_schema.items():
                if re.match(layer_info["pattern"], layer):
                    df = coerce_column_types(df, layer_info["column_types"])
                    break
            if data_type == "attributes":
                data[layer] = df
            elif data_type == "features":
                if "wkb" in df.columns:
                    df["geometry"] = wkb.loads(df["wkb"], hex=True)
                    cursor.execute("SELECT srs_id, organization, organization_coordsys_id FROM gpkg_spatial_ref_sys")
                    crs_info = cursor.fetchall()
                    crs = next(
                        (f"EPSG:{entry[2]}" for entry in crs_info if entry[1].lower() == "epsg" and entry[2] > 0), None
                    )
                    if not crs:
                        raise ValueError("No suitable CRS found in the GeoPackage.")
                    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
                    gdf = gdf.drop(columns=["wkb", "geom"])
                    data[layer] = gdf
                else:
                    raise ValueError(f"Layer {layer} does not contain a 'wkb' column.")
            else:
                raise ValueError(f"Unsupported data type {data_type} for layer {layer}")

    return data[layer_names[0]] if single_layer else data


def fetch_all_layers(gpkg_path: str):
    """Fetches all layer names from a GeoPackage.

    Args:
    - gpkg_path (str): The file path to the GeoPackage.

    Returns:
    - layers (list of str): A list of all layer names in the GeoPackage.
    """
    # Connect to the GeoPackage file
    conn = sqlite3.connect(gpkg_path)
    cursor = conn.cursor()

    # Query to list all table names from gpkg_contents without filtering by data type
    cursor.execute("SELECT table_name FROM gpkg_contents")

    # Fetch results as a list of tuples
    layers = cursor.fetchall()

    # Close the connection
    conn.close()

    # Return just the names of the layers (flatten the list)
    return [layer[0] for layer in layers]


def plot_gpkg(
    gpkg_gdf,
    fig=None,
    plot_col=None,
    title=None,
    legend="Geometries",
    buffer_degrees=1.0,
    basemap="world",
    hist=True,
    plot=True,
):
    """Plots the contents of a GeoPackage.

    Args:
    - gpkg_gdf (GeoDataFrame): The GeoDataFrame containing the geometries to plot.
    - fig (list of int): The figure size (width, height) for the plot.
    - plot_col (tuple or None): A tuple (vmin, vmax, column_name) for plotting a specific column. If None, defaults to 'steelblue' color.
    - title (str, optional): The title of the plot.
    - legend (str): The legend label for the plot.
    - buffer_degrees (float): The buffer distance in degrees around the bounding box of the geometries.
    - basemap (str): The type of basemap to use ("world" or "image").
    - hist (bool): Whether to include a histogram colorbar.
    - plot (bool): Whether to show the plot or return the map object.

    Returns:
    - m (Maps object, optional): The map object if plot=False, otherwise None.
    """
    # Set-up figure
    if fig is None:
        fig = [10, 8]

    target_crs = 4326

    # Ensure the GeoDataFrame is fully loaded
    gpkg_geom = gpkg_gdf.copy()
    _ = gpkg_geom.geometry.apply(lambda geom: geom.is_valid)

    if gpkg_geom.crs.to_epsg() != target_crs:
        gpkg_geom = gpkg_geom.to_crs(target_crs)

    m = Maps(crs=target_crs, ax=None)
    layout = {
        "figsize": fig,
        "0_map": [0.02, 0.02, 0.96, 0.95],  # map position
    }
    m.apply_layout(layout)

    bbox = gpkg_geom.total_bounds
    print("Bounding Box:", bbox)
    extent = [bbox[0] - buffer_degrees, bbox[2] + buffer_degrees, bbox[1] - buffer_degrees, bbox[3] + buffer_degrees]

    m.set_extent(crs=target_crs, extents=extent)

    if basemap == "world":
        m.add_feature.preset.coastline(lw=0.5, zorder=1)
        m.add_feature.preset.ocean(color="lightgray", zorder=2)
    elif basemap == "image":
        m.add_wms.ESRI_ArcGIS.SERVICES.World_Imagery.add_layer.xyz_layer()

    if plot_col:
        vmin, vmax, col = plot_col
        if col not in gpkg_geom.columns:
            raise ValueError(f"Column {col!r} not found in GeoDataFrame")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors = plt.cm.plasma(norm(gpkg_geom[col].values))
        gpkg_geom["x"] = gpkg_geom.geometry.x.values
        gpkg_geom["y"] = gpkg_geom.geometry.y.values
        m.set_data(data=gpkg_geom, x="x", y="y", parameter=col, crs=target_crs)
        m.set_shape.scatter_points(size=2)
        m.plot_map(vmin=vmin, vmax=vmax, indicate_masked_points=True, set_extent=False, zorder=3)
        if hist:
            hist_size = 0.8
            hist_label = legend
            label = f"Min:{gpkg_geom[col].min()}, Max:{gpkg_geom[col].max()}"
        else:
            hist_size = 0
            hist_label = None
            label = f"{legend}\n(Min:{gpkg_geom[col].min()}, Max:{gpkg_geom[col].max()})"
        cb = m.add_colorbar(label=label, orientation="vertical", hist_size=hist_size, hist_label=hist_label)
        cb.set_position([0.7, 0.1, 0.2, 0.8])
    else:
        colors = "steelblue"
        m.add_gdf(gpkg_geom, set_extent=False, color=colors, markersize=2.5)

        legend_position = (extent[0] + (extent[1] - extent[0]) * 0.01, extent[3] - (extent[3] - extent[2]) * 0.05)

        m.add_annotation(
            xy=legend_position,
            xy_crs=target_crs,
            text=f"{legend} (blue points)\nTotal: {gpkg_geom.shape[0]}",
            xytext=(2, 1),
            horizontalalignment="left",
            verticalalignment="top",
            color="steelblue",
            arrowprops=dict(ec="g", arrowstyle="-"),
            bbox=dict(fc="white", alpha=0.8),
        )

    m.add_title(title)
    if plot:
        m.show()
    else:
        return m


def plot_ms_xs(xs_df, color, units="feet"):
    """Plots cross-sectional profiles from mid-section measurements.

    Args:
    - xs_df (DataFrame): A DataFrame containing cross-sectional measurement data with 'DistanceFromInitialPointMeasure_ft' and
    'DepthMeasure_ft' columns.
    - color (str): The color of the plot line.
    - units (str): The units to plot the data in (feet or meters). Default is feet.

    Returns:
    - None
    """
    if units == "meters":
        xs_df["DistanceFromInitialPointMeasure_ft"] = xs_df["DistanceFromInitialPointMeasure_ft"] / m2ft
        xs_df["DepthMeasure_ft"] = xs_df["DepthMeasure_ft"] / m2ft
    plt.plot(
        xs_df["DistanceFromInitialPointMeasure_ft"],
        -1 * xs_df["DepthMeasure_ft"],
        marker="o",
        linestyle="-",
        color=color,
    )
    plt.xlabel("Distance From Initial Point (" + units + ")")
    plt.ylabel("Depth (" + units + ")")
    plt.grid(True)

def plot_mb_xs_dict(xs_dict, dict2, colors=None, units="meters", depth_column="Depth_NAVD88(m)", cols=3):
    """Plots multiple cross-sectional profiles from moving-boat ADCP measurements stored in a dictionary.

    Args:
    - xs_dict (dict): Dictionary where keys are identifiers and values are DataFrames with 'Distance_Meters' and depth column.
    - colors (list of str, optional): List of colors for each plot. If None, it uses a default colormap.
    - units (str): The units to plot the data in ("feet" or "meters").
    - depth_column (str): Column name for depth values.
    - cols (int): Number of columns in the grid layout.

    Returns:
    - None
    """
    # Filter out DataFrames that do not have the required depth column
    filtered_xs_dict = {k: v for k, v in xs_dict.items() if depth_column in v.columns}

    num_profiles = len(filtered_xs_dict)
    rows = math.ceil(num_profiles / cols)  # Determine the number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()  # Flatten for easy iteration

    # Use a default colormap if colors are not provided
    if colors is None:
        cmap = plt.cm.get_cmap("tab10", num_profiles)
        colors = [cmap(i) for i in range(num_profiles)]

    # Iterate over the filtered dictionary items
    for i, (key, xs_df) in enumerate(filtered_xs_dict.items()):
        ax = axes[i]  # Select subplot
        
        # Extract the numeric ID (before "_mb")
        base_id = key.split('_mb')[0]

        # Check if base_id is in dict2 before accessing it
        if base_id in dict2 and "elevation_NAVD88(m)" in dict2[base_id]:
            datum_elevation = dict2[base_id]["elevation_NAVD88(m)"].iloc[0]
        else:
            print(f"Warning: Missing elevation data for {base_id}. Skipping plot.")
            continue  # Skip this iteration to prevent KeyError

        if units == "feet":
            xs_df["Distance_Meters"] = xs_df["Distance_Meters"] * m2ft
            xs_df[depth_column] = xs_df[depth_column] * m2ft

        ax.plot(xs_df["Distance_Meters"], xs_df[depth_column],
                linewidth=2, marker="o", markersize=2, linestyle="-", color=colors[i])
        ax.set_xlabel(f"Distance ({units})")
        ax.set_ylabel(f"Depth ({units})")
        ax.set_title(f'Cross-section: {key}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_facecolor('#f2f2f2')

        # Adjust y-axis to display absolute values
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])  

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def plot_mb_xs_grid(xs_df, color, units="meters", depth_column=None,cols=3):
    """Plots a grid of cross-sectional profiles from moving-boat ADCP measurements.

    Args:
    - xs_dict (dict): A dictionary where keys are IDs and values are DataFrames containing cross-sectional measurement data.
    - color (str): The color of the plot lines.
    - units (str): The units to plot the data in (feet or meters).
    - depth_column (str): The column name for depth values.
    - cols (int): Number of columns in the grid layout.

    Returns:
    - None
    """
    if not xs_df:
        print("No data to plot.")
        return

    num_plots = len(xs_df)
    rows = int(np.ceil(num_plots / cols))  # Determine the required rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for i, (key, xs_df) in enumerate(xs_df.items()):
        ax = axes[i]

        if units == "feet":
            xs_df["Distance_Meters"] = xs_df["Distance_Meters"] * m2ft
            xs_df[depth_column] = xs_df[depth_column] * m2ft

        ax.plot(xs_df["Distance_Meters"], xs_df[depth_column] , linewidth=2, marker="o",
                markersize=2, linestyle="-", color=color)
        ax.set_xlabel(f"Distance ({units})")
        ax.set_ylabel(f"Depth ({units})")
        ax.set_title(f'Profile: {key}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_facecolor('#f2f2f2')

        # Modify y-axis to display absolute values
        y_ticks = ax.get_yticks()
        y_tick_labels = [y for y in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
    # Hide empty subplots if num_plots is not a perfect multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

