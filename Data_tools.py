"""Python helper functions.

Module Name: data_tools.py
Author: J. Daniel Velez (davelez@unc.edu)
Created: 11/02/2025

Goal: To transform data formats 
"""

import os
import pandas as pd
import geopandas as gpd
import glob
import os
import math
import duckdb
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from affine import Affine
import duckdb


###################################################################################################################################################
###################################################################################################################################################

"""Raster resampling
Raster of different resolution are resamplig up to any resolution
"""

def _utm_epsg_for_lonlat(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    north = lat >= 0
    return f"EPSG:{32600 + zone if north else 32700 + zone}"

def resampling(input_raster, output_raster, scale_factor):
    """
    Resample to target meter resolution (scale_factor, e.g., 10 -> 10 m).
    If the source CRS is geographic (degrees), reproject to auto-UTM first,
    then resample to the requested meter resolution.
    """

    with rasterio.open(input_raster) as src:
        target_res_m = float(scale_factor)

        # --- Decide destination CRS ---
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Input raster has no CRS; cannot infer meters vs degrees.")

        # Compute raster centroid in its native CRS
        cx = (src.bounds.left + src.bounds.right) / 2.0
        cy = (src.bounds.top + src.bounds.bottom) / 2.0

        # Heuristic: if CRS is geographic (degrees), pick auto-UTM from lon/lat
        if src_crs.is_geographic:
            # centroid is already lon/lat in a geographic CRS
            dst_crs = _utm_epsg_for_lonlat(cx, cy)
        else:
            # Projected CRS; assume linear units are meters (common for UTM/state-plane meters)
            dst_crs = src_crs

        # --- Compute target grid (transform, width, height) at target meter resolution ---
        # If src is in degrees and dst is UTM, this step handles reprojection + resampling grid.
        # If src is already in meters, this simply changes pixel size to target_res_m.
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=(target_res_m, target_res_m)
        )

        # --- Build output profile ---
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": max(1, dst_width),
            "height": max(1, dst_height),
            "compress": "lzw",
            "tiled": True
        })
        if src.nodata is not None:
            profile.update({"nodata": src.nodata})

        # --- Reproject + resample into the destination grid ---
        with rasterio.open(output_raster, 'w', **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.average,  # keep your choice; use 'nearest' for classes
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata
                )

##################################################################################################
#################################################
#################################################
"""CVS to Parquet conversion using DuckDB"""

def csv_to_parquet(in_csv, out_parquet):
    con = duckdb.connect()  # create a connection
    con.execute("""
        COPY (
            SELECT *
            FROM read_csv_auto(?)
        )
        TO ?
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """, [in_csv, out_parquet])
    con.close()
    print(f" Wrote: {out_parquet}")

##################################################################################################
#################################################
#################################################
"""Merging multiple shapefiles into one"""

def merge_shapefiles(folder, output_shapefile, target_epsg=None):
    """
    Merge shapefiles in a folder only if 'nodes' is in the filename (case-insensitive).
    Ensures only point geometries are kept and CRS is consistent.

    Parameters
    ----------
    folder : str | Path
        Folder containing shapefiles.
    output_shapefile : str | Path
        Output shapefile (.shp)
    target_epsg : int | None
        Reproject to this EPSG code if provided.

    Returns
    -------
    GeoDataFrame
        The merged point dataset.
    """
    folder = str(folder)

    # ✅ Only shapefiles containing "nodes" (case-insensitive)
    shp_files = sorted(glob.glob(os.path.join(folder, "*nodes*.shp")))

    if not shp_files:
        raise FileNotFoundError(f"No node shapefiles found in: {folder}")

    print(f" Found {len(shp_files)} node shapefiles")

    gdfs = []
    base_crs = None

    for shp in shp_files:
        print(f" Reading: {os.path.basename(shp)}")
        gdf = gpd.read_file(shp)

        #  Keep only POINT geometries
        gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]
        if gdf.empty:
            print(f" No points found, skipping: {os.path.basename(shp)}")
            continue

        #  CRS handling
        if target_epsg is not None:
            gdf = gdf.to_crs(epsg=int(target_epsg))
        else:
            if base_crs is None:
                base_crs = gdf.crs
            elif gdf.crs != base_crs:
                print(f" Reprojecting → {base_crs}")
                gdf = gdf.to_crs(base_crs)

        gdfs.append(gdf)

    if not gdfs:
        raise ValueError("No point features remained after filtering.")

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    print(f" Total nodes merged: {len(merged_gdf):,}")
    print(f" Saving merged file → {output_shapefile}")
    merged_gdf.to_file(output_shapefile)

    print("🎯 Done! Node shapefiles merged successfully.")
    return merged_gdf

##################################################################################################
#################################################
#################################################
"""shapefile to parquet conversion using DuckDB"""

def shapefile_to_parquet(in_shapefile, out_parquet, to_epsg=None,
                         overwrite="raise", partition_by=None, compression="ZSTD"):
    """
    Convert a Shapefile to GeoParquet using DuckDB Spatial.

    Parameters
    ----------
    in_shapefile : str | Path
        Path to .shp
    out_parquet : str | Path
        Path to .parquet (or a directory if using partition_by)
    to_epsg : int | None
        Reproject geometry to this EPSG (e.g., 4326). If None, keep source CRS.
    overwrite : {'raise','overwrite','ignore'}
        Behavior if output exists: error, overwrite, or skip writing.
    partition_by : list[str] | None
        Write partitioned GeoParquet (directory output) by these column(s).
    compression : {'ZSTD','SNAPPY','GZIP',...}
        Parquet compression codec.

    Returns
    -------
    str
        The output path (or directory).
    """
    in_shapefile = str(in_shapefile)
    out_parquet = str(out_parquet)

    if not os.path.exists(in_shapefile):
        raise FileNotFoundError(f"Input shapefile not found: {in_shapefile}")

    # Handle overwrite policy
    if partition_by is None:
        # writing a single parquet file
        if os.path.exists(out_parquet):
            if overwrite == "raise":
                raise FileExistsError(f"Output exists: {out_parquet}")
            elif overwrite == "ignore":
                print(f"⚠️ Output exists, skipping (ignore): {out_parquet}")
                return out_parquet
            # overwrite -> proceed
    else:
        # partitioned write -> target is a directory
        if os.path.isdir(out_parquet) and os.listdir(out_parquet):
            if overwrite == "raise":
                raise FileExistsError(f"Output directory not empty: {out_parquet}")
            elif overwrite == "ignore":
                print(f"⚠️ Output dir non-empty, skipping (ignore): {out_parquet}")
                return out_parquet

    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")

        # Build COPY options
        copy_opts = [f"FORMAT PARQUET", f"COMPRESSION {compression}"]
        if overwrite in ("overwrite",):
            copy_opts.append("OVERWRITE_OR_IGNORE TRUE")  # overwrite if exists
        elif overwrite in ("ignore",):
            copy_opts.append("OVERWRITE_OR_IGNORE TRUE")  # will no-op if exists
        if partition_by:
            cols = ", ".join(partition_by)
            copy_opts.append(f"PARTITION_BY ({cols})")

        copy_clause = ", ".join(copy_opts)

        if to_epsg is None:
            sql = f"""
                COPY (
                    SELECT * FROM ST_Read(?)
                )
                TO ?
                ({copy_clause});
            """
            params = [in_shapefile, out_parquet]
        else:
            # Transform geometry and avoid duplicate geom column
            sql = f"""
                COPY (
                    SELECT
                        ST_Transform(geom, ?) AS geom,
                        * EXCLUDE geom
                    FROM ST_Read(?)
                )
                TO ?
                ({copy_clause});
            """
            params = [int(to_epsg), in_shapefile, out_parquet]

        con.execute(sql, params)
        print(f"✅ Wrote GeoParquet → {out_parquet}")
        return out_parquet

    finally:
        con.close()

##################################################################################################
#################################################
#################################################
"""Merge ONE shapefile with a Parquet file based on a common attribute column"""

def merge_slope_to_nodes(parquet_path, shp_path, out_path):
    """
    Merge slope data from Parquet into a shapefile based on matching node_id,
    keep only the first slope per node, and export the result as GeoParquet.

    Parameters
    ----------
    parquet_path : str
        Path to Parquet file with columns ["node_id", "slope1"]
    shp_path : str
        Path to Shapefile with column ["node_id"]
    out_path : str
        Path where the GeoParquet result will be saved
    """

    print(" Loading Parquet...")
    pq = pd.read_parquet(parquet_path, columns=["node_id", "slope1"])
    pq["node_id"] = pq["node_id"].astype(str)

    # Keep first slope per node_id
    pq_first = (
        pq.dropna(subset=["node_id"])
          .drop_duplicates(subset=["node_id"], keep="first")
          [["node_id", "slope1"]]
    )
    print(f" Unique nodes in Parquet: {len(pq_first):,}")

    print("\n Loading Shapefile...")
    gdf = gpd.read_file(shp_path, engine="pyogrio")
    gdf["node_id"] = gdf["node_id"].astype(str)
    print(f" Nodes in shapefile: {len(gdf):,}")

    # Merge
    print("\n🔗 Merging...")
    gdf_merged = gdf.merge(pq_first, on="node_id", how="left", validate="m:1")

    missing_count = gdf_merged["slope1"].isna().sum()
    print(f" Unmatched nodes dropped: {missing_count:,}")

    # Drop missing slope values
    gdf_merged = gdf_merged.dropna(subset=["slope1"])
    gdf_merged["slope1"] = pd.to_numeric(gdf_merged["slope1"], errors="coerce")

    print(f" Final matched nodes: {len(gdf_merged):,}")

    # Export
    print("\n Saving GeoParquet...")
    gdf_merged.to_parquet(out_path, index=False)

    print("\n Merge complete!")
    print(f" Output saved to: {out_path}")
    print(f" Total rows written: {len(gdf_merged):,}")

    return gdf_merged
