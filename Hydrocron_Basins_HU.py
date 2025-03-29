# Description: This script retrieves the node IDs of a basin from the FTS API and then queries the Hydrocron API for each node ID.
#              The results are saved to a CSV file.
import dask
import dask.dataframe as dd
from dask.distributed import Client
import hvplot.dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import datetime
import psutil
from io import StringIO

# Set up Dask with SLURM's CPUs
client = Client(n_workers=40, threads_per_worker=1, memory_limit='16GB')  # 40 parallel workers

# Assign URLs to variables for APIs we uses FTS and Hydrocron
FTS_URL = "https://fts.podaac.earthdata.nasa.gov/v1"  
HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"

# Basin identifier for FTS query
BASIN_IDENTIFIER = "732"

# Function to query FTS
def query_fts(query_url, params):
    """Query Feature Translation Service (FTS) for reach identifiers using the query_url parameter.

    Parameters
    ----------
    query_url: str - URL to use to query FTS
    params: dict - Dictionary of parameters to pass to query

    Returns
    -------
    dict of results: hits, page_size, page_number, reach_ids
    """
    nodes = requests.get(query_url, params=params)
    nodes_json = nodes.json()
    
    hits = nodes_json["hits"]
    if 'search on' in nodes.json.keys():
        page_size = nodes_json['search on']['page_size']
        page_number = nodes_json['search on']['page_number']
    else:
        page_size = 0
        page_number = 0

    return {
        "hits": hits,
        "page_size": page_size,
        "page_number": page_number,
        "node_ids": [item["node_id"] for item in nodes_json["results"]]
    }

# Retrieve node IDs by searching for the basin code
query_url = f"{FTS_URL}/rivers/node/{BASIN_IDENTIFIER}"
print(f"Searching by basin... {query_url}")

page_size = 100 # Retrieve 100 results per request
page_number = 1 # Start at page 1
node_ids = [] # Initialize empty list ofr IDs

while True:
    params = {"page_size": page_size, "page_number": page_number}
    results = query_fts(query_url, params)

    # Check if the response contains valid node_ids
    if "node_ids" not in results or not results["node_ids"]:
        print(f"No more node IDs returned for page {page_number}. Ending loop.")
        break

    hits  = results.get("hits", 0)
    node_ids.extend(results['node_ids'])

    print(
        f"page_size: {page_size},"
        f"page_number: {page_number},"
        f"hits: {hits},"
        f"# node_ids: {len(node_ids)}"
    )

    # Stop condition: all hits fetched
    if len(node_ids) >= hits:
        print("All available node IDs fetched. Ending loop.")
        break

    page_number += 1

print("Total number of nodes: ", len(node_ids))
node_ids = list(set(node_ids))  # Remove duplicates
print("Total number of unique nodes: ", len(node_ids))

# Function to query Hydrocron
def query_hydrocron(query_url, node_id, start_time, ent_time, fields):
    """Query Hydrocron for node-level time series data.

    Parameters
    ----------
    query_url: str - URL to use to query FTS
    node_id: str - String SWORD node identifier
    start_time: str - String time to start query
    end_time: str - String time to end query
    fields: list - List of fields to return in query response
    empty_df: pandas.DataFrame that contains empty query results

    Returns
    -------
    pandas.DataFrame that contains query results
    """

    params = {
        "feature": "Node",
        "feature_id": node_id,
        "output": "csv",
        "start_time": start_time,
        "end_time": end_time,
        "fields": fields
    }

    results = requests.get(query_url, params=params)
    if "results" in results.json().keys():
        results_csv = results.json()["results"]["csv"]
        df = pd.read_csv(StringIO(results_csv))
    else:
        df = pd.DataFrame({
            "node_id": [np.int64(node_id)],
            "time_str": [datetime.datetime(1900, 1, 1).strftime("%Y-%m-%dT%H:%M:%S")],
            "wse": [-999999999999.0],
            "width": [-999999999999.0],
    })
        
    return df

# Define parameters

start_time = "2023-07-01T00:00:00Z"
end_time = "2025-03-10T00:00:00Z"
fields = "node_id,time_str,lat,lon,wse,width,node_q_b,dark_frac,ice_clim_f,ice_dyn_f,p_dist_out"

# Batch processing to reduce graph size
batch_size = 500
batched_results = []

for i in range(0, len(node_ids), batch_size):
    batch = node_ids[i : i + batch_size]

    # Create a delayed queries for the batch
    delayed_queries = [query_hydrocron(HYDROCRON_URL, node, start_time, end_time, fields) for node in batch]

    # Compute batch inmediately to avoid an excessively large graph
    batch_results = dask.compute(*delayed_queries)

    # Convert batcvh results to Dask DataFrame
    ddf_batch = dd.from_pandas(pd.concat(batch_results, ignore_index=True), npartitions=40)
    batched_results.append(ddf_batch)


params = {"page_size": 100, "page_number": 1}
results = query_fts(query_url, params)
node_ids = list(set(results["node_ids"]))  # Remove duplicates

# Function to query Hydrocron
@dask.delayed
def query_hydrocron(node_id, start_time, end_time, fields):
    params = {
        "feature": "Node",
        "feature_id": node_id,
        "output": "csv",
        "start_time": start_time,
        "end_time": end_time,
        "fields": fields
    }
    response = requests.get(HYDROCRON_URL, params=params)

    if "results" in response.json():
        df = pd.read_csv(StringIO(response.json()["results"]["csv"]))
    else:
        df = pd.DataFrame({
            "node_id": [node_id],
            "time_str": [datetime.datetime(1900, 1, 1).strftime("%Y-%m-%dT%H:%M:%S")],
            "wse": [-999999999999.0],
            "width": [-999999999999.0]
        })

    return df

# Define parameters
start_time = "2023-07-28T00:00:00Z"
end_time = "2024-04-16T00:00:00Z"
fields = "node_id,time_str,lat,lon,wse,width,node_q_b,dark_frac,ice_clim_f,ice_dyn_f,p_dist_out"

# Parallel batch processing
batch_size = 500  
batched_results = []

for i in range(0, len(node_ids), batch_size):
    batch = node_ids[i : i + batch_size]
    delayed_queries = [query_hydrocron(node, start_time, end_time, fields) for node in batch]
    batch_results = dask.compute(*delayed_queries)  # Run in parallel
    ddf_batch = dd.from_pandas(pd.concat(batch_results, ignore_index=True), npartitions=40)
    batched_results.append(ddf_batch)

# Combine all batches
ddf = dd.concat(batched_results)

# Save the result
ddf.compute().to_csv("processed_results.csv", index=False)


