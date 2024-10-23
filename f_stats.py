import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import geopandas as gpd
import matplotlib.ticker as ticker


def S_correlation(river_dict):
    N_Spearman = {}
    spearman_count = 0
    for id, variables in river_dict.items():
        width_values = variables['width']
        wse_values = variables['wse']

        # Ensure both lists have the same length
        if len(width_values) != len(wse_values):
            print(f"Warning: Length mismatch for node {id}. width_values has length {len(width_values)}, but wse_values has length {len(wse_values)}.")
            # Optionally, you can skip this entry or handle it as needed
            continue

        paired_values = list(zip(width_values, wse_values))
        num_pairs = len(paired_values)
        spearman_corr, p_value = scipy.stats.spearmanr(width_values, wse_values)

        N_Spearman[id] = {
            'spearman_corr': spearman_corr,
            'p_value': p_value,
            'num_pairs': num_pairs
        }
        spearman_count =+ 1

    positive_spearman_count = 0
    negative_spearman_count = 0
    zero_spearman_count = 0
    spearman_above_threshold_count = 0  # New counter for correlations >= 0.55

    # Iterate through the results dictionary
    for node_id, result in N_Spearman.items():
        spearman_corr = result['spearman_corr']
    
        if spearman_corr is not None:
            if spearman_corr > 0:
                positive_spearman_count += 1
                if spearman_corr >= 0.4:
                    spearman_above_threshold_count += 1
            elif spearman_corr < 0:
                negative_spearman_count += 1
            else:
                zero_spearman_count += 1
    
    for id, result in N_Spearman.items():
        print(f"Node {id}: Spearman Correlation = {result['spearman_corr']}, p-value = {result['p_value']}, Number of pairs = {result['num_pairs']}")
        # Display the total counts
    print(f"Number of positive Spearman correlations: {positive_spearman_count}")
    print(f"Number of negative Spearman correlations: {negative_spearman_count}")
    print(f"Number of zero Spearman correlations: {zero_spearman_count}")
    print(f"Number of Spearman correlations >= 0.4: {spearman_above_threshold_count}")

    N_Spearman_df = pd.DataFrame.from_dict(N_Spearman.copy(), orient='index')
    N_Spearman_df = N_Spearman_df.reset_index()
    # Display the DataFrame
    return N_Spearman_df

def plot_multiple_cdfs(df, river, min_num_pairs=0, max_num_pairs=None):
    # Ensure the num_pairs column is treated as integers
    df.iloc[:, 3] = pd.to_numeric(df.iloc[:, 3], errors='coerce').astype('Int64')

    # Filter and sort unique num_pairs within the specified range
    if max_num_pairs is not None:
        unique_num_pairs = np.sort(df[(df.iloc[:, 3] >= min_num_pairs) & 
                                      (df.iloc[:, 3] <= max_num_pairs)].iloc[:, 3].unique())
    else:
        unique_num_pairs = np.sort(df[df.iloc[:, 3] >= min_num_pairs].iloc[:, 3].unique())

    plt.figure(figsize=(12, 6))  # Increase figure width to make space for the legend

    for num_pairs in unique_num_pairs:
        # Filter the dataframe for the current num_pairs
        filtered_df = df[df.iloc[:, 3] == num_pairs]
        spearman_coefficients = filtered_df.iloc[:, 1]
        
        # Sort the coefficients for CDF calculation
        sorted_data = np.sort(spearman_coefficients)
        yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
        
        # Plot the CDF using dots
        plt.plot(sorted_data, yvals, '.', label=f'num_pairs = {num_pairs}')
    
    plt.xlabel('Spearman Coefficient')
    plt.ylabel('CDF')
    plt.title(f'CDFs for {river} river')
    # Place the legend outside the figure, on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make space for the legend
    plt.show()

def geojoin(gdf,df,reach_node='node_id'):
    """
    Merges a dataframe and a geodataframe base on the node_id or reach_id

    Args:
        reach_node (_type_): should be node_id or reach_id

    Returns:
        A geodataframe where each node_id or reach_id has the columns coming from 
        the dataframe
    """
    df[reach_node] = df[reach_node].astype(str)
    gdf[reach_node] = gdf[reach_node].astype(str)

    merged_gdf = gdf.merge(df, on=reach_node, how='left')

    return merged_gdf

def hypsometric(river, min_spearman=None):
    """
    Generates hypsometric scatter plots for randomly selected river nodes.
    Ensures 50% of scatter plots have Spearman correlation above 0.4, and 50% below 0.39.
    Optionally filters nodes by Spearman correlation between 'width' and 'wse' if a threshold is provided.

    Args:
        river (dict): Dictionary containing node data with 'width' and 'wse' keys.
        min_spearman (float or None): Minimum Spearman correlation value to include a node in the plot. 
                                      If None, no filtering is applied (default: None).
    """
    # Separate nodes based on Spearman correlation threshold
    above_threshold = []
    below_threshold = []
    
    for node_id in river:
        node_data = river[node_id]
        spearman_corr, _ = scipy.stats.spearmanr(node_data['width'], node_data['wse'])
        
        if spearman_corr >= 0.4:
            above_threshold.append((node_id, spearman_corr))
        elif spearman_corr <= 0.39:
            below_threshold.append((node_id, spearman_corr))

    # Randomly select nodes for plotting, 50% from each group
    num_above = min(len(above_threshold), 10)  # 50% from above 0.4
    num_below = min(len(below_threshold), 10)  # 50% from below 0.39

    selected_above = random.sample(above_threshold, num_above)
    selected_below = random.sample(below_threshold, num_below)

    # Combine selected nodes
    random_nodes = selected_above + selected_below

    # Create scatter plots in a 4x5 grid (up to 20 plots)
    plt.figure(figsize=(20, 15))
    for i, (node_id, spearman_corr) in enumerate(random_nodes, 1):
        node_data = river[node_id]

        # Create subplot
        plt.subplot(4, 5, i)  # 4x5 grid for up to 20 plots
        plt.scatter(node_data['width'], node_data['wse'], alpha=1, c="darkcyan", edgecolors='cyan', linewidths=1)
        plt.title(f"Node: {node_id}\nSpearman: {spearman_corr:.2f}")
        plt.xlabel('Width')
        plt.ylabel('WSE')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()




def profiles(river1, river2=None, river3=None):
    plt.figure(figsize=(15, 6))
    
    # First profile
    if river1 is not None:
        grouped1 = river1.groupby('p_dist_out')['width'].agg(['min', 'max', 'median']).reset_index()
        plt.fill_between(grouped1['p_dist_out'], grouped1['min'], grouped1['max'], color='b', alpha=0.3, label='Profile 16 bits Width Range')
       # plt.plot(grouped1['p_dist_out'], grouped1['median'], color='b', linestyle='-', label='Profile 1 Median')
    
    # Second profile
    if river2 is not None:
        grouped2 = river2.groupby('p_dist_out')['width'].agg(['min', 'max', 'median']).reset_index()
        plt.fill_between(grouped2['p_dist_out'], grouped2['min'], grouped2['max'], color='g', alpha=0.2, label='Profile 8 bits Width Range')
        #plt.plot(grouped2['p_dist_out'], grouped2['median'], color='g', linestyle='-', label='Profile 2 Median')

    # Third profile
    if river3 is not None:
        grouped3 = river3.groupby('p_dist_out')['width'].agg(['min', 'max', 'median']).reset_index()
        plt.fill_between(grouped3['p_dist_out'], grouped3['min'], grouped3['max'], color='r', alpha=0.3, label='Profile 4 bits Width Range')
       # plt.plot(grouped3['p_dist_out'], grouped3['median'], color='r', linestyle='-', label='Profile 3 Median')

    # Common plotting elements
    plt.autoscale()
    plt.xlabel('Distance from the outlet (m)')
    plt.ylabel('Width')
    plt.title("Width Profiles Comparison")
    plt.grid(which='both', linestyle='--', linewidth=0.3)
    plt.minorticks_on()

    # Disable scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')

    # Format minor ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(40000))
    ax.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))

    # Rotate major and minor tick labels
    plt.xticks(rotation=45)
    for label in ax.get_xticklabels(minor=True):
        label.set_rotation(45)

    plt.legend()
    plt.show()


