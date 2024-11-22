import random
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def l_regression(river, min_spearman=None, min_obs=0, show_p_value=True, min_p_value=0.05):
    """
    Generates hypsometric scatter plots for randomly selected river nodes,
    adding linear regression with R², slope, and intercept.
    Filters nodes based on minimum p-value and stores statistics for all nodes in a DataFrame.

    Args:
        river (dict): Dictionary containing node data with 'width' and 'wse' keys.
        min_spearman (float or None): Minimum Spearman correlation value to include a node in the plot.
                                      If None, plots are divided equally above and below a Spearman value of 0.4.
        min_obs (int): Minimum number of observations required to display a scatter plot for a node (default: 10).
        show_p_value (bool): If True, displays the p-value on each scatter plot (default: True).
        min_p_value (float): Minimum p-value required to include a node in the plot (default: 0.05).
    """
    # Prepare list to store statistics for all nodes
    results = []

    # Separate nodes into two lists for random plotting selection
    above_threshold = []
    below_threshold = []
    
    for node_id, node_data in river.items():
        # Ensure each node has at least `min_obs` width-WSE pairs
        if len(node_data['width']) < min_obs:
            continue  # Skip nodes with fewer observations than min_obs
            
        # Calculate Spearman correlation and p-value
        spearman_corr, p_value = scipy.stats.spearmanr(node_data['width'], node_data['wse'])
        
        # Filter based on p-value
        if p_value >= min_p_value:
            continue  # Skip nodes with p-values above or equal the threshold
        
        # Perform linear regression
        width = np.array(node_data['width']).reshape(-1, 1)
        wse = np.array(node_data['wse'])
        reg = LinearRegression().fit(width, wse)
        slope = round(reg.coef_[0], 3)
        intercept = round(reg.intercept_, 3)
        r2 = reg.score(width, wse)

        # Store node data in results DataFrame
        results.append({
            'Node': node_id,
            'Spearman': round(spearman_corr, 3),
            'p_value': round(p_value, 3),
            'R2': round(r2, 3),
            'Slope': slope,
            'Intercept': intercept
        })

        # Divide nodes based on Spearman correlation threshold
        if spearman_corr >= 0.4:
            above_threshold.append((node_id, spearman_corr, p_value, slope, intercept))
        else:
            below_threshold.append((node_id, spearman_corr, p_value, slope, intercept))

    # Determine the nodes to plot based on min_spearman value
    if min_spearman is None:
        # Select 10 nodes above 0.4 and 10 nodes below 0.4, if available
        num_above = min(len(above_threshold), 10)
        num_below = min(len(below_threshold), 10)
        selected_above = random.sample(above_threshold, num_above)
        selected_below = random.sample(below_threshold, num_below)
        random_nodes = selected_above + selected_below
    else:
        # Select up to 20 nodes that meet the min_spearman threshold
        filtered_above = [node for node in above_threshold if node[1] >= min_spearman]
        num_above = min(len(filtered_above), 20)
        random_nodes = random.sample(filtered_above, num_above)

    # Create scatter plots in a 4x5 grid (up to 20 plots)
    plt.figure(figsize=(20, 15))
    for i, (node_id, spearman_corr, p_value, slope, intercept) in enumerate(random_nodes, 1):
        node_data = river[node_id]
        width = np.array(node_data['width']).reshape(-1, 1)
        wse = np.array(node_data['wse'])

        # Refit the model per plot to verify consistency
        reg = LinearRegression().fit(width, wse)
        slope = round(reg.coef_[0], 3)
        intercept = round(reg.intercept_, 3)
        r2 = round(reg.score(width, wse), 3)
        
        # Create subplot
        plt.subplot(4, 5, i)
        plt.scatter(node_data['width'], node_data['wse'], alpha=1, c="darkcyan", edgecolors='cyan', linewidths=1)
        
        # Plot regression line
        x_range = np.linspace(width.min(), width.max(), 100)
        y_range = slope * x_range + intercept
        plt.plot(x_range, y_range, color="red", linestyle="--", linewidth=1)
        
        # Title and labels
        plt.title(f"Node: {node_id}\nSpearman: {spearman_corr:.2f}, R²: {r2:.2f}", fontsize=10)
        plt.xlabel('Width')
        plt.ylabel('WSE')

        # Display p-value, slope, and intercept in smaller font if show_p_value is True
        if show_p_value:
            plt.text(0.05, 0.85, f"p-value: {p_value:.3f}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=8, color="gray")
        plt.text(0.05, 0.75, f"Slope: {slope}\nIntercept: {intercept}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=8, color="gray")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Create DataFrame from results for all nodes and export with 3 decimals
    results_df = pd.DataFrame(results)
    results_df = results_df.round({'Spearman': 3, 'p_value': 3, 'R2': 3, 'Slope': 3, 'Intercept': 3})
    
    return results_df