import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random
import contextily as ctx
from sklearn.linear_model import HuberRegressor
from matplotlib.patches import FancyArrow
import os
from f_filter_process import call_file, export_dataframe
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TheilSenRegressor
from scipy.optimize import least_squares
import statsmodels.api as sm
from statsmodels. robust.norms import TukeyBiweight
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression
import ruptures as rpt

# Function to test normality in each node

def test_normality(gdf, node_col, width_col, wse_col, alpha=0.05):
    """_summary_
    This script performs normality test on the residuals of HUber regression for each node in a GeoDataFrame, specifically:
    1. Fits a Huber regression model (WSE ~ Width) for each node.
    2. Extracts residuals from the regression.
    3. Uses Shapiro-Wilk test to check normality of the residuals.
    4. Generates Q-Q plots for visual inspection in 20 randomly selected nodes.
    5. Plots a histogam of p-values from the normality test.
    6. Creates a heatmap indicating which nodes pass or fail the normality test.

    Args:
        gdf (GeoDataFrame): Inout geospatial dataframe containing node data.
        node_col (str): Column name representing unique node identifiers.
        width_col (str): Column name containing width value.
        wse_col (str): Column name containing water surface elevation (WSE) values.
        alpha (float, default=0.05): Significance level for the normality test.

    Returns:
        - Updated GeoDataFrame with added columns:
            - 'p_value_norm': p-value from the Shapiro-Wilk normality test.
            - 'normal_flag': Boolean flag indicating whether the node passes the normality test.
        - Q-Q plots for 20 randomly selected nodes.
        - Histogram of p-values from the normality test.
        - Heatmap indicating which nodes pass or fail the normality test.
    """
    if gdf is None or gdf.empty:
        raise ValueError("Error: Input GeoDataFrame is empty or None.")
    
    if node_col not in gdf.columns or width_col not in gdf.columns or wse_col not in gdf.columns:
        raise ValueError("Error: One or more required columns are missing in the GeoDataFrame.")

    results = []
    min_samples = 3  # Minimum observations required for valid regression

    for node, group in gdf.groupby(node_col):
        if len(group) >= min_samples:
            X = group[[width_col]].values.reshape(-1, 1)
            y = group[wse_col].values
            huber = HuberRegressor()
            huber.fit(X, y)
            residuals = y - huber.predict(X)
            stat, p_value = stats.shapiro(residuals)
            normal_flag = p_value > alpha
        else:
            p_value, normal_flag = np.nan, False  # Not enough data

        results.append({node_col: node, 'p_value_norm': p_value, 'normal_flag': normal_flag})

    normality_df = pd.DataFrame(results)

    if normality_df.empty:
        raise ValueError("No valid normality results generated.")

    return gdf.merge(normality_df, on=node_col, how='left')


# Function to plot Q-Q plots to inspect normality

def plot_qq_plots(gdf, node_col, width_col, wse_col):
    '''_summary_
    This script generates Q-Q plots to visually inspect the normality of residuals for each node in a GeoDataFrame. 
    Args:
        gdf (GeoDataFrame): Input geospatial dataframe containing node data.
        node_col (str): Column name representing unique node identifiers.
        width_col (str): Column name containing width value.
        wse_col (str): Column name containing water surface elevation (WSE) values.
    Returns:
        - Q-Q plots for 20 randomly selected nodes.
    '''
    # Check if the input GeoDataFrame is empty or None
    random_nodes = random.sample(gdf[node_col].unique().tolist(), min(20, len(gdf[node_col].unique())))
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot Q-Q plots for each node
    for i, node in enumerate(random_nodes):
        subset = gdf[gdf[node_col] == node]
        if len(subset) > 3:
            X = subset[[width_col]].values.reshape(-1, 1)
            y = subset[wse_col].values
            huber = HuberRegressor()
            huber.fit(X, y)
            residuals = y - huber.predict(X)
            res = stats.probplot(residuals, dist="norm")
            axes[i].scatter(res[0][0], res[0][1], s=12, color='firebrick', alpha=0.9)  # Adjust dot size and color
            axes[i].plot(res[0][0], res[1][0] * res[0][0] + res[1][1], color='darkgray', linestyle='-', linewidth=1.5)
            
            # Get normality test result
            p_value = subset['p_value_norm'].iloc[0]
            passed = subset['normal_flag'].iloc[0]
            status = 'Pass' if passed else 'Fail'
            color = '#0072B2' if passed else '#E69F00'
            
            axes[i].set_title(f'Node {node}\n{status}', fontsize=10, color=color)
            axes[i].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Function to plot histogram of p-values to inspect normality

def plot_p_value_histogram(gdf):
    plt.figure(figsize=(10, 6))
    sns.histplot(gdf.drop_duplicates(subset=['node_id'])['p_value_norm'].dropna(), bins=25, kde=True, color='lightsteelblue', edgecolor='black', alpha=0.8)
    plt.axvline(0.05, color='red', linestyle='--', label='Alpha = 0.05')
    plt.xlabel('p-value', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.title('Histogram of Normality Test p-values', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Compute pass/fail statistics
    unique_nodes = gdf.drop_duplicates(subset=['node_id'])
    total_nodes = len(unique_nodes)
    passing_nodes = unique_nodes['normal_flag'].sum()
    failing_nodes = total_nodes - passing_nodes
    pass_percentage = (passing_nodes / total_nodes) * 100 if total_nodes > 0 else 0
    fail_percentage = (failing_nodes / total_nodes) * 100 if total_nodes > 0 else 0
    
    # Inset with pass/fail summary (better positioning and proper categorical handling)
    inset_ax = plt.gca().inset_axes([0.62, 0.55, 0.3, 0.3])  # Adjusted position to fit well within the figure
    categories = ['Pass', 'Fail']
    values = [passing_nodes, failing_nodes]
    inset_ax.bar(range(len(categories)), values, color=["#56B4E9", "#E69F00"], edgecolor='black', alpha=0.9)
    inset_ax.set_xticks(range(len(categories)))
    inset_ax.set_xticklabels(categories)
    inset_ax.set_title('Test Results', fontsize=10, fontweight='bold')
    inset_ax.set_ylabel('Nodes', fontsize=9)
    inset_ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Add text labels inside bars
    for i, v in enumerate(values):
        inset_ax.text(i, v / 2, f'{(v / total_nodes) * 100:.1f}%', ha='center', color='black', fontsize=9, weight='bold')
    
    plt.show()

# Function to plot heatmap of normality test results

def plot_heatmap(gdf, node_col, dist_col='p_dist_out'):
    # Compute pass/fail statistics
    unique_nodes = gdf.drop_duplicates(subset=[node_col]).copy()
    total_nodes = len(unique_nodes)
    unique_nodes['pass_fail'] = unique_nodes['p_value_norm'] >= 0.05  # True = Pass, False = Fail
    passing_nodes = unique_nodes['pass_fail'].sum()
    failing_nodes = total_nodes - passing_nodes
    pass_percentage = (passing_nodes / total_nodes) * 100 if total_nodes > 0 else 0
    fail_percentage = (failing_nodes / total_nodes) * 100 if total_nodes > 0 else 0
    
    # Sort nodes by distance from outlet in descending order (upstream first)
    unique_nodes = unique_nodes.sort_values(by=dist_col, ascending=False)
    
    # Reshape data to binary pass/fail heatmap
    summary_table = unique_nodes.pivot_table(index=node_col, values='pass_fail')
    summary_table = summary_table.reindex(unique_nodes[node_col].values)  # Ensure the order is maintained
    
    plt.figure(figsize=(15, 1.8))
    cmap = sns.color_palette(["#E69F00","#0072B2"])  # Blue for pass, Orange for fail
    ax = sns.heatmap(summary_table.T, cmap=cmap, cbar=True, linewidths=0.7, vmin=0, vmax=1, cbar_kws={'ticks': [0, 1]})
    
    # Modify color bar labels
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])  # Adjust tick positions for visibility
    colorbar.set_ticklabels(["Fail (<0.05)", "Pass (≥0.05)"])
    
    # Add arrow to indicate decreasing distance downstream
    plt.xlabel('Nodes (Ordered by Distance to Outlet →)')
    plt.annotate('Outlet', xy=(0.98, -0.02), xycoords='axes fraction', fontsize=8, color='black', ha='left', va='top', fontweight='bold')
    
    plt.title(f'Binary Heatmap of Normality Test Results (Sorted by Distance to Outlet)\nPass: {passing_nodes} ({pass_percentage:.1f}%) | Fail: {failing_nodes} ({fail_percentage:.1f}%)')
    plt.xticks([])
    plt.show()

# Function to perform adaptive Huber regression. The regression could be simple linear or piecewise linear with one breakpoint, depending on the AIC value.
def compute_aic(rss, n, k):
    # AIC = n*ln(RSS/n) + 2k
    return n * np.log(rss / n) + 2 * k

def adaptive_huber_fit(X, y, is_normal):
    """
    Fit a HuberRegressor with an adaptive epsilon computed from an initial fit.
    
    Parameters:
      - X: design matrix.
      - y: target values.
      - is_normal: Boolean flag indicating if residuals are normally distributed.
      
    Returns:
      - huber: fitted HuberRegressor model.
      - y_pred: predictions from the adaptive model.
      - final_residuals: residuals from the adaptive fit.
      - epsilon: the computed epsilon value.
    """
    # Initial fit with a default epsilon (1.35) and limited iterations.
    temp_model = HuberRegressor(epsilon=1.35, max_iter=500)
    temp_model.fit(X, y)
    initial_y_pred = temp_model.predict(X)
    initial_residuals = y - initial_y_pred
    
    # Compute adaptive epsilon based on the distribution of residuals.
    if is_normal:
        mad = np.median(np.abs(initial_residuals - np.median(initial_residuals))) * 1.4826
        epsilon = max(1.35, min(mad * 1.5, 15.0))
    else:
        iqr = np.percentile(initial_residuals, 75) - np.percentile(initial_residuals, 25)
        epsilon = max(1.35, min(iqr / 1.349 * 2, 15.0))
    
    # If epsilon is too large, reduce it slightly.
    if epsilon > 8.0:
        epsilon *= 0.75  # Reduce by 25%
    
    # Refit Huber with adaptive epsilon.
    huber = HuberRegressor(epsilon=epsilon, max_iter=500)
    huber.fit(X, y)
    y_pred = huber.predict(X)
    final_residuals = y - y_pred
    return huber, y_pred, final_residuals, epsilon

def piecewise_linear_huber_2seg_aic(gdf, node_col='node_id', normal_col='normal_flag'):
    """
    For each node (group defined by node_col) in gdf, fit two candidate models using adaptive Huber regression:
      1. A simple linear regression (k=2).
      2. A one-breakpoint piecewise linear regression (k=3).
    
    Adaptive epsilon:
      - For each candidate, an initial fit is performed to compute residuals.
      - If the Boolean flag in normal_col is True, epsilon is computed from the MAD of the initial residuals.
      - Otherwise, epsilon is computed from the IQR.
      - In both cases, epsilon is constrained to be at least 1.35 and at most 15.0, and further reduced by 25% if above 8.0.
    
    Overfitting restrictions:
      - For nodes with 10 to 15 observations: each segment must have at least 3 observations.
      - For nodes with 16 or more observations: each segment must have at least 30% of observations.
    
    Additional restriction:
      - In a one-breakpoint candidate, if both effective slopes (for x ≤ breakpoint and x > breakpoint)
        lie in the range [0.9, 1] (both included), then that candidate is rejected so that the simple
        linear regression model is used instead.
    
    Results are stored in new columns:
      - 'model_type': 'simple' or 'one_breakpoint'
      - 'h0', 'h1', 'h2', 'h3': model parameters (for a simple regression, h2 and h3 will be 0)
      - 'best_breakpoint1': breakpoint (NaN if not used)
      - 'predicted_wse': model predictions
      - 'residuals': residuals
      - 'AIC': the AIC of the chosen model
      - Additionally, the effective slopes and intercepts for each segment are stored in:
            'final_intercept_1', 'final_slope_1',
            'final_intercept_2', 'final_slope_2',
            'final_intercept_3', 'final_slope_3'
    
    Parameters:
      - gdf: DataFrame with at least columns 'width' and 'wse', plus a Boolean column (default name 'normal_flag')
      - node_col: the column name used to group nodes
      - normal_col: the column name with Booleans indicating normality of residuals
    
    Returns:
      - gdf: DataFrame updated with the new regression results.
    """
    # Initialize new columns
    gdf['model_type'] = None  
    gdf['h0'] = np.nan
    gdf['h1'] = np.nan
    gdf['h2'] = np.nan
    gdf['h3'] = np.nan
    gdf['best_breakpoint1'] = np.nan
    gdf['predicted_wse'] = np.nan
    gdf['residuals'] = np.nan
    gdf['AIC'] = np.nan

    # New columns for effective slopes and intercepts
    gdf['final_intercept_1'] = np.nan
    gdf['final_slope_1'] = np.nan
    gdf['final_intercept_2'] = np.nan
    gdf['final_slope_2'] = np.nan
    gdf['final_intercept_3'] = np.nan
    gdf['final_slope_3'] = np.nan

    # Tolerance range for slopes that lie between 0.9 and 1.
    lower_bound = 0.9
    upper_bound = 1.0

    # Process each node group.
    for node, group in gdf.groupby(node_col):
        x = group['width'].values
        y = group['wse'].values
        n = len(x)
        if n < 3:
            continue

        # The normality flag is taken from the group's first row.
        is_normal = group[normal_col].iloc[0]

        candidate_results = {}  # keys: 'simple', 'one_breakpoint'

        ########################
        # 1. Simple Linear Regression (k=2)
        ########################
        X_simple = x.reshape(-1, 1)
        try:
            model, y_pred_simple, residuals, eps_used = adaptive_huber_fit(X_simple, y, is_normal)
        except Exception:
            continue
        rss_simple = np.sum(residuals**2)
        aic_simple = compute_aic(rss_simple, n, 2)
        candidate_results['simple'] = {
            'AIC': aic_simple,
            'h0': model.intercept_,
            'h1': model.coef_[0],
            'h2': 0.0,
            'h3': 0.0,
            'bp1': np.nan,
            'pred': y_pred_simple,
            'model_type': 'simple'
        }

        ########################
        # 2. One-Breakpoint Piecewise (k=3)
        ########################
        best_aic_one = np.inf
        best_bp_one = None
        best_params_one = None
        best_pred_one = None
        candidate_found = False
        candidate_breakpoints = np.unique(x)[1:-1]  # exclude extremes

        for bp in candidate_breakpoints:
            left_count = np.sum(x <= bp)
            right_count = np.sum(x > bp)
            if 10 <= n <= 15:
                if left_count < 3 or right_count < 3:
                    continue
            elif n >= 16:
                if left_count < 0.3 * n or right_count < 0.3 * n:
                    continue

            X_candidate = np.column_stack([x, np.maximum(0, x - bp)])
            try:
                model, y_pred_candidate, residuals, eps_used = adaptive_huber_fit(X_candidate, y, is_normal)
            except Exception:
                continue

            h0_candidate = model.intercept_
            h1_candidate = model.coef_[0]
            # For x > bp, effective slope becomes h1_candidate + coef_[1]
            h2_candidate = h1_candidate + model.coef_[1]

            # Reject candidate if any effective slope is negative.
            if h1_candidate < 0 or h2_candidate < 0:
                continue

            # Additional restriction:
            # If both effective slopes lie in the range [0.9, 1] (inclusive), reject the piecewise candidate.
            if (lower_bound <= h1_candidate <= upper_bound) or (lower_bound <= h2_candidate <= upper_bound):
                continue

            # Check that the breakpoint produces a sufficient intercept difference.
            if abs(model.coef_[1] * bp) < 0.6:
                continue

            rss_candidate = np.sum(residuals**2)
            aic_candidate = compute_aic(rss_candidate, n, 3)

            if aic_candidate < best_aic_one:
                best_aic_one = aic_candidate
                best_bp_one = bp
                best_params_one = (h0_candidate, h1_candidate, model.coef_[1])
                best_pred_one = y_pred_candidate
                candidate_found = True

        if candidate_found:
            candidate_results['one_breakpoint'] = {
                'AIC': best_aic_one,
                'h0': best_params_one[0],
                'h1': best_params_one[1],
                'h2': best_params_one[1] + best_params_one[2],
                'h3': 0.0,
                'bp1': best_bp_one,
                'pred': best_pred_one,
                'model_type': 'one_breakpoint'
            }

        # Select best candidate based on AIC.
        best_model_type = None
        best_aic = np.inf
        for key, res in candidate_results.items():
            if res['AIC'] < best_aic:
                best_aic = res['AIC']
                best_model_type = key
        if best_model_type is None:
            best_model = candidate_results['simple']
        else:
            best_model = candidate_results[best_model_type]

        # Store chosen model results back into the DataFrame.
        idx = group.index
        gdf.loc[idx, 'model_type'] = best_model['model_type']
        gdf.loc[idx, 'h0'] = best_model['h0']
        gdf.loc[idx, 'h1'] = best_model['h1']
        gdf.loc[idx, 'h2'] = best_model['h2']
        gdf.loc[idx, 'h3'] = best_model.get('h3', 0.0)
        gdf.loc[idx, 'best_breakpoint1'] = best_model['bp1']
        gdf.loc[idx, 'predicted_wse'] = best_model['pred']
        gdf.loc[idx, 'residuals'] = y - best_model['pred']
        gdf.loc[idx, 'AIC'] = best_model['AIC']

        # Compute and store effective slopes and intercepts for the chosen model.
        if best_model['model_type'] == 'simple':
            # Simple model: one segment.
            gdf.loc[idx, 'final_intercept_1'] = best_model['h0']
            gdf.loc[idx, 'final_slope_1'] = best_model['h1']
            gdf.loc[idx, 'final_intercept_2'] = np.nan
            gdf.loc[idx, 'final_slope_2'] = np.nan
            gdf.loc[idx, 'final_intercept_3'] = np.nan
            gdf.loc[idx, 'final_slope_3'] = np.nan
        elif best_model['model_type'] == 'one_breakpoint':
            # One-breakpoint model: two segments.
            seg1_int = best_model['h0']
            seg1_slope = best_model['h1']
            seg2_int = best_model['h0'] - (best_model['h2'] - best_model['h1']) * best_model['bp1']
            seg2_slope = best_model['h2']
            gdf.loc[idx, 'final_intercept_1'] = seg1_int
            gdf.loc[idx, 'final_slope_1'] = seg1_slope
            gdf.loc[idx, 'final_intercept_2'] = seg2_int
            gdf.loc[idx, 'final_slope_2'] = seg2_slope
            gdf.loc[idx, 'final_intercept_3'] = np.nan
            gdf.loc[idx, 'final_slope_3'] = np.nan

    return gdf

# Function to plot the regression results for a specific node_id.

def plot_node_reg_2segs(gdf, node_id, node_col='node_id'):
    """
    Plots the observed data and fitted regression for a specific node_id.
    
    The function supports:
      - A simple model (one segment) where the fitted line is plotted over the entire range.
      - A one-breakpoint model (two segments) where:
          - Segment 1 is for x values <= breakpoint.
          - Segment 2 is for x values > breakpoint.
    
    Parameters:
      - gdf: DataFrame that includes the regression results from piecewise_linear_huber_aic.
      - node_id: the specific node_id to plot.
      - node_col: the column name used for node identification (default 'node_id').
    """
    # Filter data for the specified node_id and sort by 'width'.
    subset = gdf[gdf[node_col] == node_id].copy()
    if subset.empty:
        print(f"No data found for node_id: {node_id}")
        return
    subset.sort_values(by='width', inplace=True)
    
    x_obs = subset['width'].values
    y_obs = subset['wse'].values
    model_type = subset['model_type'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_obs, y_obs, color='blue', label='Observed data')
    
    # Plot according to the model type.
    if model_type == 'simple':
        seg1_int = subset['final_intercept_1'].iloc[0]
        seg1_slope = subset['final_slope_1'].iloc[0]
        x_line = np.linspace(np.min(x_obs), np.max(x_obs), 200)
        y_line = seg1_int + seg1_slope * x_line
        plt.plot(x_line, y_line, color='red',
                 label=f"Segment 1 (All data): intercept = {seg1_int:.2f}, slope = {seg1_slope:.2f}")
    
    elif model_type == 'one_breakpoint':
        bp = subset['best_breakpoint1'].iloc[0]
        seg1_int = subset['final_intercept_1'].iloc[0]
        seg1_slope = subset['final_slope_1'].iloc[0]
        seg2_int = subset['final_intercept_2'].iloc[0]
        seg2_slope = subset['final_slope_2'].iloc[0]
        
        # Create piecewise segments.
        x_line_seg1 = np.linspace(np.min(x_obs), bp, 100)
        y_line_seg1 = seg1_int + seg1_slope * x_line_seg1
        x_line_seg2 = np.linspace(bp, np.max(x_obs), 100)
        y_line_seg2 = seg2_int + seg2_slope * x_line_seg2
        
        plt.plot(x_line_seg1, y_line_seg1, color='red',
                 label=f"Segment 1 (x ≤ {bp:.2f}): intercept = {seg1_int:.2f}, slope = {seg1_slope:.2f}")
        plt.plot(x_line_seg2, y_line_seg2, color='green',
                 label=f"Segment 2 (x > {bp:.2f}): intercept = {seg2_int:.2f}, slope = {seg2_slope:.2f}")
        plt.axvline(x=bp, color='purple', linestyle='--', label=f"Breakpoint: {bp:.2f}")
    
    plt.xlabel("Width")
    plt.ylabel("WSE")
    plt.title(f"Regression for node_id: {node_id} ({model_type})")
    plt.legend()
    plt.show()

# Function to plot regressions for multiple randomly selected nodes.
def plot_random_nodes_Hub(gdf, num_nodes=10, node_col='node_id'):
    """
    Randomly selects num_nodes unique node IDs from the GeoDataFrame and plots
    their piecewise regressions in a grid of subplots.
    
    Parameters:
      - gdf: GeoDataFrame with regression results. It must contain at least the columns
             'width', 'wse', 'model_type', 'best_breakpoint1', and (if applicable) 
             'best_breakpoint2', as well as final effective parameters.
      - num_nodes: number of nodes to plot (default 20).
      - node_col: column name for the node identifier.
    """
    # Get unique node IDs.
    unique_nodes = gdf[node_col].unique()
    if len(unique_nodes) < num_nodes:
        num_nodes = len(unique_nodes)
    selected_nodes = np.random.choice(unique_nodes, size=num_nodes, replace=False)
    
    # Create a grid of subplots.
    ncols = 5
    nrows = int(np.ceil(num_nodes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    
    # Define colors for segments.
    colors = {'seg1': 'red', 'seg2': 'green', 'seg3': 'blue'}
    
    for ax, node in zip(axes, selected_nodes):
        # Filter and sort the data for the current node.
        subset = gdf[gdf[node_col] == node].copy()
        if subset.empty:
            continue
        subset.sort_values(by='width', inplace=True)
        x_obs = subset['width'].values
        y_obs = subset['wse'].values
        
        # Determine model type and breakpoints.
        model_type = subset['model_type'].iloc[0]
        bp1 = subset['best_breakpoint1'].iloc[0] if 'best_breakpoint1' in subset.columns else np.nan
        bp2 = subset['best_breakpoint2'].iloc[0] if 'best_breakpoint2' in subset.columns else np.nan
        
        # Plot the observed data.
        ax.scatter(x_obs, y_obs, color='blue', label='Observed data')
        x_min, x_max = np.min(x_obs), np.max(x_obs)
        
        if model_type == 'simple':
            seg1_int = subset['final_intercept_1'].iloc[0]
            seg1_slope = subset['final_slope_1'].iloc[0]
            x_line = np.linspace(x_min, x_max, 200)
            y_line = seg1_int + seg1_slope * x_line
            ax.plot(x_line, y_line, color=colors['seg1'],
                    label=f'Seg1 (All): int = {seg1_int:.2f}, slope = {seg1_slope:.2f}')
        
        elif model_type == 'one_breakpoint':
            seg1_int = subset['final_intercept_1'].iloc[0]
            seg1_slope = subset['final_slope_1'].iloc[0]
            seg2_int = subset['final_intercept_2'].iloc[0]
            seg2_slope = subset['final_slope_2'].iloc[0]
            
            x_line_seg1 = np.linspace(x_min, bp1, 100)
            x_line_seg2 = np.linspace(bp1, x_max, 100)
            y_line_seg1 = seg1_int + seg1_slope * x_line_seg1
            y_line_seg2 = seg2_int + seg2_slope * x_line_seg2
            
            ax.plot(x_line_seg1, y_line_seg1, color=colors['seg1'],
                    label=f'Seg1 (x ≤ {bp1:.2f}): int = {seg1_int:.2f}, slope = {seg1_slope:.2f}')
            ax.plot(x_line_seg2, y_line_seg2, color=colors['seg2'],
                    label=f'Seg2 (x > {bp1:.2f}): int = {seg2_int:.2f}, slope = {seg2_slope:.2f}')
            ax.axvline(x=bp1, color='purple', linestyle='--', label=f'BP: {bp1:.2f}')
        
        elif model_type == 'two_breakpoints':
            seg1_int = subset['final_intercept_1'].iloc[0]
            seg1_slope = subset['final_slope_1'].iloc[0]
            seg2_int = subset['final_intercept_2'].iloc[0]
            seg2_slope = subset['final_slope_2'].iloc[0]
            seg3_int = (subset['final_intercept_3'].iloc[0]
                        if 'final_intercept_3' in subset.columns else np.nan)
            seg3_slope = (subset['final_slope_3'].iloc[0]
                          if 'final_slope_3' in subset.columns else np.nan)
            
            x_line_seg1 = np.linspace(x_min, bp1, 100)
            x_line_seg2 = np.linspace(bp1, bp2, 100)
            x_line_seg3 = np.linspace(bp2, x_max, 100)
            y_line_seg1 = seg1_int + seg1_slope * x_line_seg1
            y_line_seg2 = seg2_int + seg2_slope * x_line_seg2
            if not np.isnan(seg3_int) and not np.isnan(seg3_slope):
                y_line_seg3 = seg3_int + seg3_slope * x_line_seg3
            else:
                y_line_seg3 = None
            
            ax.plot(x_line_seg1, y_line_seg1, color=colors['seg1'],
                    label=f'Seg1 (x ≤ {bp1:.2f}): int = {seg1_int:.2f}, slope = {seg1_slope:.2f}')
            ax.plot(x_line_seg2, y_line_seg2, color=colors['seg2'],
                    label=f'Seg2 ({bp1:.2f} < x ≤ {bp2:.2f}): int = {seg2_int:.2f}, slope = {seg2_slope:.2f}')
            if y_line_seg3 is not None:
                ax.plot(x_line_seg3, y_line_seg3, color=colors['seg3'],
                        label=f'Seg3 (x > {bp2:.2f}): int = {seg3_int:.2f}, slope = {seg3_slope:.2f}')
            ax.axvline(x=bp1, color='black', linestyle='--', label=f'BP1: {bp1:.2f}')
            ax.axvline(x=bp2, color='gray', linestyle='--', label=f'BP2: {bp2:.2f}')
        
        ax.set_xlabel("Width")
        ax.set_ylabel("WSE")
        ax.set_title(f"node_id: {node} ({model_type})")
        ax.legend(fontsize='x-small', loc='best')
    
    # Hide any unused subplots.
    for i in range(num_nodes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()