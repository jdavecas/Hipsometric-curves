import random
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.optimize import curve_fit
from scipy.optimize import minimize


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


def l_regression_node(
    river, 
    node_id=None, 
    min_spearman=None, 
    min_obs=0, 
    show_p_value=True, 
    min_p_value=0.05
):
    """
    Generates a hypsometric scatter plot for a specific river node with linear regression, 
    or multiple plots if no node_id is provided. Adds a residual plot to the right side of the scatter plot.

    Args:
        river (dict): Dictionary containing node data with 'width' and 'wse' keys.
        node_id (str or None): Specific node to plot. If None, plots nodes based on 
                            random selection or threshold criteria.
        min_spearman (float or None): Minimum Spearman correlation value to include a node in the plot.
                                    If None, plots are divided equally above and below a Spearman value of 0.4.
        min_obs (int): Minimum number of observations required to display a scatter plot for a node (default: 10).
        show_p_value (bool): If True, displays the p-value on the scatter plot (default: True).
        min_p_value (float): Minimum p-value required to include a node in the plot (default: 0.05).
    """
    import scipy.stats
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import random

    # Validate node_id, if provided
    if node_id:
        if node_id not in river:
            raise ValueError(f"Node ID '{node_id}' not found in the dataset.")

    # Prepare list to store statistics for all nodes
    results = []

    # Function to plot a single node
    def plot_node(node_id, node_data):
        # Perform linear regression
        widths = np.array(node_data['width']).reshape(-1, 1)
        wses = np.array(node_data['wse'])
        reg = LinearRegression().fit(widths, wses)
        slope = round(reg.coef_[0], 3)
        intercept = round(reg.intercept_, 3)
        r_squared = round(reg.score(widths, wses), 3)

        # Calculate residuals
        residuals = wses - reg.predict(widths)

        # Calculate Spearman correlation and p-value
        spearman_corr, p_value = scipy.stats.spearmanr(node_data['width'], node_data['wse'])

        # Create the figure with two subplots: scatter plot and residual plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Scatter plot with regression line
        axes[0].scatter(node_data['width'], node_data['wse'], alpha=0.8, color="blue", edgecolor="white", label="Data Points")
        x_range = np.linspace(widths.min(), widths.max(), 100)
        y_range = slope * x_range + intercept
        axes[0].plot(x_range, y_range, color="red", linestyle="--", linewidth=2, label="OLS Regression Line")
        axes[0].set_title(f"Node ID: {node_id}\nHuber Regression\nR²: {r_squared}, Slope: {slope}, Intercept: {intercept}")
        axes[0].set_xlabel("Width")
        axes[0].set_ylabel("WSE")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Residual plot on the right side
        axes[1].scatter(node_data['width'], residuals, alpha=0.8, color="green", edgecolor="black", label="Residuals")
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_title("Residuals")
        axes[1].set_xlabel("Width")
        axes[1].set_ylabel("Residuals")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Display p-value, slope, and intercept if show_p_value is True
        if show_p_value:
            axes[0].text(0.05, 0.85, f"p-value: {p_value:.3f}", ha='left', va='center', transform=axes[0].transAxes, fontsize=10, color="gray")
        axes[0].text(0.05, 0.75, f"Slope: {slope}\nIntercept: {intercept}", ha='left', va='center', transform=axes[0].transAxes, fontsize=10, color="gray")

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

        return {
            'Node': node_id,
            'Spearman': round(spearman_corr, 3),
            'p_value': round(p_value, 3),
            'R2': round(r_squared, 3),
            'Slope': slope,
            'Intercept': intercept
        }

    if node_id:
        # Plot for the specific node_id
        node_data = river[node_id]
        if len(node_data['width']) < min_obs:
            raise ValueError(f"Node {node_id} does not have enough observations (min_obs={min_obs}).")
        if len(node_data['width']) >= 2:
            return pd.DataFrame([plot_node(node_id, node_data)])
    else:
        # Original behavior: filter and plot multiple nodes
        above_threshold = []
        below_threshold = []

        for nid, node_data in river.items():
            if len(node_data['width']) < min_obs:
                continue
            spearman_corr, p_value = scipy.stats.spearmanr(node_data['width'], node_data['wse'])
            if p_value >= min_p_value:
                continue
            if spearman_corr >= 0.4:
                above_threshold.append(nid)
            else:
                below_threshold.append(nid)

        # Randomly select nodes for plotting
        num_above = min(len(above_threshold), 10)
        num_below = min(len(below_threshold), 10)
        random_nodes = random.sample(above_threshold, num_above) + random.sample(below_threshold, num_below)

        for nid in random_nodes:
            results.append(plot_node(nid, river[nid]))

        return pd.DataFrame(results).round(3)



# Huber regression with fixed epsilon=1.345, and using AIC (Akaike Information Criterion) for simple or piecewise linear regression
def compute_aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k

def fixed_huber_fit(X, y):
    model = HuberRegressor(epsilon=1.345, max_iter=500)
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    return model, y_pred, residuals

def fixed_huber_piecewise_aic(river_dict):
    """
    Apply fixed-epsilon Huber regression to a dictionary of river nodes.
    Returns a DataFrame summarizing model type and parameters per node.
    """
    results = []

    for node_id, node_data in river_dict.items():
        x = np.array(node_data['width'])
        y = np.array(node_data['wse'])
        n = len(x)

        if n < 3:
            continue

        node_result = {
            'node_id': node_id,  # <- Renamed here
            'model_type': None,
            'h0': np.nan, 'h1': np.nan, 'h2': np.nan, 'h3': np.nan,
            'breakpoint': np.nan,
            'AIC': np.nan,
            'final_intercept_1': np.nan, 'final_slope_1': np.nan,
            'final_intercept_2': np.nan, 'final_slope_2': np.nan
        }

        # 1. Simple Linear Regression
        X_simple = x.reshape(-1, 1)
        try:
            model_simple, y_pred_simple, residuals_simple = fixed_huber_fit(X_simple, y)
        except Exception:
            continue

        rss_simple = np.sum(residuals_simple ** 2)
        aic_simple = compute_aic(rss_simple, n, 2)

        best_model = {
            'type': 'simple',
            'AIC': aic_simple,
            'model': model_simple,
            'pred': y_pred_simple,
            'breakpoint': None,
            'h0': model_simple.intercept_,
            'h1': model_simple.coef_[0],
            'h2': 0,
            'h3': 0
        }

        # 2. One-Breakpoint Piecewise Linear Regression
        candidate_breakpoints = np.unique(x)[1:-1]
        for bp in candidate_breakpoints:
            left_count = np.sum(x <= bp)
            right_count = np.sum(x > bp)

            if 10 <= n <= 15 and (left_count < 5 or right_count < 5):
                continue
            elif n >= 16 and (left_count < 0.3 * n or right_count < 0.3 * n):
                continue

            X_piecewise = np.column_stack([x, np.maximum(0, x - bp)])

            try:
                model_pw, y_pred_pw, residuals_pw = fixed_huber_fit(X_piecewise, y)
            except Exception:
                continue

            h0 = model_pw.intercept_
            h1 = model_pw.coef_[0]
            delta = model_pw.coef_[1]
            h2 = h1 + delta

            # reject if slope change is too small
            if abs(h2 - h1) < 0.08:
                continue
            if h1 < 0 or h2 < 0:
                continue
            if 0.9 <= h1 <= 1.0 and 0.9 <= h2 <= 1.0:
                continue
            if abs(delta * bp) < 0.6:
                continue

            rss_pw = np.sum(residuals_pw ** 2)
            aic_pw = compute_aic(rss_pw, n, 3)

            if aic_pw < best_model['AIC']:
                best_model = {
                    'type': 'one_breakpoint',
                    'AIC': aic_pw,
                    'model': model_pw,
                    'pred': y_pred_pw,
                    'breakpoint': bp,
                    'h0': h0,
                    'h1': h1,
                    'h2': h2,
                    'h3': delta
                }

        # Fill in results
        node_result.update({
            'model_type': best_model['type'],
            'h0': best_model['h0'],
            'h1': best_model['h1'],
            'h2': best_model['h2'],
            'h3': best_model['h3'],
            'breakpoint': best_model['breakpoint'],
            'AIC': best_model['AIC'],
            'initial_slope_2': best_model['h1']
        })

        if best_model['type'] == 'simple':
            node_result['final_intercept_1'] = best_model['h0']
            node_result['final_slope_1'] = best_model['h1']
        else:
            seg1_int = best_model['h0']
            seg1_slope = best_model['h1']
            seg2_slope = best_model['h2']
            seg2_int = best_model['h0'] - (best_model['h2'] - best_model['h1']) * best_model['breakpoint']
            node_result['final_intercept_1'] = seg1_int
            node_result['final_slope_1'] = seg1_slope
            node_result['final_intercept_2'] = seg2_int
            node_result['final_slope_2'] = seg2_slope

        results.append(node_result)

    return pd.DataFrame(results)

def plot_node_reg_2segs_fixed(gdf, node_id, node_col='node_id'):
    """
    Plots the observed data and fitted regression for a specific node_id.

    Works with the output of the fixed_huber_piecewise_aic() function.

    Parameters:
    - gdf: Merged GeoDataFrame or DataFrame that includes both the original observations
            (columns: 'width', 'wse') and the regression results.
    - node_id: The specific node_id to plot.
    - node_col: The column name used for node identification (default 'node_id').
    """
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

    if model_type == 'simple':
        seg1_int = subset['final_intercept_1'].iloc[0]
        seg1_slope = subset['final_slope_1'].iloc[0]
        x_line = np.linspace(np.min(x_obs), np.max(x_obs), 200)
        y_line = seg1_int + seg1_slope * x_line
        plt.plot(x_line, y_line, color='red',
                label=f"Simple model: intercept = {seg1_int:.2f}, slope = {seg1_slope:.2f}")

    elif model_type == 'one_breakpoint':
        bp = subset['breakpoint'].iloc[0]
        seg1_int = subset['final_intercept_1'].iloc[0]
        seg1_slope = subset['final_slope_1'].iloc[0]
        seg2_int = subset['final_intercept_2'].iloc[0]
        seg2_slope = subset['final_slope_2'].iloc[0]

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
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Huber adaptative regression with AIC
def compute_aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k

def adaptive_huber_fit(X, y):
    """
    Fit a Huber regression with an adaptive epsilon.
    The epsilon is chosen based on the residual distribution from an OLS fit:
      - If the proportion of residuals beyond 2*std is ≤ 5% (nearly normal),
        then epsilon is set to 1.5 so that Huber behaves like OLS.
      - If the proportion is ≥ 15%, then epsilon is set to 1 (maximally robust).
      - For intermediate cases, epsilon is linearly interpolated between 1.5 and 1.
    """
    # Initial OLS fit to assess residual distribution
    ols = LinearRegression()
    ols.fit(X, y)
    y_pred_ols = ols.predict(X)
    residuals_ols = y - y_pred_ols
    std_res = np.std(residuals_ols, ddof=1)
    
    # Compute proportion of residuals that are "extreme"
    p_outliers = np.mean(np.abs(residuals_ols) > 2 * std_res)
    
    # Adaptive epsilon between 1.5 (OLS-like) and 1 (robust)
    if p_outliers <= 0.05:
        eps = 1.5
    elif p_outliers >= 0.15:
        eps = 1.0
    else:
        # Linear interpolation between 1.5 and 1.0 for 0.05 < p_outliers < 0.15
        eps = 1.5 - ((p_outliers - 0.05) / (0.15 - 0.05)) * 0.5
        
    # Fit the HuberRegressor with the computed epsilon
    model = HuberRegressor(epsilon=eps, max_iter=500)
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    return model, y_pred, residuals

def adaptive_huber_piecewise_aic(river_dict):
    """
    Apply adaptive-epsilon Huber regression to a dictionary of river nodes.
    Returns a DataFrame summarizing model type and parameters per node.
    """
    results = []

    for node_id, node_data in river_dict.items():
        x = np.array(node_data['width'])
        y = np.array(node_data['wse'])
        n = len(x)

        if n < 3:
            continue

        node_result = {
            'node_id': node_id,  # Node identifier
            'model_type': None,
            'h0': np.nan, 'h1': np.nan, 'h2': np.nan, 'h3': np.nan,
            'breakpoint': np.nan,
            'AIC': np.nan,
            'final_intercept_1': np.nan, 'final_slope_1': np.nan,
            'final_intercept_2': np.nan, 'final_slope_2': np.nan
        }

        # 1. Simple Linear Regression with adaptive Huber
        X_simple = x.reshape(-1, 1)
        try:
            model_simple, y_pred_simple, residuals_simple = adaptive_huber_fit(X_simple, y)
        except Exception:
            continue

        rss_simple = np.sum(residuals_simple ** 2)
        aic_simple = compute_aic(rss_simple, n, 2)

        best_model = {
            'type': 'simple',
            'AIC': aic_simple,
            'model': model_simple,
            'pred': y_pred_simple,
            'breakpoint': None,
            'h0': model_simple.intercept_,
            'h1': model_simple.coef_[0],
            'h2': 0,
            'h3': 0
        }

        # 2. One-Breakpoint Piecewise Linear Regression
        candidate_breakpoints = np.unique(x)[1:-1]
        for bp in candidate_breakpoints:
            left_count = np.sum(x <= bp)
            right_count = np.sum(x > bp)

            if 10 <= n <= 15 and (left_count < 5 or right_count < 5):
                continue
            elif n >= 16 and (left_count < 0.3 * n or right_count < 0.3 * n):
                continue

            X_piecewise = np.column_stack([x, np.maximum(0, x - bp)])

            try:
                model_pw, y_pred_pw, residuals_pw = adaptive_huber_fit(X_piecewise, y)
            except Exception:
                continue

            h0 = model_pw.intercept_
            h1 = model_pw.coef_[0]
            delta = model_pw.coef_[1]
            h2 = h1 + delta

            # Reject models with small slope changes or unrealistic slopes
            if abs(h2 - h1) < 0.08:
                continue
            if h1 < 0 or h2 < 0:
                continue
            if 0.9 <= h1 <= 1.0 and 0.9 <= h2 <= 1.0:
                continue
            if abs(delta * bp) < 0.6:
                continue

            rss_pw = np.sum(residuals_pw ** 2)
            aic_pw = compute_aic(rss_pw, n, 3)

            if aic_pw < best_model['AIC']:
                best_model = {
                    'type': 'one_breakpoint',
                    'AIC': aic_pw,
                    'model': model_pw,
                    'pred': y_pred_pw,
                    'breakpoint': bp,
                    'h0': h0,
                    'h1': h1,
                    'h2': h2,
                    'h3': delta
                }

        # Fill in results for the node
        node_result.update({
            'model_type': best_model['type'],
            'h0': best_model['h0'],
            'h1': best_model['h1'],
            'h2': best_model['h2'],
            'h3': best_model['h3'],
            'breakpoint': best_model['breakpoint'],
            'AIC': best_model['AIC'],
            'initial_slope_2': best_model['h1']
        })

        if best_model['type'] == 'simple':
            node_result['final_intercept_1'] = best_model['h0']
            node_result['final_slope_1'] = best_model['h1']
        else:
            seg1_int = best_model['h0']
            seg1_slope = best_model['h1']
            seg2_slope = best_model['h2']
            seg2_int = best_model['h0'] - (best_model['h2'] - best_model['h1']) * best_model['breakpoint']
            node_result['final_intercept_1'] = seg1_int
            node_result['final_slope_1'] = seg1_slope
            node_result['final_intercept_2'] = seg2_int
            node_result['final_slope_2'] = seg2_slope

        results.append(node_result)

    return pd.DataFrame(results)