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


def huber_regression(river, node_id, epsilon=1.35, min_obs=10):
    """
    Performs Huber regression for a specific node in the river dataset, 
    handling outliers and reporting robust statistics.
    
    Args:
        river (dict): Dictionary containing node data with 'width' and 'wse' keys.
        node_id (str): ID of the node to analyze.
        epsilon (float): Parameter controlling the threshold for outlier influence in Huber regression (default: 1.35).
        min_obs (int): Minimum number of observations required to perform regression (default: 10).
    
    Returns:
        dict: A dictionary with regression statistics including Spearman and Pearson correlations.
    """
    # Check if node_id exists
    if node_id not in river:
        raise ValueError(f"Node ID '{node_id}' not found in the dataset.")
    
    node_data = river[node_id]
    widths = np.array(node_data['width'])
    wses = np.array(node_data['wse'])
    
    # Ensure sufficient observations
    if len(widths) < min_obs:
        raise ValueError(f"Node ID '{node_id}' has fewer than {min_obs} observations.")
    
    # Perform Huber regression
    model = HuberRegressor(epsilon=epsilon).fit(widths.reshape(-1, 1), wses)
    slope = round(model.coef_[0], 3)
    intercept = round(model.intercept_, 3)
    r_squared = round(model.score(widths.reshape(-1, 1), wses), 3)
    
    # Calculate correlations
    spearman_corr, spearman_p_value = scipy.stats.spearmanr(widths, wses)
    pearson_corr, pearson_p_value = scipy.stats.pearsonr(widths, wses)
    
    # Compute residuals
    predicted_wse = model.predict(widths.reshape(-1, 1))
    residuals = wses - predicted_wse

    # Plot scatter plot and regression line, and residuals on the right side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # Scatter plot with regression line
    axes[0].scatter(widths, wses, alpha=0.8, color="blue", edgecolor="white", label="Data Points")
    x_range = np.linspace(widths.min(), widths.max(), 100)
    y_range = slope * x_range + intercept
    axes[0].plot(x_range, y_range, color="red", linestyle="--", linewidth=2, label="Huber Regression Line")
    axes[0].set_title(f"Node ID: {node_id}\nHuber Regression\nR²: {r_squared}, Slope: {slope}, Intercept: {intercept}")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("WSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot residuals on the right side
    axes[1].scatter(widths, residuals, alpha=0.8, color="green", edgecolor="black", label="Residuals")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Residuals")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Return results
    report = {
        "Node": node_id,
        "Spearman_Correlation": round(spearman_corr, 3),
        "Spearman_p_value": round(spearman_p_value, 3),
        "Pearson_Correlation": round(pearson_corr, 3),
        "Pearson_p_value": round(pearson_p_value, 3),
        "Slope": slope,
        "Intercept": intercept,
        "R_squared": r_squared,
    }
    
    return report

def pw_l_regression_node(
    river, 
    node_id=None, 
    min_spearman=None, 
    min_obs=0, 
    show_p_value=True, 
    min_p_value=0.05
):
    """
    Generates a hypsometric scatter plot with piece-wise linear regression for a specific river node,
    or multiple plots if no node_id is provided. Includes a residuals plot.

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

    def piecewise_fit(x, y):
        """Find the optimal breakpoint for piecewise linear regression."""
        def piecewise_model(params):
            # Breakpoint, slope1, slope2, intercept
            breakpoint, slope1, slope2, intercept = params
            y_pred = np.where(
                x <= breakpoint,
                slope1 * x + intercept,
                slope2 * (x - breakpoint) + (slope1 * breakpoint + intercept)
            )
            return np.sum((y - y_pred) ** 2)

        # Initial guesses: midpoint as breakpoint, slopes, and intercept
        init_params = [
            np.median(x),  # Breakpoint
            (y[-1] - y[0]) / (x[-1] - x[0]),  # Slope1
            (y[-1] - y[0]) / (x[-1] - x[0]),  # Slope2
            np.mean(y)  # Intercept
        ]
        bounds = [
            (x.min(), x.max()),  # Breakpoint
            (-np.inf, np.inf),   # Slope1
            (-np.inf, np.inf),   # Slope2
            (-np.inf, np.inf)    # Intercept
        ]
        result = minimize(piecewise_model, init_params, bounds=bounds)
        return result.x  # Optimal parameters

    def plot_node(node_id, node_data):
        # Data preparation
        width = np.array(node_data['width'])
        wse = np.array(node_data['wse'])

        # Fit piecewise model
        breakpoint, slope1, slope2, intercept = piecewise_fit(width, wse)

        # Calculate R²
        segment1 = slope1 * width[width <= breakpoint] + intercept
        segment2 = slope2 * (width[width > breakpoint] - breakpoint) + (slope1 * breakpoint + intercept)
        y_pred = np.concatenate([segment1, segment2])
        r2 = 1 - np.sum((wse - y_pred) ** 2) / np.sum((wse - np.mean(wse)) ** 2)

        # Residuals
        residuals = wse - y_pred

        # Spearman correlation
        spearman_corr, p_value = scipy.stats.spearmanr(width, wse)

        # Plot scatter and residuals
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # Scatter plot with regression
        ax1 = axes[0]
        ax1.scatter(width, wse, alpha=0.8, c="darkcyan", edgecolors='cyan', linewidths=1)
        x_segment1 = np.linspace(width.min(), breakpoint, 100)
        x_segment2 = np.linspace(breakpoint, width.max(), 100)
        y_segment1 = slope1 * x_segment1 + intercept
        y_segment2 = slope2 * (x_segment2 - breakpoint) + (slope1 * breakpoint + intercept)
        ax1.plot(x_segment1, y_segment1, color="blue", label=f"Segment 1: y = {slope1:.3f}x + {intercept:.3f}")
        ax1.plot(x_segment2, y_segment2, color="orange", label=f"Segment 2: y = {slope2:.3f}(x-{breakpoint:.3f}) + {slope1 * breakpoint + intercept:.3f}")
        ax1.set_title(f"Node: {node_id}\nR²: {r2:.2f}, Spearman: {spearman_corr:.2f}", fontsize=12)
        ax1.set_xlabel("Width")
        ax1.set_ylabel("WSE")
        ax1.legend()
        if show_p_value:
            ax1.text(0.05, 0.85, f"p-value: {p_value:.3f}", ha='left', va='center', transform=ax1.transAxes, fontsize=10, color="gray")
        ax1.grid(alpha=0.3)

        # Residuals plot
        ax2 = axes[1]
        ax2.scatter(width, residuals, alpha=0.8, c="darkred", edgecolors='white', linewidths=1)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_title("Residuals")
        ax2.set_xlabel("Width")
        ax2.set_ylabel("Residuals")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {
            'Node': node_id,
            'Breakpoint': round(breakpoint, 3),
            'Slope1': round(slope1, 3),
            'Slope2': round(slope2, 3),
            'Intercept': round(intercept, 3),
            'R2': round(r2, 3),
            'Spearman': round(spearman_corr, 3),
            'p_value': round(p_value, 3)
        }

    results = []
    if node_id:
        node_data = river[node_id]
        if len(node_data['width']) < min_obs:
            raise ValueError(f"Node {node_id} does not have enough observations (min_obs={min_obs}).")
        results.append(plot_node(node_id, node_data))
    else:
        for nid, node_data in river.items():
            if len(node_data['width']) < min_obs:
                continue
            results.append(plot_node(nid, node_data))
    return pd.DataFrame(results).round(3)


def pw_l_regression_huber_node(
    river, 
    node_id=None, 
    min_spearman=None, 
    min_obs=1, 
    show_p_value=True, 
    min_p_value=None,
    delta=1.345  # Default delta for Huber loss
):
    """
    Generates a hypsometric scatter plot with piece-wise linear regression for a specific river node,
    or multiple plots if no node_id is provided. Also includes a residuals plot.
    """

    def huber_loss(residuals, delta):
        """Huber loss function."""
        abs_residuals = np.abs(residuals)
        return np.where(
            abs_residuals <= delta,
            0.5 * residuals**2,
            delta * (abs_residuals - 0.5 * delta)
        )
    
    def piecewise_huber_loss(params, x, y, delta):
        """Piecewise linear model with Huber loss."""
        breakpoint, slope1, slope2, intercept = params
        y_pred = np.where(
            x <= breakpoint,
            slope1 * x + intercept,
            slope2 * (x - breakpoint) + (slope1 * breakpoint + intercept)
        )
        residuals = y - y_pred
        return np.sum(huber_loss(residuals, delta))

    def piecewise_fit_huber(x, y, delta):
        """Find the optimal breakpoint for piecewise linear regression with Huber loss."""
        init_params = [
            np.median(x),  # Breakpoint
            (y[-1] - y[0]) / (x[-1] - x[0]),  # Slope1
            (y[-1] - y[0]) / (x[-1] - x[0]),  # Slope2
            np.mean(y)  # Intercept
        ]
        bounds = [
            (x.min(), x.max()),  # Breakpoint
            (-np.inf, np.inf),   # Slope1
            (-np.inf, np.inf),   # Slope2
            (-np.inf, np.inf)    # Intercept
        ]
        result = minimize(piecewise_huber_loss, init_params, args=(x, y, delta), bounds=bounds)
        return result.x

    def plot_node(node_id, node_data, delta):
        # Data preparation
        width = np.array(node_data['width'])
        wse = np.array(node_data['wse'])

        # Fit piecewise model with Huber loss
        breakpoint, slope1, slope2, intercept = piecewise_fit_huber(width, wse, delta)

        # Calculate predicted values and residuals
        segment1 = slope1 * width[width <= breakpoint] + intercept
        segment2 = slope2 * (width[width > breakpoint] - breakpoint) + (slope1 * breakpoint + intercept)
        y_pred = np.concatenate([segment1, segment2])
        residuals = wse - y_pred
        
        # Pseudo-R²
        mad_y = np.median(np.abs(wse - np.median(wse)))
        mad_residuals = np.median(np.abs(residuals))
        r2_pseudo = 1 - (mad_residuals / mad_y)**2 if mad_y > 0 else 0
        r2_pseudo = max(0, r2_pseudo)  # Avoid negative R² due to numerical issues

        # Spearman correlation
        spearman_corr, p_value = scipy.stats.spearmanr(width, wse)

        # Plot scatter plot and residuals side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # Scatter plot with regression
        axes[0].scatter(width, wse, alpha=0.8, c="darkcyan", edgecolors='cyan', linewidths=1)
        x_segment1 = np.linspace(width.min(), breakpoint, 100)
        x_segment2 = np.linspace(breakpoint, width.max(), 100)
        y_segment1 = slope1 * x_segment1 + intercept
        y_segment2 = slope2 * (x_segment2 - breakpoint) + (slope1 * breakpoint + intercept)
        axes[0].plot(x_segment1, y_segment1, color="blue", label=f"Segment 1: y = {slope1:.3f}x + {intercept:.3f}")
        axes[0].plot(x_segment2, y_segment2, color="orange", label=f"Segment 2: y = {slope2:.3f}(x-{breakpoint:.3f}) + {slope1 * breakpoint + intercept:.3f}")
        axes[0].set_title(f"Node: {node_id}\n$R^2_{{pseudo}}$: {r2_pseudo:.2f}, Spearman: {spearman_corr:.2f}")
        axes[0].set_xlabel("Width")
        axes[0].set_ylabel("WSE")
        axes[0].legend()
        if show_p_value:
            axes[0].text(0.05, 0.85, f"p-value: {p_value:.3f}", ha='left', va='center', transform=axes[0].transAxes, fontsize=10, color="gray")

        # Residuals plot
        axes[1].scatter(width, residuals, alpha=0.8, c="red", edgecolors='black', linewidths=0.5)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Residuals Plot")
        axes[1].set_xlabel("Width")
        axes[1].set_ylabel("Residuals")

        plt.tight_layout()
        plt.show()

        return {
            'Node': node_id,
            'Breakpoint': round(breakpoint, 3),
            'Slope1': round(slope1, 3),
            'Slope2': round(slope2, 3),
            'Intercept': round(intercept, 3),
            'R2_pseudo': round(r2_pseudo, 3),
            'Spearman': round(spearman_corr, 3),
            'p_value': round(p_value, 3)
        }

    results = []
    if node_id:
        node_data = river[node_id]
        if len(node_data['width']) < min_obs:
            raise ValueError(f"Node {node_id} does not have enough observations (min_obs={min_obs}).")
        
        if min_spearman is not None or min_p_value is not None:
            spearman_corr, p_value = scipy.stats.spearmanr(node_data['width'], node_data['wse'])
            if min_spearman is not None and spearman_corr < min_spearman:
                return pd.DataFrame(results)
            if min_p_value is not None and p_value > min_p_value:
                return pd.DataFrame(results)

        results.append(plot_node(node_id, node_data, delta))
    else:
        for nid, node_data in river.items():
            if len(node_data['width']) < min_obs:
                continue
            
            if min_spearman is not None or min_p_value is not None:
                spearman_corr, p_value = scipy.stats.spearmanr(node_data['width'], node_data['wse'])
                if min_spearman is not None and spearman_corr < min_spearman:
                    continue
                if min_p_value is not None and p_value > min_p_value:
                    continue
            
            results.append(plot_node(nid, node_data, delta))

    return pd.DataFrame(results).round(3)
