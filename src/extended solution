
"""
This script solves the parametric curve fitting problem by finding
optimal values for θ, M, and X that minimize the L1 distance between
the predicted parametric curve and observed data points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. PARAMETRIC CURVE DEFINITION
# ============================================================================

def parametric_curve(t, theta, M, X):
    """
    Generate x, y coordinates for the parametric curve.

    Parameters:
    -----------
    t : numpy array
        Parameter values (6 < t < 60)
    theta : float
        Angle parameter (in radians)
    M : float
        Exponential coefficient (-0.05 < M < 0.05)
    X : float
        Horizontal offset (0 < X < 100)

    Returns:
    --------
    x, y : tuple of numpy arrays
        Coordinates of the parametric curve
    """
    x = (t * np.cos(theta) - 
         np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + 
         X)
    y = (42 + t * np.sin(theta) + 
         np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta))
    return x, y


# ============================================================================
# 2. LOSS FUNCTION DEFINITION
# ============================================================================

def calculate_l1_distance(predicted_points, observed_points):
    """
    Calculate mean L1 distance between predicted and observed points.

    For each predicted point, finds the closest observed point using
    Manhattan distance (L1 norm).

    Parameters:
    -----------
    predicted_points : np.ndarray, shape (n_pred, 2)
        Predicted curve points
    observed_points : np.ndarray, shape (n_obs, 2)
        Observed data points

    Returns:
    --------
    mean_distance : float
        Mean L1 distance
    """
    distances = cdist(predicted_points, observed_points, metric='cityblock')
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)


def objective_function(params, observed_points, t_values):
    """
    Objective function to minimize: mean L1 distance.

    Parameters:
    -----------
    params : list or array
        [theta_deg, M, X] where theta_deg is in degrees
    observed_points : np.ndarray
        Known data points
    t_values : np.ndarray
        Parameter values for sampling the curve

    Returns:
    --------
    loss : float
        Mean L1 distance (loss value to minimize)
    """
    theta_deg, M, X = params
    theta_rad = np.radians(theta_deg)

    try:
        # Generate predicted curve
        x_pred, y_pred = parametric_curve(t_values, theta_rad, M, X)
        predicted_points = np.column_stack((x_pred, y_pred))

        # Calculate L1 distance
        loss = calculate_l1_distance(predicted_points, observed_points)
        return loss
    except:
        return 1e6  # Return large penalty on error


# ============================================================================
# 3. OPTIMIZATION PIPELINE
# ============================================================================

def optimize_parameters(data_file, verbose=True):
    """
    Main optimization pipeline to find optimal parameters.

    Stages:
    1. Global optimization using Differential Evolution
    2. Local refinement using Nelder-Mead
    3. Fine-tuning using L-BFGS-B

    Parameters:
    -----------
    data_file : str
        Path to CSV file with observed points (columns: x, y)
    verbose : bool
        Print intermediate results

    Returns:
    --------
    results : dict
        Dictionary containing optimal parameters and metrics
    """

    # Load data
    if verbose:
        print("Loading data...")
    data = pd.read_csv(data_file)
    observed_points = data.values

    if verbose:
        print(f"  Loaded {len(observed_points)} points")
        print(f"  X range: [{observed_points[:, 0].min():.2f}, {observed_points[:, 0].max():.2f}]")
        print(f"  Y range: [{observed_points[:, 1].min():.2f}, {observed_points[:, 1].max():.2f}]")

    # Parameter bounds
    bounds = [
        (0, 50),        # theta in degrees
        (-0.05, 0.05),  # M
        (0, 100)        # X
    ]

    # Sampling for curve evaluation
    t_values = np.linspace(6, 60, 300)

    # STAGE 1: Global Optimization
    if verbose:
        print("\n" + "="*70)
        print("STAGE 1: Global Optimization (Differential Evolution)")
        print("="*70)

    result_de = differential_evolution(
        lambda p: objective_function(p, observed_points, t_values),
        bounds,
        maxiter=500,
        popsize=30,
        atol=1e-6,
        tol=1e-6,
        seed=42,
        polish=True
    )

    if verbose:
        print(f"Iterations: {result_de.nit}")
        print(f"Function evaluations: {result_de.nfev}")
        print(f"L1 Distance: {result_de.fun:.10f}")
        print(f"  θ = {result_de.x[0]:.6f}°")
        print(f"  M = {result_de.x[1]:.6f}")
        print(f"  X = {result_de.x[2]:.6f}")

    best_params = result_de.x
    best_loss = result_de.fun

    # STAGE 2: Local Refinement (Nelder-Mead)
    if verbose:
        print("\n" + "="*70)
        print("STAGE 2: Local Refinement (Nelder-Mead)")
        print("="*70)

    result_nm = minimize(
        lambda p: objective_function(p, observed_points, t_values),
        best_params,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
    )

    if verbose:
        print(f"L1 Distance: {result_nm.fun:.10f}")
        print(f"  θ = {result_nm.x[0]:.8f}°")
        print(f"  M = {result_nm.x[1]:.8f}")
        print(f"  X = {result_nm.x[2]:.8f}")

    if result_nm.fun < best_loss:
        best_params = result_nm.x
        best_loss = result_nm.fun

    # STAGE 3: Fine-tuning (L-BFGS-B)
    if verbose:
        print("\n" + "="*70)
        print("STAGE 3: Fine-tuning (L-BFGS-B)")
        print("="*70)

    result_lbfgs = minimize(
        lambda p: objective_function(p, observed_points, t_values),
        best_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    if verbose:
        print(f"L1 Distance: {result_lbfgs.fun:.10f}")
        print(f"  θ = {result_lbfgs.x[0]:.10f}°")
        print(f"  M = {result_lbfgs.x[1]:.10f}")
        print(f"  X = {result_lbfgs.x[2]:.10f}")

    if result_lbfgs.fun < best_loss:
        best_params = result_lbfgs.x
        best_loss = result_lbfgs.fun

    # Calculate validation metrics
    theta_rad = np.radians(best_params[0])
    t_fine = np.linspace(6, 60, 1000)
    x_pred, y_pred = parametric_curve(t_fine, theta_rad, best_params[1], best_params[2])
    predicted_points = np.column_stack((x_pred, y_pred))

    distances = cdist(predicted_points, observed_points, metric='cityblock')
    min_distances = np.min(distances, axis=1)

    results = {
        'theta_deg': best_params[0],
        'theta_rad': theta_rad,
        'M': best_params[1],
        'X': best_params[2],
        'l1_distance_mean': np.mean(min_distances),
        'l1_distance_median': np.median(min_distances),
        'l1_distance_std': np.std(min_distances),
        'l1_distance_max': np.max(min_distances),
        'l1_distance_min': np.min(min_distances),
        'observed_points': observed_points,
        'predicted_curve': (x_pred, y_pred)
    }

    return results


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def visualize_results(results, output_file='curve_fit_analysis.png'):
    """
    Create visualization of the curve fit.

    Parameters:
    -----------
    results : dict
        Results dictionary from optimize_parameters()
    output_file : str
        Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extract results
    observed = results['observed_points']
    x_pred, y_pred = results['predicted_curve']

    # Calculate distances for histogram
    distances_array = cdist(
        np.column_stack((x_pred, y_pred)), 
        observed, 
        metric='cityblock'
    )
    min_distances = np.min(distances_array, axis=1)

    # Plot 1: Curve fit
    ax1 = axes[0]
    ax1.scatter(observed[:, 0], observed[:, 1], alpha=0.3, s=10, 
                label='Observed points', color='blue')
    ax1.plot(x_pred, y_pred, 'r-', linewidth=2, label='Predicted curve')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Parametric Curve Fit', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: L1 distance histogram
    ax2 = axes[1]
    ax2.hist(min_distances, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(results['l1_distance_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {results['l1_distance_mean']:.6f}")
    ax2.axvline(results['l1_distance_median'], color='orange', linestyle='--', 
                linewidth=2, label=f"Median: {results['l1_distance_median']:.6f}")
    ax2.set_xlabel('L1 Distance', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('L1 Distance Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


# ============================================================================
# 5. SUBMISSION FORMAT
# ============================================================================

def generate_submission_format(results):
    """
    Generate LaTeX submission format for Desmos.

    Parameters:
    -----------
    results : dict
        Results dictionary from optimize_parameters()

    Returns:
    --------
    latex_str : str
        LaTeX formatted string ready for Desmos
    """
    theta = round(results['theta_deg'], 6)
    M = round(results['M'], 6)
    X = round(results['X'], 6)

    latex_str = (
        f"\\left(t*\\cos({theta})-e^{{{M}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\sin({theta})\\ +{X},42+\\ "
        f"t*\\sin({theta})+e^{{{M}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\cos({theta})\\right)"
    )

    return latex_str, theta, M, X


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FLAM ASSIGNMENT - PARAMETRIC CURVE FITTING")
    print("="*70)

    # Run optimization
    results = optimize_parameters('xy_data.csv', verbose=True)

    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nOptimal Parameters:")
    print(f"  θ = {results['theta_deg']:.10f}°")
    print(f"  M = {results['M']:.10f}")
    print(f"  X = {results['X']:.10f}")

    print(f"\nValidation Metrics:")
    print(f"  Mean L1 Distance: {results['l1_distance_mean']:.10f}")
    print(f"  Median L1 Distance: {results['l1_distance_median']:.10f}")
    print(f"  Std Dev: {results['l1_distance_std']:.10f}")
    print(f"  Max Distance: {results['l1_distance_max']:.10f}")
    print(f"  Min Distance: {results['l1_distance_min']:.10f}")

    # Generate submission format
    latex_str, theta_rounded, M_rounded, X_rounded = generate_submission_format(results)

    print(f"\nSubmission Format (Rounded to 6 decimals):")
    print(f"  θ = {theta_rounded}°")
    print(f"  M = {M_rounded}")
    print(f"  X = {X_rounded}")

    print(f"\nDesmos LaTeX (copy-paste):")
    print(latex_str)

    # Generate visualization
    visualize_results(results)

    print("\n" + "="*70)
    print("Optimization complete!")
    print("="*70)
