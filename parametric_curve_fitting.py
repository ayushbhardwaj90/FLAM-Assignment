"""
FLAM Assignment - Parametric Curve Fitting Solution
Author: Ayush Bhardwaj
Date: 2025-11-08

This script finds optimal θ, M, X for the parametric equations by minimizing the L1 distance to given data points.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

data = pd.read_csv('xy_data.csv')
observed_points = data.values

def parametric_curve(t, theta, M, X):
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y

def calculate_l1_distance(predicted_points, observed_points):
    distances = cdist(predicted_points, observed_points, metric='cityblock')
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)

def objective_function(params, observed_points, t_values):
    theta, M, X = params
    theta_rad = np.radians(theta)
    x_pred, y_pred = parametric_curve(t_values, theta_rad, M, X)
    predicted_points = np.column_stack((x_pred, y_pred))
    return calculate_l1_distance(predicted_points, observed_points)

bounds = [
    (0, 50),       # θ in degrees
    (-0.05, 0.05), # M
    (0, 100)       # X
]
t_values = np.linspace(6, 60, 300)

result_de = differential_evolution(
    lambda p: objective_function(p, observed_points, t_values),
    bounds,
    maxiter=500,
    popsize=30,
    updating='deferred',
    polish=True
)
best_params = result_de.x
result_nm = minimize(
    lambda p: objective_function(p, observed_points, t_values),
    best_params,
    method='Nelder-Mead',
    options={'maxiter': 1000}
)
result_final = minimize(
    lambda p: objective_function(p, observed_points, t_values),
    result_nm.x,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000}
)

theta_opt, M_opt, X_opt = result_final.x
theta_opt = round(theta_opt, 6)
M_opt = round(M_opt, 6)
X_opt = round(X_opt, 6)

print(f"θ (degrees): {theta_opt}")
print(f"M: {M_opt}")
print(f"X: {X_opt}")

theta_rad_opt = np.radians(theta_opt)
t_fine = np.linspace(6, 60, 1000)
x_pred, y_pred = parametric_curve(t_fine, theta_rad_opt, M_opt, X_opt)

plt.figure(figsize=(10,6))
plt.scatter(observed_points[:,0], observed_points[:,1], s=5, alpha=0.2, label='Observed data')
plt.plot(x_pred, y_pred, 'r-', label='Predicted curve', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Parametric Curve Fit')
plt.savefig('curve_fit_analysis.png', dpi=150)
plt.close()
print("Plot saved as curve_fit_analysis.png.")
