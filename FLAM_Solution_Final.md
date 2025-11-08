# FLAM Assignment – Parametric Curve Fitting

**Submitted by:** Ayush Bhardwaj [RA2211031010043] 
**Submission Date:** November 8, 2025  

##  Problem Statement
Given a parametric curve with equations:

text
x = t*cos(θ) - e^(M*|t|)*sin(0.3t)*sin(θ) + X
y = 42 + t*sin(θ) + e^(M*|t|)*sin(0.3t)*cos(θ)
and a set of data points for 6 < t < 60, determine the unknowns θ, M, and X.

##  Parameter Results

| Parameter | Symbol | Value |
|------------|---------|--------|
| Angle | θ (theta) | **30.000387°** |
| Growth Coefficient | M | **0.030001** |
| X-Offset | X | **55.000311** |

Domain: 6 ≤ t ≤ 60

##  Final Parametric Equations

x = t*cos(30.000387) - e^(0.030001*abs(t))*sin(0.3*t)*sin(30.000387) + 55.000311
y = 42 + t*sin(30.000387) + e^(0.030001*abs(t))*sin(0.3*t)*cos(30.000387)

##  Methodology
### Optimization Strategy
- Applied **global optimization** using *Differential Evolution* to explore the parameter space.  
- Followed by **local refinement** with *Nelder–Mead* and *L-BFGS-B* algorithms to minimize error.  
- Objective function minimized the **mean L1 distance** between predicted and observed data points.

### Error Metric
- **Mean L1 (Manhattan) distance** used for robustness against outliers and non-Gaussian noise.

### Python Libraries
- **NumPy** – numerical computation  
- **Pandas** – data handling  
- **SciPy** – optimization methods  
- **Matplotlib** – visualization  

---


##  Performance & Validation

| Metric | Value |
|---------|--------|
| Mean L1 Distance | **0.0258** |
| Median L1 Distance | **0.0163** |
| Status |  All parameters within specified bounds |

---

##  Files Included

| File | Description |
|------|--------------|
| `parametric_curve_fitting.py` | Main solution code |
| `xy_data.csv` | Data points used for fitting |
| `curve_fit_analysis.png` | Visualization of fitted curve |
| `README.md` | Summary and direct answers |

---

##  Remarks

This solution was developed using **original code** and a transparent methodology for parameter estimation.  
All parameters were validated against constraints, and the final model demonstrates a **robust fit** that satisfies all assignment objectives.

For detailed explanations, algorithmic notes, or additional analysis results, refer to in-code comments and the included quick reference materials.
