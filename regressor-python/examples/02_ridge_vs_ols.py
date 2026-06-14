"""Ridge vs. ordinary least squares on noisy, correlated features.

When two features are highly correlated, OLS can hand out large, unstable
coefficients. Ridge's L2 penalty shrinks them toward zero (the intercept is left
un-penalized), trading a little bias for much steadier estimates.

Run with:  python regressor-python/examples/02_ridge_vs_ols.py
"""
import numpy as np

from regressor.model.linear_model import LinearRegression, Ridge, Metric

# --- Two nearly-collinear features ------------------------------------------
rng = np.random.default_rng(42)
n = 60
x1 = rng.normal(0, 1, n)
x2 = x1 + rng.normal(0, 0.01, n)      # almost a copy of x1
y = 3.0 + 2.0 * x1 + rng.normal(0, 0.5, n)

X = np.column_stack([x1, x2])

# --- Ordinary least squares -------------------------------------------------
ols = LinearRegression()
ols.fit(X, y)
ols_intercept, ols_coefs = ols.params()

# --- Ridge with a moderate penalty ------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
ridge_intercept, ridge_coefs = ridge.params()

print("Coefficients on two nearly-identical features:")
print(f"  OLS   : intercept={ols_intercept:6.3f}  coefs={[round(c, 3) for c in ols_coefs]}")
print(f"  Ridge : intercept={ridge_intercept:6.3f}  coefs={[round(c, 3) for c in ridge_coefs]}")

ols_norm = float(np.hypot(*ols_coefs))
ridge_norm = float(np.hypot(*ridge_coefs))
print(f"\nCoefficient magnitude ||w||:  OLS={ols_norm:.3f}  Ridge={ridge_norm:.3f}")
print("Ridge keeps the slope sane instead of letting the two features fight.")

# --- Both still explain the data well ---------------------------------------
print("\nTraining R2:")
print(f"  OLS   = {ols.score(y, ols.predict(X), Metric.R2):.4f}")
print(f"  Ridge = {ridge.score(y, ridge.predict(X), Metric.R2):.4f}")
