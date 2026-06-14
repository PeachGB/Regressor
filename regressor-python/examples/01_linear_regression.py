"""Linear regression: recover a known relationship, score it, save and reload.

Run with:  python regressor-python/examples/01_linear_regression.py
"""
import os
import tempfile

import numpy as np

from regressor.model.linear_model import LinearRegression, Metric

# --- Synthetic data ---------------------------------------------------------
# A toy "apartment price" model:  price = 50 + 0.3*area + 15*rooms + noise.
rng = np.random.default_rng(0)
n = 200
area = rng.uniform(30, 120, n)        # m^2
rooms = rng.integers(1, 5, n).astype(float)
noise = rng.normal(0, 5, n)
price = 50 + 0.3 * area + 15 * rooms + noise

X = np.column_stack([area, rooms])    # only real features; intercept is automatic
y = price

# --- Fit --------------------------------------------------------------------
model = LinearRegression()            # fit_intercept=True by default
model.fit(X, y)

intercept, coefs = model.params()
print("Recovered model:")
print(f"  intercept ~ {intercept:6.2f}   (true 50)")
print(f"  area coef ~ {coefs[0]:6.3f}   (true 0.3)")
print(f"  room coef ~ {coefs[1]:6.2f}   (true 15)")

# --- Score ------------------------------------------------------------------
preds = model.predict(X)
print("\nFit quality on the training set:")
print(f"  R2   = {model.score(y, preds, Metric.R2):.4f}")
print(f"  MSE  = {model.score(y, preds, Metric.MSE):.3f}")
print(f"  RMSE = {model.score(y, preds, Metric.RMSE):.3f}")

# --- Predict on new data ----------------------------------------------------
new = np.array([[55.0, 2.0], [100.0, 4.0]])
print("\nPredictions for new apartments:")
for (a, r), p in zip(new, model.predict(new)):
    print(f"  {a:5.0f} m^2, {r:.0f} rooms -> {p:7.2f}k")

# --- Persist ----------------------------------------------------------------
path = os.path.join(tempfile.gettempdir(), "apartments_linreg.json")
model.save(path)
reloaded = LinearRegression.load(path)
assert np.allclose(model.predict(new), reloaded.predict(new))
print(f"\nSaved to {path} and reloaded with identical predictions.")
