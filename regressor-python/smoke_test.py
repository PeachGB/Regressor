"""Smoke test: every supported input family must produce identical results."""
import tempfile, os
import numpy as np
import pandas as pd
import polars as pl
from regressor.model.linear_model import (
    LinearRegression, Ridge, LogisticRegression, Metric, Penalty,
)
from regressor.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# y = 1 + 2*x0 + 3*x1. No ones column: the intercept is fit automatically.
X = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]]
y = [1 + 2 * r[0] + 3 * r[1] for r in X]

variants = {
    "list":   (X, y),
    "numpy":  (np.array(X), np.array(y)),
    "pandas": (pd.DataFrame(X), pd.Series(y)),
    "polars": (pl.DataFrame(X, schema=["x0", "x1"]), pl.Series(y)),
}

print("== LinearRegression: params must match across input types ==")
ref = None
for name, (xv, yv) in variants.items():
    m = LinearRegression()
    m.fit(xv, yv)
    intercept, coefs = m.params()
    preds = m.predict(xv)
    r2 = m.score(yv, preds, Metric.R2)
    print(f"  {name:7} intercept={intercept:+.4f} coefs={[round(c,4) for c in coefs]} R2={r2:.4f}")
    assert all(np.isfinite(preds)), f"{name}: non-finite predictions"
    if ref is None:
        ref = (intercept, coefs)
    else:
        assert np.allclose(ref[0], intercept), f"{name}: intercept differs"
        assert np.allclose(ref[1], coefs), f"{name}: coefs differ"
assert np.allclose(ref[0], 1.0) and np.allclose(ref[1], [2.0, 3.0])
print("  -> all input types agree, intercept fit automatically ✓")

print("== save / load round-trip ==")
m = LinearRegression(); m.fit(np.array(X), np.array(y))
path = os.path.join(tempfile.gettempdir(), "lr.json")
m.save(path)
loaded = LinearRegression.load(path)
assert np.allclose(m.params()[0], loaded.params()[0])
assert np.allclose(m.params()[1], loaded.params()[1])
print("  -> reloaded params identical ✓")

print("== Ridge: L2 penalty shrinks coefficients ==")
r = Ridge(alpha=10.0); r.fit(np.array(X), np.array(y))
_, ridge_coefs = r.params()
assert all(abs(c) < o for c, o in zip(ridge_coefs, [2.0, 3.0]))
print(f"  ridge coefs={[round(c,4) for c in ridge_coefs]} (shrunk vs [2, 3]) ✓")

print("== LogisticRegression (with L2 penalty) ==")
Xc = np.array([[x] for x in [-2, -1, -0.5, 0.5, 1, 2]], dtype=float)
yc = np.array([0, 0, 0, 1, 1, 1], dtype=float)
clf = LogisticRegression(0.5, 2000)
clf.set_penalty(Penalty.l2(0.01))
clf.fit(Xc, yc)
probs = clf.predict(Xc)
acc = clf.score(yc, probs)
print(f"  probs={[round(p,3) for p in probs]} accuracy={acc:.3f}")
assert acc == 1.0, "expected perfect separation on this toy set"
print("  -> classifier fits ✓")

print("== StandardScaler / MinMaxScaler ==")
raw = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
z = StandardScaler().fit_transform(raw)
# The scaler standardizes with the sample std (ddof=1), so check with ddof=1.
assert np.allclose(z.mean(axis=0), 0.0) and np.allclose(z.std(axis=0, ddof=1), 1.0)
mm = MinMaxScaler().fit_transform(raw)
assert np.allclose(mm.min(axis=0), 0.0) and np.allclose(mm.max(axis=0), 1.0)
print("  -> StandardScaler centered/scaled, MinMaxScaler in [0,1] ✓")

print("== LabelEncoder ==")
enc = LabelEncoder()
codes = enc.fit_transform(["b", "a", "c", "a"])
assert codes == [1.0, 0.0, 2.0, 0.0]
assert enc.inverse_transform(codes) == ["b", "a", "c", "a"]
assert enc.classes() == ["a", "b", "c"]
print("  -> labels encoded and round-tripped ✓")

print("\nALL SMOKE TESTS PASSED")
