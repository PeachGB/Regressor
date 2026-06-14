# Regressor

A from-scratch regression library written in **Rust**, with first-class **Python** bindings.
The numerical core implements linear, ridge, and logistic regression by hand on top of
[`ndarray`](https://docs.rs/ndarray) / [`ndarray-linalg`](https://docs.rs/ndarray-linalg),
together with `StandardScaler` / `MinMaxScaler` / `LabelEncoder` preprocessing. The Python layer
exposes all of it through [PyO3](https://pyo3.rs) / [maturin](https://www.maturin.rs) as a package
called `regressor`.

The Python API is deliberately permissive about input: features and targets can be **NumPy arrays
(any numeric dtype), pandas `DataFrame`/`Series`, polars `DataFrame`/`Series`, or plain Python
lists** — all are normalized to `float64` internally.

---

## Architecture

A Cargo **workspace** with two members. Shared dependency versions are pinned once in the root
`Cargo.toml` under `[workspace.dependencies]`.

```
Regressor/
├── Cargo.toml                 # workspace manifest (shared dependency versions)
├── regressor-rs/              # pure-Rust numerical core (no Python deps)
│   └── src/
│       ├── model/
│       │   ├── mod.rs                 # Model + Differentiable traits
│       │   ├── functions.rs           # generic gradient_descent optimizer
│       │   └── linear_model/
│       │       ├── mod.rs             # LinearRegression, Ridge, LogisticRegression, Metric, Penalty
│       │       └── regression.rs      # OLS / ridge / sigmoid math + intercept helper
│       ├── preprocessing.rs           # StandardScaler, MinMaxScaler, LabelEncoder
│       ├── utils/                     # error types + metrics (R²/MSE/RMSE)
│       └── tests/                     # unit tests, one file per source module
└── regressor-python/          # PyO3 binding crate -> Python package `regressor`
    ├── src/
    │   ├── lib.rs                     # #[pymodule] registration
    │   ├── interop/mod.rs             # generic numeric conversion layer
    │   ├── preprocessing/mod.rs       # StandardScaler / MinMaxScaler / LabelEncoder wrappers
    │   └── model/
    │       ├── enums.rs               # Metric / Penalty Python wrappers
    │       └── linear_model/
    │           ├── linear_regression.rs
    │           ├── ridge.rs
    │           └── logistic_regression.rs
    ├── examples/                      # runnable example programs (see below)
    └── tests/                         # pytest suite for the binding
```

**Design principle:** `regressor-rs` is completely `pyo3`-free — this is a hard boundary. Everything
Python-specific (argument conversion, exception mapping, the `#[pyclass]` wrappers) lives in
`regressor-python`. The binding mirrors the core's module layout one-to-one and keeps each model in
its own thin wrapper file.

### The core (`regressor-rs`)

- **`Model` trait** — `fit(&mut self, x, y)` / `predict(&self, x)`, with associated
  `Input`/`Target`/`Output` types.
- **`Differentiable` trait** — `compute_gradient(...)`, implemented by iteratively-trained models so
  they reuse the single generic `gradient_descent` routine in `model/functions.rs`.
- **`LinearRegression`** — ordinary least squares. Solves the normal equations (`XᵀX β = Xᵀy`) via
  `ndarray-linalg`, through one path for any feature count.
- **`Ridge`** — L2-penalized least squares: solves `(XᵀX + αI) β = Xᵀy` in closed form, leaving the
  intercept un-penalized.
- **`LogisticRegression`** — binary classifier trained with the generic `gradient_descent` routine;
  supports optional **L1** / **L2** penalties (intercept left un-penalized when `fit_intercept` is on).
- **Automatic intercept** — every model carries a `fit_intercept` flag (default `true`). When set, a
  ones column is prepended internally so callers pass only real features, and `params()` peels the
  bias off as `(intercept, coefficients)`. With `fit_intercept=false`, no column is added and the
  intercept is reported as `0.0`.
- **Metrics** — `r_squared`, `mean_squared_error` (mean of squared errors), and
  `root_mean_squared_error` (`√(Σe²/n)`); all reject empty / size-mismatched input.
- **Preprocessing** — `StandardScaler` (zero mean, unit std), `MinMaxScaler` (`[0, 1]` rescale), and
  `LabelEncoder` (categorical labels → integer codes), each with `fit` / `transform` /
  `fit_transform` and serde derives.

### The binding (`regressor-python`)

- **`interop`** — `pyany_to_array2` / `pyany_to_array1` funnel every supported Python container
  through `numpy.asarray(obj, dtype="float64")` (with an explicit `to_numpy()` step for polars) into
  owned `ndarray` matrices/vectors. Core errors (`Box<dyn Error>`) are mapped to `PyValueError` via
  `to_py_err`.
- **`model::enums`** — `Metric` (enum: `R2` / `MSE` / `RMSE`) and `Penalty` (with `Penalty.l1(λ)` /
  `Penalty.l2(λ)` constructors) Python wrappers.
- **`model::linear_model`** — `LinearRegression`, `Ridge`, and `LogisticRegression` `#[pyclass]`es,
  each forwarding to the core model held in an `inner` field.
- **`preprocessing`** — `StandardScaler`, `MinMaxScaler`, and `LabelEncoder` `#[pyclass]`es under
  `regressor.preprocessing`. The scalers return NumPy arrays; `LabelEncoder` accepts any Python
  iterable (each element stringified via `str()`) and exposes `classes()` / `inverse_transform()`.

---

## Python usage

Install [maturin](https://www.maturin.rs) into a virtualenv and build the extension:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pandas polars
maturin develop -m regressor-python/Cargo.toml      # build + install into the venv
```

```python
import numpy as np
import pandas as pd
import polars as pl
from regressor.model.linear_model import (
    LinearRegression, Ridge, LogisticRegression, Metric, Penalty,
)
from regressor.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# No ones column needed — the intercept is fit automatically (fit_intercept=True).
X = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
y = [1 + 2 * r[0] + 3 * r[1] for r in X]

model = LinearRegression()                        # LinearRegression(fit_intercept=False) to opt out

# Any of these inputs work interchangeably:
model.fit(X, y)                                   # Python lists
model.fit(np.array(X), np.array(y))               # NumPy
model.fit(pd.DataFrame(X), pd.Series(y))          # pandas
model.fit(pl.DataFrame(X), pl.Series(y))          # polars

intercept, coefs = model.params()                 # -> (1.0, [2.0, 3.0])
preds = model.predict(X)
r2 = model.score(y, preds, Metric.R2)             # -> 1.0

model.save("model.json")
model = LinearRegression.load("model.json")

# Ridge regression (L2 penalty of strength `alpha`, intercept left un-penalized):
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Logistic regression with optional regularization:
clf = LogisticRegression(learning_rate=0.5, epochs=2000)
clf.set_penalty(Penalty.l2(0.01))
xs = np.array([[x] for x in (-2, -1, -0.5, 0.5, 1, 2)], dtype=float)
ys = np.array([0, 0, 0, 1, 1, 1], dtype=float)
clf.fit(xs, ys)
probs = clf.predict([[0.7]])                      # predict returns class probabilities
acc = clf.score(ys, clf.predict(xs))              # accuracy (thresholded at 0.5)

# Preprocessing:
scaled = StandardScaler().fit_transform(np.array(X, dtype=float))   # zero mean, unit std
unit   = MinMaxScaler().fit_transform(np.array(X, dtype=float))     # rescaled into [0, 1]
enc = LabelEncoder()
codes = enc.fit_transform(["b", "a", "c", "a"])   # -> [1.0, 0.0, 2.0, 0.0]
enc.classes()                                     # -> ['a', 'b', 'c']
enc.inverse_transform(codes)                      # -> ['b', 'a', 'c', 'a']
```

A runnable version of the above lives at `regressor-python/smoke_test.py`.

### Example programs

`regressor-python/examples/` contains standalone, didactic scripts. Build the extension with
`maturin develop` first, then run any of them:

| Script | Shows |
| --- | --- |
| `01_linear_regression.py` | Fit a multi-feature model, recover coefficients, score (R²/MSE/RMSE), save & reload. |
| `02_ridge_vs_ols.py` | How Ridge's L2 penalty tames coefficients when features are collinear. |
| `03_logistic_regression.py` | Binary classification: probabilities, a decision threshold, accuracy. |
| `04_preprocessing.py` | `StandardScaler`, `MinMaxScaler`, and `LabelEncoder` on raw data. |
| `05_pipeline.py` | End-to-end pandas → `LabelEncoder` + `StandardScaler` + `LogisticRegression` with a train/test split. |

```bash
python regressor-python/examples/01_linear_regression.py
```

---

## Building & testing

```bash
cargo build --workspace            # build core + binding
cargo test  -p regressor-rs        # core unit tests (all live under regressor-rs/src/tests/)
maturin develop -m regressor-python/Cargo.toml   # build the Python extension
python regressor-python/smoke_test.py            # end-to-end Python check
pytest regressor-python/tests                    # Python test suite (after `maturin develop`)
python regressor-python/examples/01_linear_regression.py   # any example script
```

The Rust unit tests are organized as a dedicated `regressor-rs/src/tests/` submodule with one
file per source module (`error`, `metrics`, `model`, `functions`, `regression`, `preprocessing`,
`linear_model`) rather than inline `#[cfg(test)]` blocks.

`ndarray-linalg` is configured with the `intel-mkl-static` feature, so Intel MKL is linked
statically — no system BLAS/LAPACK install is required. The first build is slow because MKL is
compiled in.

---

## Current state

**Working**
- ✅ Single Cargo workspace; both crates build cleanly (`cargo build --workspace`).
- ✅ `regressor-rs` core: `LinearRegression` (OLS), `Ridge` (closed-form L2), `LogisticRegression`
  (GD + L1/L2), R²/MSE/RMSE metrics, `StandardScaler` / `MinMaxScaler` / `LabelEncoder`, and
  `save`/`load` (JSON via serde) on the models.
- ✅ Automatic, unified intercept handling (`fit_intercept`, default on) across all models.
- ✅ `regressor-python` binding exposing `LinearRegression`, `Ridge`, `LogisticRegression`, `Metric`,
  `Penalty` under `regressor.model.linear_model`, and `StandardScaler` / `MinMaxScaler` /
  `LabelEncoder` under `regressor.preprocessing`.
- ✅ Generic numeric input layer (NumPy / pandas / polars / lists), verified to produce identical
  results across all four input types.
- ✅ Correct MSE / RMSE formulas, with empty / size-mismatch error paths.
- ✅ Per-module Rust unit tests (in `regressor-rs/src/tests/`), a Python smoke test, a Python
  `pytest` suite, and five runnable example programs — all covering fit/predict/score/save/load,
  edge cases, and error paths.

**Notes / caveats**
- ℹ️ `fit_intercept=True` is the default, so pass only real features. Set `fit_intercept=False` if you
  supply your own intercept column; `params()` then reports the intercept as `0.0`.
- ℹ️ `StandardScaler` standardizes with the **sample** standard deviation (ddof=1); compare with
  `numpy.std(..., ddof=1)`.
- ℹ️ `LogisticRegression.predict` returns class **probabilities**; `score` thresholds them at 0.5 to
  compute accuracy.

---

## Roadmap

Future ideas: more solvers (gradient-descent `LinearRegression`, elastic-net), additional models
(polynomial features, KNN), `predict_proba` / class-label outputs for the classifier, and exposing
`save`/`load` for the preprocessing transformers.
