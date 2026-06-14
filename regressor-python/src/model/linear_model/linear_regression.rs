//! Python binding for the ordinary least-squares linear regression model.

use pyo3::prelude::*;

use regressor_rs::model::Model;
use regressor_rs::model::linear_model::LinearRegression as CoreLinearRegression;

use crate::interop::{pyany_to_array1, pyany_to_array2, to_py_err};
use crate::model::enums::PyMetric;

/// Ordinary least-squares linear regression.
///
/// Accepts NumPy arrays, pandas/polars frames, or Python lists for `x`/`y`.
/// With `fit_intercept=True` (the default) a bias term is added for you, so `x`
/// should contain only real features.
#[pyclass(name = "LinearRegression")]
#[derive(Clone)]
pub struct LinearRegression {
    inner: CoreLinearRegression,
}

#[pymethods]
impl LinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept = true))]
    fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            inner: CoreLinearRegression::with_fit_intercept(fit_intercept),
        }
    }

    /// Fit the model to features `x` and target `y`.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = pyany_to_array2(py, x)?;
        let y = pyany_to_array1(py, y)?;
        Model::fit(&mut self.inner, &x, &y).map_err(to_py_err)
    }

    /// Predict targets for features `x`.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let x = pyany_to_array2(py, x)?;
        let out = Model::predict(&self.inner, &x).map_err(to_py_err)?;
        Ok(out.to_vec())
    }

    /// Return `(intercept, coefficients)` for the fitted model.
    fn params(&self) -> PyResult<(f64, Vec<f64>)> {
        self.inner.params().map_err(to_py_err)
    }

    /// Score predictions against ground truth using the given metric.
    fn score(
        &self,
        py: Python<'_>,
        y_true: &Bound<'_, PyAny>,
        y_pred: &Bound<'_, PyAny>,
        metric: PyMetric,
    ) -> PyResult<f64> {
        let y_true = pyany_to_array1(py, y_true)?;
        let y_pred = pyany_to_array1(py, y_pred)?;
        self.inner
            .score(y_true, y_pred, metric.into())
            .map_err(to_py_err)
    }

    /// Serialize the fitted model to `path` as JSON.
    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(path).map_err(to_py_err)
    }

    /// Load a model previously written with `save`.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(LinearRegression {
            inner: CoreLinearRegression::load(path).map_err(to_py_err)?,
        })
    }
}
