//! Python binding for the gradient-descent logistic regression model.

use pyo3::prelude::*;

use regressor_rs::model::Model;
use regressor_rs::model::linear_model::LogisticRegression as CoreLogisticRegression;

use crate::interop::{pyany_to_array1, pyany_to_array2, to_py_err};
use crate::model::enums::PyPenalty;

/// Binary logistic regression trained with gradient descent.
///
/// Accepts NumPy arrays, pandas/polars frames, or Python lists for `x`/`y`.
#[pyclass(name = "LogisticRegression")]
#[derive(Clone)]
pub struct LogisticRegression {
    inner: CoreLogisticRegression,
}

#[pymethods]
impl LogisticRegression {
    #[new]
    #[pyo3(signature = (learning_rate, epochs, fit_intercept = true))]
    fn new(learning_rate: f64, epochs: usize, fit_intercept: bool) -> Self {
        LogisticRegression {
            inner: CoreLogisticRegression::with_fit_intercept(learning_rate, epochs, fit_intercept),
        }
    }

    /// Apply an L1 or L2 regularization penalty (see `Penalty.l1` / `Penalty.l2`).
    fn set_penalty(&mut self, penalty: PyPenalty) {
        self.inner.set_penalty(penalty.inner);
    }

    /// Fit the model to features `x` and binary target `y`.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = pyany_to_array2(py, x)?;
        let y = pyany_to_array1(py, y)?;
        Model::fit(&mut self.inner, &x, &y).map_err(to_py_err)
    }

    /// Predict class probabilities for features `x`.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let x = pyany_to_array2(py, x)?;
        let out = Model::predict(&self.inner, &x).map_err(to_py_err)?;
        Ok(out.to_vec())
    }

    /// Return `(intercept, coefficients)` for the fitted model.
    fn params(&self) -> PyResult<(f64, Vec<f64>)> {
        self.inner.params().map_err(to_py_err)
    }

    /// Classification accuracy of predicted probabilities against ground truth.
    fn score(
        &self,
        py: Python<'_>,
        y_true: &Bound<'_, PyAny>,
        y_pred: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let y_true = pyany_to_array1(py, y_true)?;
        let y_pred = pyany_to_array1(py, y_pred)?;
        self.inner.score(y_true, y_pred).map_err(to_py_err)
    }

    /// Serialize the fitted model to `path` as JSON.
    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(path).map_err(to_py_err)
    }

    /// Load a model previously written with `save`.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(LogisticRegression {
            inner: CoreLogisticRegression::load(path).map_err(to_py_err)?,
        })
    }
}
