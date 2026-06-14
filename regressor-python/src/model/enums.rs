//! Python-facing wrappers around the core `Metric` and `Penalty` enums.
//!
//! These keep `pyo3` out of the `regressor-rs` core: the core defines plain
//! Rust enums, and this module exposes thin `#[pyclass]` mirrors that convert
//! to and from them.

use pyo3::prelude::*;

use regressor_rs::model::linear_model::{Metric as CoreMetric, Penalty as CorePenalty};

/// Scoring metric for regression models.
#[pyclass(name = "Metric", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyMetric {
    R2,
    MSE,
    RMSE,
}

impl From<PyMetric> for CoreMetric {
    fn from(metric: PyMetric) -> Self {
        match metric {
            PyMetric::R2 => CoreMetric::R2,
            PyMetric::MSE => CoreMetric::MSE,
            PyMetric::RMSE => CoreMetric::RMSE,
        }
    }
}

/// Regularization penalty for logistic regression.
///
/// Construct with the `l1` / `l2` static methods, e.g. `Penalty.l2(0.1)`.
#[pyclass(name = "Penalty")]
#[derive(Clone)]
pub struct PyPenalty {
    pub(crate) inner: CorePenalty,
}

#[pymethods]
impl PyPenalty {
    /// L1 (lasso) penalty with regularization strength `lambda`.
    #[staticmethod]
    fn l1(lambda: f64) -> Self {
        PyPenalty {
            inner: CorePenalty::L1(lambda),
        }
    }

    /// L2 (ridge) penalty with regularization strength `lambda`.
    #[staticmethod]
    fn l2(lambda: f64) -> Self {
        PyPenalty {
            inner: CorePenalty::L2(lambda),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            CorePenalty::L1(l) => format!("Penalty.l1({l})"),
            CorePenalty::L2(l) => format!("Penalty.l2({l})"),
        }
    }
}
