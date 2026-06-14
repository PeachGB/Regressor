//! Python bindings for the `regressor-rs` preprocessing utilities.
//!
//! Mirrors the core's `preprocessing` module: thin `#[pyclass]` wrappers that
//! reuse the generic [`crate::interop`] conversion layer for numeric input and
//! forward to the core scalers / encoder.

use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;

use regressor_rs::preprocessing::{
    LabelEncoder as CoreLabelEncoder, MinMaxScaler as CoreMinMaxScaler,
    StandardScaler as CoreStandardScaler,
};

use crate::interop::{pyany_to_array2, to_py_err};

/// Read any Python iterable of labels into owned strings (via `str()`).
fn pyany_to_labels(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut labels = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        labels.push(item.str()?.extract::<String>()?);
    }
    Ok(labels)
}

/// Column-wise standardization to zero mean and unit variance.
#[pyclass(name = "StandardScaler")]
#[derive(Clone)]
pub struct StandardScaler {
    inner: CoreStandardScaler,
}

#[pymethods]
impl StandardScaler {
    #[new]
    fn new() -> Self {
        StandardScaler {
            inner: CoreStandardScaler::new(),
        }
    }

    /// Learn the per-column mean and standard deviation of `x`.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = pyany_to_array2(py, x)?;
        self.inner.fit(&x).map_err(to_py_err)
    }

    /// Standardize `x` using the previously learned statistics.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = pyany_to_array2(py, x)?;
        let out = self.inner.transform(&x).map_err(to_py_err)?;
        Ok(out.to_pyarray(py))
    }

    /// Fit to `x` and return the standardized matrix in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = pyany_to_array2(py, x)?;
        let out = self.inner.fit_transform(&x).map_err(to_py_err)?;
        Ok(out.to_pyarray(py))
    }
}

/// Column-wise rescaling of each feature into the `[0, 1]` range.
#[pyclass(name = "MinMaxScaler")]
#[derive(Clone)]
pub struct MinMaxScaler {
    inner: CoreMinMaxScaler,
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    fn new() -> Self {
        MinMaxScaler {
            inner: CoreMinMaxScaler::new(),
        }
    }

    /// Learn the per-column minimum and maximum of `x`.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = pyany_to_array2(py, x)?;
        self.inner.fit(&x).map_err(to_py_err)
    }

    /// Rescale `x` into `[0, 1]` using the previously learned bounds.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = pyany_to_array2(py, x)?;
        let out = self.inner.transform(&x).map_err(to_py_err)?;
        Ok(out.to_pyarray(py))
    }

    /// Fit to `x` and return the rescaled matrix in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = pyany_to_array2(py, x)?;
        let out = self.inner.fit_transform(&x).map_err(to_py_err)?;
        Ok(out.to_pyarray(py))
    }
}

/// Map categorical labels to consecutive integer codes (`0..n_classes`).
#[pyclass(name = "LabelEncoder")]
#[derive(Clone)]
pub struct LabelEncoder {
    inner: CoreLabelEncoder,
}

#[pymethods]
impl LabelEncoder {
    #[new]
    fn new() -> Self {
        LabelEncoder {
            inner: CoreLabelEncoder::new(),
        }
    }

    /// Learn the sorted set of unique labels in `labels`.
    fn fit(&mut self, labels: &Bound<'_, PyAny>) -> PyResult<()> {
        let labels = pyany_to_labels(labels)?;
        self.inner.fit(&labels).map_err(to_py_err)
    }

    /// Encode `labels` into their integer codes.
    fn transform(&self, labels: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let labels = pyany_to_labels(labels)?;
        Ok(self.inner.transform(&labels).map_err(to_py_err)?.to_vec())
    }

    /// Fit to `labels` and return their codes in one step.
    fn fit_transform(&mut self, labels: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let labels = pyany_to_labels(labels)?;
        Ok(self
            .inner
            .fit_transform(&labels)
            .map_err(to_py_err)?
            .to_vec())
    }

    /// Map integer codes back to their original labels.
    fn inverse_transform(&self, codes: Vec<f64>) -> PyResult<Vec<String>> {
        self.inner.inverse_transform(&codes).map_err(to_py_err)
    }

    /// The learned classes, in code order.
    fn classes(&self) -> Option<Vec<String>> {
        self.inner.classes().map(|c| c.to_vec())
    }
}
