//! Generic numeric conversion layer.
//!
//! Bridges the gap between flexible Python numeric containers and the strict
//! `ndarray` types the `regressor-rs` core operates on. Every supported input
//! family — NumPy arrays (any numeric dtype), pandas `DataFrame`/`Series`,
//! polars `DataFrame`/`Series`, and (nested) Python lists/sequences — is funneled
//! through a single `numpy.asarray(obj, dtype="float64")` normalization so the
//! per-model bindings never have to care which one the caller passed.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Map a core `RegressorResult` error into a Python `ValueError`.
pub fn to_py_err(e: Box<dyn std::error::Error>) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Normalize any supported Python numeric container into a NumPy `float64` array.
///
/// Polars objects are converted with `to_numpy()` first (they do not expose a
/// uniform `__array__`); everything else is handed straight to `numpy.asarray`,
/// which natively understands NumPy arrays, pandas objects and Python sequences.
fn to_numpy_f64<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let module: String = obj
        .get_type()
        .getattr("__module__")
        .and_then(|m| m.extract())
        .unwrap_or_default();

    let prepared = if module.starts_with("polars") {
        obj.call_method0("to_numpy")?
    } else {
        obj.clone()
    };

    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    np.call_method("asarray", (prepared,), Some(&kwargs))
}

/// Convert a generic Python numeric input into an owned 2-D feature matrix.
///
/// A 1-D input is treated as a single feature column (`reshape(-1, 1)`).
pub fn pyany_to_array2<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Array2<f64>> {
    let arr = to_numpy_f64(py, obj)?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    let arr = match ndim {
        1 => arr.call_method1("reshape", (-1i64, 1i64))?,
        2 => arr,
        n => {
            return Err(PyValueError::new_err(format!(
                "expected a 1-D or 2-D numeric input for X, got {n} dimensions"
            )));
        }
    };
    let pyarr = arr.cast::<PyArray2<f64>>().map_err(|e| {
        PyValueError::new_err(format!("could not read X as a numeric matrix: {e}"))
    })?;
    Ok(pyarr.readonly().as_array().to_owned())
}

/// Convert a generic Python numeric input into an owned 1-D vector (raveled).
pub fn pyany_to_array1<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Array1<f64>> {
    let arr = to_numpy_f64(py, obj)?;
    let arr = arr.call_method0("ravel")?;
    let pyarr = arr.cast::<PyArray1<f64>>().map_err(|e| {
        PyValueError::new_err(format!("could not read input as a numeric vector: {e}"))
    })?;
    Ok(pyarr.readonly().as_array().to_owned())
}
