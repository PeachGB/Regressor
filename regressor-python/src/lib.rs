//! `regressor` — Python bindings for the `regressor-rs` machine-learning core.
//!
//! The Rust core is kept entirely `pyo3`-free; this crate is a thin, modular
//! wrapper that mirrors the core's layout and exposes the models to Python under
//! `regressor.model.linear_model` and the preprocessing utilities under
//! `regressor.preprocessing`. All numeric input conversion lives in [`interop`].

use pyo3::prelude::*;

mod interop;
mod model;
mod preprocessing;

use model::enums::{PyMetric, PyPenalty};
use model::linear_model::linear_regression::LinearRegression;
use model::linear_model::logistic_regression::LogisticRegression;
use model::linear_model::ridge::Ridge;
use preprocessing::{LabelEncoder, MinMaxScaler, StandardScaler};

/// The top-level `regressor` Python module.
#[pymodule]
fn regressor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    let model_mod = PyModule::new(py, "model")?;
    let linear_model_mod = PyModule::new(py, "linear_model")?;

    linear_model_mod.add_class::<LinearRegression>()?;
    linear_model_mod.add_class::<Ridge>()?;
    linear_model_mod.add_class::<LogisticRegression>()?;
    linear_model_mod.add_class::<PyMetric>()?;
    linear_model_mod.add_class::<PyPenalty>()?;

    model_mod.add_submodule(&linear_model_mod)?;
    m.add_submodule(&model_mod)?;

    let preprocessing_mod = PyModule::new(py, "preprocessing")?;
    preprocessing_mod.add_class::<StandardScaler>()?;
    preprocessing_mod.add_class::<MinMaxScaler>()?;
    preprocessing_mod.add_class::<LabelEncoder>()?;
    m.add_submodule(&preprocessing_mod)?;

    // Register the submodules in `sys.modules` so that
    // `from regressor.model.linear_model import LinearRegression` and
    // `from regressor.preprocessing import StandardScaler` work.
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("regressor.model", &model_mod)?;
    modules.set_item("regressor.model.linear_model", &linear_model_mod)?;
    modules.set_item("regressor.preprocessing", &preprocessing_mod)?;

    Ok(())
}
