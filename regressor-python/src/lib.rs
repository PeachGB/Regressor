pub mod model;
pub mod utils;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub fn regressor(py: Python, m: &Bound<'_,PyModule>) -> PyResult<()> {
    let model_mod = PyModule::new(py, "model")?;
    let linear_model_mod = PyModule::new(py, "linear_model")?;


    linear_model_mod.add_class::<model::linear_model::LinearRegression>()?;

    model_mod.add_submodule(&linear_model_mod)?;

    m.add_submodule(&model_mod)?;

    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;

    modules.set_item("regressor.model", &model_mod)?;
    modules.set_item("regressor.model.linear_model", &linear_model_mod)?;


    Ok(())
}
