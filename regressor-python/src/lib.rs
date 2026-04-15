pub mod model;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
pub fn regressor(py: Python, m: &Bound<'_,PyModule>) -> PyResult<()> {
    let model_mod = PyModule::new(py, "model")?;
    let linear_model_mod = PyModule::new(py, "linear_model")?;
    let regression_mod = PyModule::new(py, "regression")?;

    regression_mod.add_class::<model::linear_model::regression::LinearRegression>()?;
    regression_mod.add_class::<model::linear_model::regression::Solver>()?;

    linear_model_mod.add_submodule(&regression_mod)?;
    model_mod.add_submodule(&linear_model_mod)?;

    m.add_submodule(&model_mod)?;



    Ok(())
}
