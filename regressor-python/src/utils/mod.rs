use ndarray::{Array1, Array2};
use numpy::{PyArray, PyArray2, PyReadonlyArray2};
use polars::datatypes::{DataType, Float64Type};
use polars::frame::DataFrame;
use polars::prelude::*;
use pyo3::Bound;
use pyo3::types::PyAny;
use pyo3::types::PyAnyMethods;
use pyo3_polars::PyDataFrame;
use crate::utils::error::{RegressorError, RegressorResult};

pub mod math;
pub mod error;
pub mod metrics;

pub fn series_to_model_vector(y:Series) -> Result<Array1<f64>, RegressorError>{
    if !y.dtype().is_numeric(){return Err(RegressorError::NonNumericInput(y.dtype().to_string()))}
    let y = y.cast(&DataType::Float64)
        .map_err(|e| RegressorError::NonNumericInput(e.to_string()))?
        .f64()?
        .iter()
        .map(|n| n.unwrap()).collect::<Vec<f64>>();
    Ok(Array1::from(y))
}
pub fn data_frame_to_model_matrix(mut x:DataFrame) -> Result<Array2<f64>, RegressorError>{
    for dtype in x.dtypes(){
        if !dtype.is_numeric(){return Err(RegressorError::NonNumericInput(dtype.to_string()))}
    }
    let n_rows = x.height();

    let bias = Series::new("bias".into(), vec![1.0; n_rows]);

    x.insert_column(0, Column::from(bias))?;

    let x = x.to_ndarray::<Float64Type>(IndexOrder::C)?;

    Ok(x)
}

