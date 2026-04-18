use ndarray::{Array1, Array2};
use polars::datatypes::{DataType, Float64Type, PolarsNumericType};
use polars::frame::DataFrame;
use polars::prelude::{IndexOrder, Series};
use crate::utils::error::RegressorError;

pub mod math;
pub mod error;

pub fn series_to_ndarray(x:Series) -> Result<Array1<f64>, RegressorError>{
    if !x.dtype().is_numeric(){return Err(RegressorError::NonNumericInput(x.dtype().to_string()))}
    let x = x.cast(&DataType::Float64).map_err(|e| RegressorError::NonNumericInput(e.to_string()))?.f64()?.iter().map(|n| n.unwrap()).collect::<Vec<f64>>();
    Ok(Array1::from(x))
}
pub fn data_frame_to_ndarray(x:DataFrame) -> Result<Array2<f64>, RegressorError>{
    for dtype in x.dtypes(){
        if !dtype.is_numeric(){return Err(RegressorError::NonNumericInput(dtype.to_string()))}
    }


    Ok(x.to_ndarray::<Float64Type>(IndexOrder::C)?)
}


