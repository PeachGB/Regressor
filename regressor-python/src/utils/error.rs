use ndarray::ShapeError;
use ndarray_linalg::error::LinalgError;
use polars::error::{ErrString, PolarsError};
use pyo3::{PyErr, PyResult};
use crate::utils::error::RegressorError::*;

#[derive(Debug)]
pub enum RegressorError{
    NotFitted,
    SizeMismatch,
    NonNumericInput(String),
    InvalidInput(String),
    EmptyInput,
    Exception(String),
}


pub type RegressorResult<T> = Result<T, RegressorError>;

const MODEL_NOT_FITTED_MSG: &str = "Model not fitted yet";
const SIZE_MISMATCH_MSG: &str = "Size of Features and Target dont match";
const NON_NUMERIC_INPUT_MSG: &str = "Input is not numeric";
const INVALID_INPUT_MSG: &str = "Invalid Input";
const EMPTY_INPUT_MSG: &str = "Empty Input";

impl From<RegressorError> for PyErr{
    fn from(err: RegressorError) -> Self {
        match err{
            NotFitted => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(MODEL_NOT_FITTED_MSG),
            SizeMismatch => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(SIZE_MISMATCH_MSG),
            EmptyInput => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(EMPTY_INPUT_MSG),
            InvalidInput(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
            NonNumericInput(x) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(NON_NUMERIC_INPUT_MSG.to_owned() + &*x),
            Exception(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
        }
    }
}
impl From<RegressorError> for PolarsError{
    fn from(err: RegressorError) -> Self {
        match err{
            NotFitted => PolarsError::NoData(ErrString::from(MODEL_NOT_FITTED_MSG)),
            SizeMismatch => PolarsError::ShapeMismatch(ErrString::from(SIZE_MISMATCH_MSG)),
            NonNumericInput(x) => PolarsError::InvalidOperation(ErrString::from(NON_NUMERIC_INPUT_MSG.to_owned() + &*x)),
            InvalidInput(_) => PolarsError::InvalidOperation(ErrString::from(INVALID_INPUT_MSG)) ,
            EmptyInput => PolarsError::InvalidOperation(ErrString::from(EMPTY_INPUT_MSG)),
            Exception(e) => PolarsError::InvalidOperation(ErrString::from(e)),
        }
    }
}
impl From<PolarsError> for RegressorError{
    fn from(err: PolarsError) -> Self {
        match err{
            PolarsError::NoData(_) => NotFitted,
            PolarsError::ShapeMismatch(_) => SizeMismatch,
            PolarsError::InvalidOperation(_) => InvalidInput(INVALID_INPUT_MSG.to_string()),
            _ => Exception(err.to_string()),
        }

    }
}

impl From<LinalgError> for RegressorError{
    fn from(err:LinalgError) -> Self{
        match err{
            LinalgError::NotSquare { .. } => SizeMismatch,
            LinalgError::Lapack(x) => Exception(x.to_string()),
            LinalgError::InvalidStride { .. } => InvalidInput(INVALID_INPUT_MSG.to_string()),
            LinalgError::MemoryNotCont => Exception(INVALID_INPUT_MSG.to_string()),
            LinalgError::NotStandardShape { .. } => SizeMismatch,
            LinalgError::Shape(_) => SizeMismatch,
        }
    }
}