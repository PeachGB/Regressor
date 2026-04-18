use polars::error::{ErrString, PolarsError};
use pyo3::{PyErr, PyResult};
use crate::utils::error::RegressorError::SizeMismatch;

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
            RegressorError::NotFitted => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(MODEL_NOT_FITTED_MSG),
            RegressorError::SizeMismatch => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(SIZE_MISMATCH_MSG),
            RegressorError::EmptyInput => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(EMPTY_INPUT_MSG),
            RegressorError::InvalidInput(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
            RegressorError::NonNumericInput(x) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(NON_NUMERIC_INPUT_MSG.to_owned() + &*x),
            RegressorError::Exception(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
        }
    }
}
impl From<RegressorError> for PolarsError{
    fn from(err: RegressorError) -> Self {
        match err{
            RegressorError::NotFitted => PolarsError::NoData(ErrString::from(MODEL_NOT_FITTED_MSG)),
            RegressorError::SizeMismatch => PolarsError::ShapeMismatch(ErrString::from(SIZE_MISMATCH_MSG)),
            RegressorError::NonNumericInput(x) => PolarsError::InvalidOperation(ErrString::from(NON_NUMERIC_INPUT_MSG.to_owned() + &*x)),
            RegressorError::InvalidInput(_) => PolarsError::InvalidOperation(ErrString::from(INVALID_INPUT_MSG)) ,
            RegressorError::EmptyInput => PolarsError::InvalidOperation(ErrString::from(EMPTY_INPUT_MSG)),
            RegressorError::Exception(e) => PolarsError::InvalidOperation(ErrString::from(e)),
        }
    }
}
impl From<PolarsError> for RegressorError{
    fn from(err: PolarsError) -> Self {
        match err{
            PolarsError::NoData(_) => RegressorError::NotFitted,
            PolarsError::ShapeMismatch(_) => SizeMismatch,
            PolarsError::InvalidOperation(_) => RegressorError::InvalidInput(INVALID_INPUT_MSG.to_string()),
            _ => RegressorError::Exception(err.to_string()),
        }

    }
}

