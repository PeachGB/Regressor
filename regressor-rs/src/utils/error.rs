use std::error::Error;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RegressorError {
    #[error("Model not fitted yet")]
    NotFitted,
    #[error("Size of Features and Target dont match")]
    SizeMismatch,
    #[error("Invalid Input")]
    InvalidInput(String),
    #[error("Empty Input")]
    EmptyInput,
    #[error("Exception: {0}")]
    Exception(String),
}

pub type RegressorResult<T> = Result<T, Box<dyn Error>>;
