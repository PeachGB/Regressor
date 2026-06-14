use crate::utils::error::{RegressorError, RegressorResult};
use ndarray::Array1;

pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> RegressorResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Box::new(RegressorError::SizeMismatch));
    }
    if y_pred == y_true {
        return Ok(1.0);
    }
    let Some(y_hat) = y_true.mean() else {
        return Err(Box::new(RegressorError::EmptyInput));
    };

    let ss_res = (y_true - y_pred).pow2().sum();
    let ss_tot = (y_true - y_hat).pow2().sum();
    if ss_tot == 0f64 {
        return Err(Box::new(RegressorError::Exception(String::from(
            "Cannot compute R^2 with zero variance in y_true ",
        ))));
    }
    Ok(1f64 - (ss_res / ss_tot))
}

pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> RegressorResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Box::new(RegressorError::SizeMismatch));
    }
    if y_true.is_empty() {
        return Err(Box::new(RegressorError::EmptyInput));
    }
    let mse = (y_true - y_pred).pow2().sum() / (y_true.len() as f64);

    Ok(mse)
}

pub fn root_mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> RegressorResult<f64> {
    Ok(mean_squared_error(y_true, y_pred)?.sqrt())
}
