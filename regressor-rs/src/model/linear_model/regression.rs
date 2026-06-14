use crate::utils::error::{RegressorError, RegressorResult};
use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_linalg::Solve;

/// Prepend a column of ones to `x`, producing the design matrix used when a
/// model is configured to fit an intercept. The bias coefficient then lands at
/// index 0 of the solved `betas`, which `params()` peels off as the intercept.
pub fn with_intercept_column(x: &Array2<f64>) -> Array2<f64> {
    let ones = Array2::<f64>::ones((x.nrows(), 1));
    concatenate(Axis(1), &[ones.view(), x.view()])
        .expect("ones column shares its row count with x")
}

/// Solve the ordinary least-squares normal equations `XᵀX β = Xᵀy`.
///
/// This is the single, unified path for every feature count: callers are
/// expected to have already shaped `X` to include an intercept column when one
/// is wanted (the model layer does this when `fit_intercept` is set), so the
/// closed form here makes no special case for the single-feature problem.
pub fn linear_regression_betas(x: &Array2<f64>, y: &Array1<f64>) -> RegressorResult<Array1<f64>> {
    if x.nrows() != y.len() {
        return Err(Box::new(RegressorError::SizeMismatch));
    }
    if x.is_empty() {
        return Err(Box::new(RegressorError::EmptyInput));
    }

    let xt = x.t();
    let a = xt.dot(x);
    let b = xt.dot(y);
    let betas = a.solve_into(b)?;
    Ok(betas)
}

/// Solve the ridge (L2-penalized) normal equations
/// `(XᵀX + αI) β = Xᵀy`.
///
/// When `has_intercept` is true the first coefficient is treated as the bias
/// term and left un-penalized (its diagonal entry of the penalty is zeroed), so
/// the intercept is never shrunk toward zero.
pub fn ridge_regression_betas(
    x: &Array2<f64>,
    y: &Array1<f64>,
    alpha: f64,
    has_intercept: bool,
) -> RegressorResult<Array1<f64>> {
    if x.nrows() != y.len() {
        return Err(Box::new(RegressorError::SizeMismatch));
    }
    if x.is_empty() {
        return Err(Box::new(RegressorError::EmptyInput));
    }
    if alpha < 0.0 {
        return Err(Box::new(RegressorError::InvalidInput(
            "ridge alpha must be non-negative".to_string(),
        )));
    }

    let xt = x.t();
    let mut a = xt.dot(x);
    let mut penalty = Array2::<f64>::eye(x.ncols()) * alpha;
    if has_intercept {
        penalty[[0, 0]] = 0.0;
    }
    a = a + penalty;
    let b = xt.dot(y);
    let betas = a.solve_into(b)?;
    Ok(betas)
}

pub fn linear_regression(x: &Array2<f64>, betas: &Array1<f64>) -> Array1<f64> {
    x.dot(betas)
}

pub fn logistic_regression(x: &Array2<f64>, betas: &Array1<f64>) -> Array1<f64> {
    let z = x.dot(betas);
    z.mapv(|z| 1.0 / (1.0 + (-z).exp()))
}
