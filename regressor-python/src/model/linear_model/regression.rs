use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;
use crate::utils::error::{RegressorError, RegressorResult};

pub fn linear_regression_betas(x: &Array2<f64>, y: &Array1<f64>) -> RegressorResult<Array1<f64>> {
    if x.nrows() != y.len() {
        return Err(RegressorError::SizeMismatch);
    }

    if x.ncols() <= 2 {
        let x = x.index_axis(Axis(1), 0).to_owned();

        let Some(x_hat) = x.mean() else {
            return Err(RegressorError::EmptyInput);
        };
        let Some(y_hat) = y.mean() else {
            return Err(RegressorError::EmptyInput);
        };

        let mut sum_dividend = 0f64;
        let mut sum_divisor = 0f64;

        for (xi, yi) in x.iter().zip(y.iter()) {
            sum_dividend += (xi - x_hat) * (yi - y_hat);
            sum_divisor += (xi - x_hat) * (xi - x_hat);
        }

        if sum_divisor == 0.0 {
            return Err(RegressorError::InvalidInput(
                "Cannot fit simple regression with zero variance in X".to_string(),
            ));
        }

        let beta1 = sum_dividend / sum_divisor;
        let beta0 = y_hat - beta1 * x_hat;

        Ok(Array1::from_vec(vec![beta0, beta1]))
    } else {
        let xt = x.t();
        let a = xt.dot(x);
        let b = xt.dot(y);
        let betas = a.solve_into(b)?;
        Ok(betas)
    }
}


pub fn linear_regression(x: &Array2<f64>, betas: &Array1<f64>) -> Array1<f64> {
    x.dot(betas)
}

pub fn logistic_regression(x: &Array2<f64>, betas: &Array1<f64>) -> Array1<f64> {
    let z = x.dot(betas);
    z.mapv(|z| 1.0 / (1.0 + (-z).exp()))
}
