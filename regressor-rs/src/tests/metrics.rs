//! Tests for `utils::metrics`.

use crate::utils::error::RegressorError;
use crate::utils::metrics::{mean_squared_error, r_squared, root_mean_squared_error};
use ndarray::{Array1, array};

fn assert_variant(err: Box<dyn std::error::Error>, want: &RegressorError) {
    let got = err
        .downcast_ref::<RegressorError>()
        .expect("expected a RegressorError");
    assert_eq!(
        std::mem::discriminant(got),
        std::mem::discriminant(want),
        "wrong RegressorError variant"
    );
}

#[test]
fn r_squared_is_one_for_perfect_predictions() {
    let y = array![1.0, 2.0, 3.0, 4.0];
    assert_eq!(r_squared(&y, &y).unwrap(), 1.0);
}

#[test]
fn r_squared_is_zero_for_mean_predictions() {
    // Predicting the mean (2.0) of y_true gives R^2 = 0.
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![2.0, 2.0, 2.0];
    assert!((r_squared(&y_true, &y_pred).unwrap() - 0.0).abs() < 1e-12);
}

#[test]
fn r_squared_rejects_size_mismatch() {
    assert_variant(
        r_squared(&array![1.0, 2.0], &array![1.0]).unwrap_err(),
        &RegressorError::SizeMismatch,
    );
}

#[test]
fn r_squared_rejects_zero_variance_target() {
    let y_true = array![5.0, 5.0, 5.0];
    let y_pred = array![1.0, 2.0, 3.0];
    assert_variant(
        r_squared(&y_true, &y_pred).unwrap_err(),
        &RegressorError::Exception(String::new()),
    );
}

#[test]
fn mse_is_mean_of_squared_errors() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 5.0]; // single error of 2 -> 4/3
    assert!((mean_squared_error(&y_true, &y_pred).unwrap() - 4.0 / 3.0).abs() < 1e-12);
}

#[test]
fn mse_rejects_mismatch_and_empty() {
    assert_variant(
        mean_squared_error(&array![1.0, 2.0], &array![1.0]).unwrap_err(),
        &RegressorError::SizeMismatch,
    );
    let empty = Array1::<f64>::zeros(0);
    assert_variant(
        mean_squared_error(&empty, &empty).unwrap_err(),
        &RegressorError::EmptyInput,
    );
}

#[test]
fn rmse_is_sqrt_of_mse() {
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 5.0];
    let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
    assert!((rmse - (4.0_f64 / 3.0).sqrt()).abs() < 1e-12);
}
