//! Tests for `model::linear_model::regression` (the closed-form math).

use crate::model::linear_model::regression::{
    linear_regression, linear_regression_betas, logistic_regression, ridge_regression_betas,
    with_intercept_column,
};
use crate::utils::error::RegressorError;
use ndarray::{Array1, Array2, array};

fn assert_variant(err: Box<dyn std::error::Error>, want: &RegressorError) {
    let got = err
        .downcast_ref::<RegressorError>()
        .expect("expected a RegressorError");
    assert_eq!(std::mem::discriminant(got), std::mem::discriminant(want));
}

#[test]
fn with_intercept_column_prepends_ones() {
    let x = array![[2.0, 3.0], [4.0, 5.0]];
    let design = with_intercept_column(&x);
    assert_eq!(design.dim(), (2, 3));
    assert_eq!(design.column(0).to_vec(), vec![1.0, 1.0]);
    assert_eq!(design.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn linear_betas_recover_known_coefficients() {
    // y = 1 + 2*x0 + 3*x1, solved on a design matrix with an intercept column.
    let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| 1.0 + 2.0 * r[0] + 3.0 * r[1])
        .collect();
    let design = with_intercept_column(&x);
    let betas = linear_regression_betas(&design, &y).unwrap();
    assert!((betas[0] - 1.0).abs() < 1e-9);
    assert!((betas[1] - 2.0).abs() < 1e-9);
    assert!((betas[2] - 3.0).abs() < 1e-9);
}

#[test]
fn linear_betas_reject_mismatch_and_empty() {
    let x = array![[1.0], [2.0]];
    assert_variant(
        linear_regression_betas(&x, &array![1.0]).unwrap_err(),
        &RegressorError::SizeMismatch,
    );
    let empty = Array2::<f64>::zeros((0, 0));
    assert_variant(
        linear_regression_betas(&empty, &Array1::<f64>::zeros(0)).unwrap_err(),
        &RegressorError::EmptyInput,
    );
}

#[test]
fn ridge_with_zero_alpha_matches_ols() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let design = with_intercept_column(&x);
    let ols = linear_regression_betas(&design, &y).unwrap();
    let ridge = ridge_regression_betas(&design, &y, 0.0, true).unwrap();
    assert!((ols[0] - ridge[0]).abs() < 1e-9);
    assert!((ols[1] - ridge[1]).abs() < 1e-9);
}

#[test]
fn ridge_rejects_negative_alpha() {
    let x = array![[1.0], [2.0]];
    let y = array![1.0, 2.0];
    assert_variant(
        ridge_regression_betas(&x, &y, -1.0, false).unwrap_err(),
        &RegressorError::InvalidInput(String::new()),
    );
}

#[test]
fn ridge_leaves_intercept_unpenalized() {
    // With a strong penalty the slope shrinks but the intercept stays free to fit.
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![12.0, 14.0, 16.0, 18.0, 20.0]; // y = 10 + 2x
    let design = with_intercept_column(&x);
    let ridge = ridge_regression_betas(&design, &y, 1000.0, true).unwrap();
    assert!(ridge[1].abs() < 2.0, "slope not shrunk: {}", ridge[1]);
    assert!(ridge[0] > 5.0, "intercept was shrunk: {}", ridge[0]);
}

#[test]
fn linear_regression_applies_betas() {
    let x = array![[1.0, 2.0], [1.0, 4.0]];
    let betas = array![3.0, 5.0];
    let preds = linear_regression(&x, &betas);
    assert_eq!(preds.to_vec(), vec![3.0 + 10.0, 3.0 + 20.0]);
}

#[test]
fn logistic_regression_is_sigmoid() {
    // z = 0 -> 0.5; large +z -> ~1; large -z -> ~0.
    let x = array![[0.0], [10.0], [-10.0]];
    let betas = array![1.0];
    let p = logistic_regression(&x, &betas);
    assert!((p[0] - 0.5).abs() < 1e-12);
    assert!(p[1] > 0.999);
    assert!(p[2] < 0.001);
}
