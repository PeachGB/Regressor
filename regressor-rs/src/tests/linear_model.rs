//! Tests for `model::linear_model` (LinearRegression / Ridge / LogisticRegression).

use crate::model::Model;
use crate::model::linear_model::{LinearRegression, LogisticRegression, Metric, Penalty, Ridge};
use crate::utils::error::RegressorError;
use ndarray::{Array1, array};

fn assert_variant(err: Box<dyn std::error::Error>, want: &RegressorError) {
    let got = err
        .downcast_ref::<RegressorError>()
        .expect("expected a RegressorError");
    assert_eq!(std::mem::discriminant(got), std::mem::discriminant(want));
}

/// Assert two prediction vectors agree (JSON round-tripping f64 can perturb the
/// last bit, so an exact `==` is too strict).
fn assert_predictions_close(a: &Array1<f64>, b: &Array1<f64>) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-9, "predictions diverged: {x} vs {y}");
    }
}

/// A unique, temporary file path for save/load round-trips.
fn temp_path(tag: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir()
        .join(format!("regressor_{tag}_{nanos}.json"))
        .to_string_lossy()
        .into_owned()
}

// --- LinearRegression -------------------------------------------------------

#[test]
fn linear_regression_recovers_known_coefficients() {
    // y = 1 + 2*x0 + 3*x1; the intercept is fit automatically (no ones column).
    let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| 1.0 + 2.0 * r[0] + 3.0 * r[1])
        .collect();

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let (intercept, coefs) = model.params().unwrap();
    assert!((intercept - 1.0).abs() < 1e-9);
    assert!((coefs[0] - 2.0).abs() < 1e-9);
    assert!((coefs[1] - 3.0).abs() < 1e-9);

    let preds = model.predict(&x).unwrap();
    let r2 = model.score(y, preds, Metric::R2).unwrap();
    assert!((r2 - 1.0).abs() < 1e-9);
}

#[test]
fn linear_regression_single_feature_fits_intercept() {
    // y = 4 + 3*x, single feature: the unified path must still recover both.
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let y = array![4.0, 7.0, 10.0, 13.0];

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let (intercept, coefs) = model.params().unwrap();
    assert!((intercept - 4.0).abs() < 1e-9);
    assert!((coefs[0] - 3.0).abs() < 1e-9);
}

#[test]
fn linear_regression_without_intercept_reports_zero_intercept() {
    // Caller supplies their own ones column; fit_intercept is off.
    let x = array![[1.0, 2.0], [1.0, 1.0], [1.0, 4.0], [1.0, 3.0]];
    let y: Array1<f64> = x.rows().into_iter().map(|r| 1.0 + 3.0 * r[1]).collect();

    let mut model = LinearRegression::with_fit_intercept(false);
    model.fit(&x, &y).unwrap();
    let (intercept, coefs) = model.params().unwrap();
    // No intercept peeled off: the bias lands in coefs[0].
    assert_eq!(intercept, 0.0);
    assert!((coefs[0] - 1.0).abs() < 1e-9);
    assert!((coefs[1] - 3.0).abs() < 1e-9);
}

#[test]
fn predict_and_params_before_fit_error() {
    let model = LinearRegression::new();
    assert_variant(
        model.predict(&array![[1.0]]).unwrap_err(),
        &RegressorError::NotFitted,
    );
    assert_variant(model.params().unwrap_err(), &RegressorError::NotFitted);
}

#[test]
fn linear_regression_score_dispatches_each_metric() {
    let mut model = LinearRegression::new();
    model.fit(&array![[1.0], [2.0], [3.0]], &array![1.0, 2.0, 3.0]).unwrap();
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0, 2.0, 5.0]; // one error of 2 -> MSE 4/3
    assert!((model.score(y_true.clone(), y_pred.clone(), Metric::MSE).unwrap() - 4.0 / 3.0).abs() < 1e-12);
    assert!(
        (model.score(y_true, y_pred, Metric::RMSE).unwrap() - (4.0_f64 / 3.0).sqrt()).abs() < 1e-12
    );
}

#[test]
fn linear_regression_save_load_roundtrip() {
    let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]];
    let y: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| 1.0 + 2.0 * r[0] + 3.0 * r[1])
        .collect();
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let path = temp_path("linreg");
    model.save(path.clone()).unwrap();
    let loaded = LinearRegression::load(path.clone()).unwrap();
    std::fs::remove_file(&path).ok();

    let (i0, c0) = model.params().unwrap();
    let (i1, c1) = loaded.params().unwrap();
    assert_eq!(i0, i1);
    assert_eq!(c0, c1);
    assert_predictions_close(&model.predict(&x).unwrap(), &loaded.predict(&x).unwrap());
}

// --- Ridge ------------------------------------------------------------------

#[test]
fn ridge_shrinks_coefficients() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x

    let mut ols = LinearRegression::new();
    ols.fit(&x, &y).unwrap();
    let (_, ols_coefs) = ols.params().unwrap();

    let mut ridge = Ridge::new(10.0);
    ridge.fit(&x, &y).unwrap();
    let (_, ridge_coefs) = ridge.params().unwrap();

    // Regularization pulls the slope below the OLS estimate but keeps it positive.
    assert!(ridge_coefs[0] < ols_coefs[0]);
    assert!(ridge_coefs[0] > 0.0);
}

#[test]
fn ridge_save_load_roundtrip() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let mut ridge = Ridge::new(1.0);
    ridge.fit(&x, &y).unwrap();

    let path = temp_path("ridge");
    ridge.save(path.clone()).unwrap();
    let loaded = Ridge::load(path.clone()).unwrap();
    std::fs::remove_file(&path).ok();

    assert_predictions_close(&ridge.predict(&x).unwrap(), &loaded.predict(&x).unwrap());
}

// --- LogisticRegression -----------------------------------------------------

#[test]
fn logistic_regression_separates_toy_data() {
    let x = array![[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut clf = LogisticRegression::new(0.5, 2000);
    clf.set_penalty(Penalty::L2(0.01));
    clf.fit(&x, &y).unwrap();
    let preds = clf.predict(&x).unwrap();
    let acc = clf.score(y, preds).unwrap();
    assert_eq!(acc, 1.0);
}

#[test]
fn logistic_regression_score_rejects_mismatch_and_empty() {
    let clf = LogisticRegression::new(0.5, 10);
    assert_variant(
        clf.score(array![1.0, 0.0], array![1.0]).unwrap_err(),
        &RegressorError::SizeMismatch,
    );
    assert_variant(
        clf.score(Array1::zeros(0), Array1::zeros(0)).unwrap_err(),
        &RegressorError::EmptyInput,
    );
}

#[test]
fn logistic_regression_save_load_roundtrip() {
    let x = array![[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut clf = LogisticRegression::new(0.5, 500);
    clf.fit(&x, &y).unwrap();

    let path = temp_path("logreg");
    clf.save(path.clone()).unwrap();
    let loaded = LogisticRegression::load(path.clone()).unwrap();
    std::fs::remove_file(&path).ok();

    assert_predictions_close(&clf.predict(&x).unwrap(), &loaded.predict(&x).unwrap());
}
