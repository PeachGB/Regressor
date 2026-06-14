//! Tests for `preprocessing` (StandardScaler / MinMaxScaler / LabelEncoder).

use crate::preprocessing::{LabelEncoder, MinMaxScaler, StandardScaler};
use crate::utils::error::RegressorError;
use ndarray::{Array2, array};

fn assert_variant(err: Box<dyn std::error::Error>, want: &RegressorError) {
    let got = err
        .downcast_ref::<RegressorError>()
        .expect("expected a RegressorError");
    assert_eq!(std::mem::discriminant(got), std::mem::discriminant(want));
}

// --- StandardScaler ---------------------------------------------------------

#[test]
fn standard_scaler_zero_mean_unit_std() {
    let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let mut scaler = StandardScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    for col in z.columns() {
        assert!(col.mean().unwrap().abs() < 1e-12);
        assert!((col.std(1.0) - 1.0).abs() < 1e-12); // sample std (ddof=1)
    }
}

#[test]
fn standard_scaler_zero_variance_column_is_centered_not_nan() {
    let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]];
    let mut scaler = StandardScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    // Constant column -> all zeros, no division-by-zero NaNs.
    for v in z.column(0) {
        assert_eq!(*v, 0.0);
    }
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn standard_scaler_transform_before_fit_errors() {
    let scaler = StandardScaler::new();
    assert_variant(
        scaler.transform(&array![[1.0]]).unwrap_err(),
        &RegressorError::NotFitted,
    );
}

#[test]
fn standard_scaler_rejects_empty_and_mismatched() {
    let mut scaler = StandardScaler::new();
    assert_variant(
        scaler.fit(&Array2::<f64>::zeros((0, 0))).unwrap_err(),
        &RegressorError::EmptyInput,
    );
    scaler.fit(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
    assert_variant(
        scaler.transform(&array![[1.0]]).unwrap_err(),
        &RegressorError::SizeMismatch,
    );
}

// --- MinMaxScaler -----------------------------------------------------------

#[test]
fn minmax_scaler_unit_range() {
    let x = array![[1.0], [3.0], [5.0]];
    let mut scaler = MinMaxScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    assert!((z[[0, 0]] - 0.0).abs() < 1e-12);
    assert!((z[[2, 0]] - 1.0).abs() < 1e-12);
}

#[test]
fn minmax_scaler_zero_range_column_maps_to_zero() {
    let x = array![[7.0], [7.0], [7.0]];
    let mut scaler = MinMaxScaler::new();
    let z = scaler.fit_transform(&x).unwrap();
    assert!(z.iter().all(|v| *v == 0.0));
}

#[test]
fn minmax_scaler_transform_before_fit_errors() {
    let scaler = MinMaxScaler::new();
    assert_variant(
        scaler.transform(&array![[1.0]]).unwrap_err(),
        &RegressorError::NotFitted,
    );
}

#[test]
fn minmax_scaler_rejects_empty() {
    let mut scaler = MinMaxScaler::new();
    assert_variant(
        scaler.fit(&Array2::<f64>::zeros((0, 0))).unwrap_err(),
        &RegressorError::EmptyInput,
    );
}

// --- LabelEncoder -----------------------------------------------------------

fn labels(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn label_encoder_round_trips_and_orders_classes() {
    let mut enc = LabelEncoder::new();
    let codes = enc.fit_transform(&labels(&["b", "a", "c", "a"])).unwrap();
    // sorted classes: a=0, b=1, c=2
    assert_eq!(codes.to_vec(), vec![1.0, 0.0, 2.0, 0.0]);
    assert_eq!(enc.classes(), Some(labels(&["a", "b", "c"]).as_slice()));
    let back = enc.inverse_transform(&codes.to_vec()).unwrap();
    assert_eq!(back, labels(&["b", "a", "c", "a"]));
}

#[test]
fn label_encoder_unknown_label_errors() {
    let mut enc = LabelEncoder::new();
    enc.fit(&labels(&["a", "b"])).unwrap();
    assert_variant(
        enc.transform(&labels(&["z"])).unwrap_err(),
        &RegressorError::InvalidInput(String::new()),
    );
}

#[test]
fn label_encoder_before_fit_errors() {
    let enc = LabelEncoder::new();
    assert_variant(
        enc.transform(&labels(&["a"])).unwrap_err(),
        &RegressorError::NotFitted,
    );
}

#[test]
fn label_encoder_out_of_range_code_errors() {
    let mut enc = LabelEncoder::new();
    enc.fit(&labels(&["a", "b"])).unwrap();
    assert_variant(
        enc.inverse_transform(&[5.0]).unwrap_err(),
        &RegressorError::InvalidInput(String::new()),
    );
}

#[test]
fn label_encoder_rejects_empty_fit() {
    let mut enc = LabelEncoder::new();
    assert_variant(
        enc.fit(&[]).unwrap_err(),
        &RegressorError::EmptyInput,
    );
}
