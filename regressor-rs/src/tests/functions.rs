//! Tests for the generic `model::functions::gradient_descent` optimizer.

use crate::model::Differentiable;
use crate::model::functions::gradient_descent;
use crate::utils::error::RegressorResult;
use ndarray::{Array1, Array2, array};

/// Minimizes f(b) = ½‖b‖²; its gradient is `b`, so descent drives b -> 0.
struct Quadratic;
impl Differentiable for Quadratic {
    type Input = ();
    type Target = ();
    type Betas = Array1<f64>;
    fn compute_gradient(
        &self,
        _x: &(),
        _y: &(),
        betas: &Array1<f64>,
    ) -> RegressorResult<Array1<f64>> {
        Ok(betas.clone())
    }
}

/// Least-squares linear model: gradient = Xᵀ(Xb − y) / n.
struct LinearGd {
    x: Array2<f64>,
    y: Array1<f64>,
}
impl Differentiable for LinearGd {
    type Input = ();
    type Target = ();
    type Betas = Array1<f64>;
    fn compute_gradient(
        &self,
        _x: &(),
        _y: &(),
        betas: &Array1<f64>,
    ) -> RegressorResult<Array1<f64>> {
        let n = self.y.len() as f64;
        let error = self.x.dot(betas) - &self.y;
        Ok(self.x.t().dot(&error) / n)
    }
}

#[test]
fn descent_drives_betas_toward_minimum() {
    let betas = array![10.0, -4.0, 1.0];
    let out = gradient_descent(&Quadratic, &(), &(), betas, 0.1, 500).unwrap();
    assert!(
        out.iter().all(|b| b.abs() < 1e-6),
        "betas did not converge: {out:?}"
    );
}

#[test]
fn descent_recovers_known_slope() {
    // y = 2x through the origin.
    let model = LinearGd {
        x: array![[1.0], [2.0], [3.0], [4.0]],
        y: array![2.0, 4.0, 6.0, 8.0],
    };
    let out = gradient_descent(&model, &(), &(), Array1::zeros(1), 0.05, 5000).unwrap();
    assert!((out[0] - 2.0).abs() < 1e-3, "got slope {}", out[0]);
}

#[test]
fn zero_epochs_returns_initial_betas() {
    let betas = array![3.0, 5.0];
    let out = gradient_descent(&Quadratic, &(), &(), betas.clone(), 0.1, 0).unwrap();
    assert_eq!(out, betas);
}
