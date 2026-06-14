//! Tests for the `model` traits (`Model` / `Differentiable` contracts).

use crate::model::Model;
use crate::utils::error::RegressorResult;

/// A trivial model that "learns" the last target it was fitted on and predicts
/// it back for every row. Exercises the `Model` trait contract end to end.
struct Constant(f64);

impl Model for Constant {
    type Input = Vec<f64>;
    type Target = f64;
    type Output = Vec<f64>;

    fn fit(&mut self, _x: &Self::Input, target: &Self::Target) -> RegressorResult<()> {
        self.0 = *target;
        Ok(())
    }

    fn predict(&self, x: &Self::Input) -> RegressorResult<Self::Output> {
        Ok(x.iter().map(|_| self.0).collect())
    }
}

#[test]
fn model_fit_then_predict_roundtrips() {
    let mut model = Constant(0.0);
    model.fit(&vec![1.0, 2.0], &7.0).unwrap();
    assert_eq!(
        model.predict(&vec![0.0, 0.0, 0.0]).unwrap(),
        vec![7.0, 7.0, 7.0]
    );
}
