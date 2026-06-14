pub mod regression;

use crate::model::functions::gradient_descent;
use crate::model::linear_model::regression::{logistic_regression, with_intercept_column};
use crate::model::{Differentiable, Model};
use crate::utils::error::{RegressorError, RegressorResult};
use crate::utils::metrics::{mean_squared_error, r_squared, root_mean_squared_error};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

#[derive(Clone)]
pub enum Metric {
    R2,
    MSE,
    RMSE,
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Penalty {
    L1(f64),
    L2(f64),
}

/// Split a fitted `betas` vector into `(intercept, coefficients)`.
///
/// When the model fit an intercept the bias lives at index 0; otherwise there
/// is no intercept and it is reported as `0.0`.
fn split_intercept(betas: &Array1<f64>, fit_intercept: bool) -> (f64, Vec<f64>) {
    let mut coefs = betas.to_vec();
    if fit_intercept && !coefs.is_empty() {
        let intercept = coefs.remove(0);
        (intercept, coefs)
    } else {
        (0.0, coefs)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinearRegression {
    betas: Option<Array1<f64>>,
    #[serde(default = "default_true")]
    fit_intercept: bool,
}
impl LinearRegression {
    pub fn betas(&self) -> Option<&Array1<f64>> {
        self.betas.as_ref()
    }

    fn r_squared(&self, y_true: Array1<f64>, y_pred: Array1<f64>) -> RegressorResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err(RegressorError::SizeMismatch.into());
        }
        r_squared(&y_true, &y_pred).map_err(Into::into)
    }

    fn mse(&self, y_true: Array1<f64>, y_pred: Array1<f64>) -> RegressorResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err(RegressorError::SizeMismatch.into());
        }

        mean_squared_error(&y_true, &y_pred).map_err(Into::into)
    }
    fn rmse(&self, y_true: Array1<f64>, y_pred: Array1<f64>) -> RegressorResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err(RegressorError::SizeMismatch.into());
        }

        root_mean_squared_error(&y_true, &y_pred).map_err(Into::into)
    }
}
impl Model for LinearRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Output = Array1<f64>;

    fn fit(&mut self, x: &Self::Input, y: &Self::Target) -> RegressorResult<()> {
        let design = if self.fit_intercept {
            with_intercept_column(x)
        } else {
            x.clone()
        };
        let betas = regression::linear_regression_betas(&design, y)?;
        self.betas = Some(betas);
        Ok(())
    }

    fn predict(&self, x: &Self::Input) -> RegressorResult<Self::Output> {
        let Some(betas) = self.betas.as_ref() else {
            return Err(RegressorError::NotFitted.into());
        };
        let design = if self.fit_intercept {
            with_intercept_column(x)
        } else {
            x.clone()
        };
        Ok(regression::linear_regression(&design, betas))
    }
}
impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}
impl LinearRegression {
    /// Create a model that fits an intercept automatically.
    pub fn new() -> LinearRegression {
        LinearRegression {
            betas: None,
            fit_intercept: true,
        }
    }
    /// Create a model, choosing whether an intercept column is prepended for you.
    pub fn with_fit_intercept(fit_intercept: bool) -> LinearRegression {
        LinearRegression {
            betas: None,
            fit_intercept,
        }
    }
    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }
    pub fn params(&self) -> RegressorResult<(f64, Vec<f64>)> {
        let Some(betas) = self.betas() else {
            return Err(RegressorError::NotFitted.into());
        };
        Ok(split_intercept(betas, self.fit_intercept))
    }

    pub fn score(
        &self,
        y_true: Array1<f64>,
        y_pred: Array1<f64>,
        metric: Metric,
    ) -> RegressorResult<f64> {
        match metric {
            Metric::R2 => Ok(self.r_squared(y_true, y_pred)?),
            Metric::MSE => Ok(self.mse(y_true, y_pred)?),
            Metric::RMSE => Ok(self.rmse(y_true, y_pred)?),
        }
    }

    pub fn save(&self, path: String) -> RegressorResult<()> {
        let serialized =
            serde_json::to_string(self).map_err(|e| RegressorError::Exception(e.to_string()))?;
        std::fs::write(path, serialized).map_err(|e| RegressorError::Exception(e.to_string()))?;
        Ok(())
    }
    pub fn load(path: String) -> RegressorResult<Self> {
        let serialized =
            std::fs::read_to_string(path).map_err(|e| RegressorError::Exception(e.to_string()))?;
        let model: Self = serde_json::from_str(&serialized)
            .map_err(|e| RegressorError::Exception(e.to_string()))?;
        Ok(model)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Ridge {
    betas: Option<Array1<f64>>,
    alpha: f64,
    #[serde(default = "default_true")]
    fit_intercept: bool,
}

impl Ridge {
    /// Create a ridge regressor with L2 strength `alpha`, fitting an intercept.
    pub fn new(alpha: f64) -> Ridge {
        Ridge {
            betas: None,
            alpha,
            fit_intercept: true,
        }
    }
    pub fn with_fit_intercept(alpha: f64, fit_intercept: bool) -> Ridge {
        Ridge {
            betas: None,
            alpha,
            fit_intercept,
        }
    }
    pub fn betas(&self) -> Option<&Array1<f64>> {
        self.betas.as_ref()
    }
    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }
    pub fn params(&self) -> RegressorResult<(f64, Vec<f64>)> {
        let Some(betas) = self.betas() else {
            return Err(RegressorError::NotFitted.into());
        };
        Ok(split_intercept(betas, self.fit_intercept))
    }
    pub fn score(
        &self,
        y_true: Array1<f64>,
        y_pred: Array1<f64>,
        metric: Metric,
    ) -> RegressorResult<f64> {
        match metric {
            Metric::R2 => r_squared(&y_true, &y_pred).map_err(Into::into),
            Metric::MSE => mean_squared_error(&y_true, &y_pred).map_err(Into::into),
            Metric::RMSE => root_mean_squared_error(&y_true, &y_pred).map_err(Into::into),
        }
    }
    pub fn save(&self, path: String) -> RegressorResult<()> {
        let serialized =
            serde_json::to_string(self).map_err(|e| RegressorError::Exception(e.to_string()))?;
        std::fs::write(path, serialized).map_err(|e| RegressorError::Exception(e.to_string()))?;
        Ok(())
    }
    pub fn load(path: String) -> RegressorResult<Self> {
        let serialized =
            std::fs::read_to_string(path).map_err(|e| RegressorError::Exception(e.to_string()))?;
        let model: Self = serde_json::from_str(&serialized)
            .map_err(|e| RegressorError::Exception(e.to_string()))?;
        Ok(model)
    }
}

impl Model for Ridge {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Output = Array1<f64>;

    fn fit(&mut self, x: &Self::Input, y: &Self::Target) -> RegressorResult<()> {
        let design = if self.fit_intercept {
            with_intercept_column(x)
        } else {
            x.clone()
        };
        let betas =
            regression::ridge_regression_betas(&design, y, self.alpha, self.fit_intercept)?;
        self.betas = Some(betas);
        Ok(())
    }

    fn predict(&self, x: &Self::Input) -> RegressorResult<Self::Output> {
        let Some(betas) = self.betas.as_ref() else {
            return Err(RegressorError::NotFitted.into());
        };
        let design = if self.fit_intercept {
            with_intercept_column(x)
        } else {
            x.clone()
        };
        Ok(regression::linear_regression(&design, betas))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogisticRegression {
    betas: Option<Array1<f64>>,
    penalty: Option<Penalty>,
    learning_rate: f64,
    epochs: usize,
    #[serde(default = "default_true")]
    fit_intercept: bool,
}

impl LogisticRegression {
    pub fn betas(&self) -> Option<&Array1<f64>> {
        self.betas.as_ref()
    }
    pub fn set_penalty(&mut self, penalty: Penalty) {
        self.penalty = Some(penalty);
    }
    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }
}
impl Model for LogisticRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Output = Array1<f64>;

    fn fit(&mut self, x: &Self::Input, target: &Self::Target) -> RegressorResult<()> {
        let design = if self.fit_intercept {
            with_intercept_column(x)
        } else {
            x.clone()
        };
        let betas: Array1<f64> = Array1::zeros(design.ncols());

        let betas =
            gradient_descent(self, &design, target, betas, self.learning_rate, self.epochs)?;

        self.betas = Some(betas);
        Ok(())
    }

    fn predict(&self, x: &Self::Input) -> RegressorResult<Self::Output> {
        let Some(betas) = self.betas.as_ref() else {
            return Err(RegressorError::NotFitted.into());
        };
        let design = if self.fit_intercept {
            with_intercept_column(x)
        } else {
            x.clone()
        };
        let output = logistic_regression(&design, betas);
        Ok(output)
    }
}
impl Differentiable for LogisticRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Betas = Array1<f64>;

    fn compute_gradient(
        &self,
        x: &Self::Input,
        target: &Self::Target,
        betas: &Self::Betas,
    ) -> RegressorResult<Self::Betas> {
        let z = x.dot(betas);

        let y_pred = z.mapv(|z| 1.0 / (1.0 + (-z).exp()));

        let error = y_pred - target;

        let n = target.len() as f64;

        let mut gradient = x.t().dot(&error) / n;

        let Some(penalty) = &self.penalty else {
            return Ok(gradient);
        };
        match penalty {
            Penalty::L1(lambda) => {
                let mut l1 = betas.mapv(|x| x.signum() * lambda / n);
                if self.fit_intercept {
                    l1[0] = 0.0;
                }
                gradient = gradient + l1;
            }
            Penalty::L2(lambda) => {
                let mut l2 = betas.mapv(|x| x * lambda / n);
                if self.fit_intercept {
                    l2[0] = 0.0;
                }
                gradient = gradient + l2;
            }
        }

        Ok(gradient)
    }
}

impl LogisticRegression {
    pub fn new(learning_rate: f64, epochs: usize) -> LogisticRegression {
        LogisticRegression {
            betas: None,
            penalty: None,
            learning_rate,
            epochs,
            fit_intercept: true,
        }
    }
    pub fn with_fit_intercept(
        learning_rate: f64,
        epochs: usize,
        fit_intercept: bool,
    ) -> LogisticRegression {
        LogisticRegression {
            betas: None,
            penalty: None,
            learning_rate,
            epochs,
            fit_intercept,
        }
    }
    pub fn params(&self) -> RegressorResult<(f64, Vec<f64>)> {
        let Some(betas) = self.betas() else {
            return Err(RegressorError::NotFitted.into());
        };
        Ok(split_intercept(betas, self.fit_intercept))
    }
    pub fn score(&self, y_true: Array1<f64>, y_pred: Array1<f64>) -> RegressorResult<f64> {
        if y_true.len() != y_pred.len() {
            return Err(RegressorError::SizeMismatch.into());
        }
        if y_true.is_empty() {
            return Err(RegressorError::EmptyInput.into());
        }

        let y_pred = y_pred.mapv(|x| if x > 0.5 { 1 } else { 0 });
        let y_true = y_true.mapv(|x| if x > 0.5 { 1 } else { 0 });

        let hits = y_pred
            .iter()
            .zip(y_true.iter())
            .filter(|&(x, y)| x == y)
            .count();
        Ok(hits as f64 / y_true.len() as f64)
    }

    pub fn save(&self, path: String) -> RegressorResult<()> {
        let serialized =
            serde_json::to_string(self).map_err(|e| RegressorError::Exception(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    pub fn load(path: String) -> RegressorResult<Self> {
        let serialized =
            std::fs::read_to_string(path).map_err(|e| RegressorError::Exception(e.to_string()))?;
        let model: Self = serde_json::from_str(&serialized)
            .map_err(|e| RegressorError::Exception(e.to_string()))?;
        Ok(model)
    }
}
