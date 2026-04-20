pub mod regression;

use ndarray::prelude::*;
use polars::frame::DataFrame;
use polars::series::Series;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use pyo3_polars::{PyDataFrame, PySeries};
use serde::{Deserialize, Serialize};
use crate::model::{Differentiable, Model, RegressorResult};
use crate::model::functions::gradient_descent;
use crate::model::linear_model::regression::logistic_regression;
use crate::utils::{data_frame_to_model_matrix, series_to_model_vector};
use crate::utils::error::RegressorError;
use crate::utils::metrics::{mean_squared_error, r_squared, root_mean_squared_error};

 type LinearOutput = Vec<f64>;
fn linear_model_fit(x: PyDataFrame, y:PySeries) -> RegressorResult<(Array2<f64>,Array1<f64>)>{

        let x:DataFrame = x.into();
        let y:Series = y.into();

        let x = data_frame_to_model_matrix(x)?;
        let y = series_to_model_vector(y)?;
    Ok((x,y))

}

#[pyclass]
#[derive(Clone)]
pub enum Metric {
    R2,
    MSE,
    RMSE
}
#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinearRegression {
    betas: Option <Array1<f64>>,
}
impl LinearRegression {
    fn betas(&self) -> Option<Array1<f64>>{
        self.betas.clone()
    }



    fn r_squared(&self, y_true: PySeries, y_pred: PySeries) -> RegressorResult<f64> {
        let y_true:Series = y_true.into();
        let y_pred:Series = y_pred.into();
        if y_true.len() != y_pred.len(){return Err(RegressorError::SizeMismatch.into())}
        let y_true = series_to_model_vector(y_true)?;
        let y_pred = series_to_model_vector(y_pred)?;
        r_squared(&y_true, &y_pred).map_err(Into::into)

    }

    fn mse(&self, y_true: PySeries, y_pred: PySeries) -> RegressorResult<f64> {
        let y_true:Series = y_true.into();
        let y_pred:Series = y_pred.into();
        if y_true.len() != y_pred.len() {return Err(RegressorError::SizeMismatch.into())}

        let y_true = series_to_model_vector(y_true)?;
        let y_pred = series_to_model_vector(y_pred)?;

        mean_squared_error(&y_true,&y_pred).map_err(Into::into)
    }
    fn rmse(&self, y_true: PySeries, y_pred: PySeries) -> RegressorResult<f64> {
        let y_true:Series = y_true.into();
        let y_pred:Series = y_pred.into();
        if y_true.len() != y_pred.len() {return Err(RegressorError::SizeMismatch.into())}

        let y_true = series_to_model_vector(y_true)?;
        let y_pred = series_to_model_vector(y_pred)?;

        root_mean_squared_error(&y_true,&y_pred).map_err(Into::into)

    }
}
impl Model for LinearRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Output = Array1<f64>;




    fn fit(&mut self, x: &Self::Input, y: &Self::Target) -> RegressorResult<()> {
        let betas = regression::linear_regression_betas(x, y)?;
        self.betas = Some(betas);
        Ok(())
        }

    fn predict(&self, x: &Self::Input) -> RegressorResult<Self::Output> {
        let Some(betas) = self.betas.as_ref() else {return Err(RegressorError::NotFitted.into())};

        Ok(regression::linear_regression(
            &x,
            betas
        ))
    }
}
#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> LinearRegression {
        LinearRegression {
            betas: None,
        }
    }
    fn params(&self) -> PyResult<(f64, Vec<f64>)> {

        let Some(betas) = self.betas() else {return Err(RegressorError::NotFitted.into())};

        let mut betas = betas.to_vec();

        let intercept = betas.remove(0);

        Ok((intercept,betas))
    }

    fn fit(&mut self, x: PyDataFrame, y: PySeries) -> PyResult<()> {
        let (x,y) = linear_model_fit(x,y)?.into();

        Model::fit(self, &x, &y).map_err(Into::into)
        
    }
    fn predict(&self, x: PyDataFrame) -> PyResult<LinearOutput> {
        let x:DataFrame = x.into();
        let x = data_frame_to_model_matrix(x)?;
        let output = Model::predict(self, &x)?.to_vec();
        Ok(output)

    }

    fn score(&self, y_true: PySeries, y_pred: PySeries, metric: Metric) -> PyResult<f64> {
        match metric {
            Metric::R2 => Ok(self.r_squared(y_true, y_pred)?),
            Metric::MSE => Ok(self.mse(y_true, y_pred)?),
            Metric::RMSE => Ok(self.rmse(y_true, y_pred)?),
        }
    }


    fn save(&self, path:String) -> PyResult<()> {
        let serialized = serde_json::to_string(self).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let serialized = std::fs::read_to_string(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let model: Self = serde_json::from_str(&serialized).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(model)
    }


}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogisticRegression {
    betas: Option <Array1<f64>>,
    learning_rate: f64,
    epochs: usize,
}

impl LogisticRegression {
    fn betas(&self) -> Option<&Array1<f64>>{
        if let Some(betas) = self.betas.as_ref() {
            return Some(&betas);
        }
        None
    }
}
impl Model for LogisticRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Output = Array1<f64>;

    fn fit(&mut self, x: &Self::Input, target: &Self::Target) -> RegressorResult<()> {
        let betas:Array1<f64> = Array1::zeros(x.ncols());

        let betas = gradient_descent(self, x, target, betas, self.learning_rate, self.epochs)?;

        self.betas = Some(betas);
        Ok(())

    }

    fn predict(&self, x: &Self::Input) -> RegressorResult<Self::Output> {
        let Some(betas) = self.betas.as_ref() else {return Err(RegressorError::NotFitted.into())};
        let output = logistic_regression(x, betas);
        Ok(output)
    }
}
impl Differentiable for LogisticRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Betas = Array1<f64>;

    fn compute_gradient(&self, x: &Self::Input, target: &Self::Target, betas: &Self::Betas) -> RegressorResult<Self::Betas> {

        let z = x.dot(betas);

        let y_pred = z.mapv(|z| 1.0 / (1.0 + (-z).exp()));

        let error = y_pred - target;

        let n = target.len() as f64;

        let gradient = x.t().dot(&error) / n;

        Ok(gradient)
    }
}

#[pymethods]
impl LogisticRegression{


    #[new]
    fn new(learning_rate: f64, epochs: usize) -> LogisticRegression {
        LogisticRegression {
            betas: None,
            learning_rate,
            epochs,
        }
    }
    fn params(&self) -> PyResult<(f64, Vec<f64>)> {

        let Some(betas) = self.betas() else {return Err(RegressorError::NotFitted.into())};
        let mut betas = betas.to_vec();
        let intercept = betas.remove(0);
        Ok((intercept,betas))
        }
    fn fit(&mut self, x: PyDataFrame, y: PySeries) -> PyResult<()> {
        let (x,y) = linear_model_fit(x,y)?.into();

        Model::fit(self, &x, &y).map_err(Into::into)
    }
    fn predict(&self, x: PyDataFrame) -> PyResult<LinearOutput> {
        let x:DataFrame = x.into();
        let x = data_frame_to_model_matrix(x)?;
        let output = Model::predict(self, &x)?.to_vec();
        Ok(output)
    }
    fn score(&self, y_true: PySeries, y_pred: PySeries) -> PyResult<f64> {
        let y_true:Series = y_true.into();
        let y_pred:Series = y_pred.into();

        if y_true.len() != y_pred.len() {return Err(RegressorError::SizeMismatch.into())}
        let y_true = series_to_model_vector(y_true)?;
        let y_pred = series_to_model_vector(y_pred)?;

        let y_pred = y_pred.mapv(|x| if x > 0.5 {1} else {0});
        let y_true = y_true.mapv(|x| if x > 0.5 {1} else {0});

        let hits = y_pred.iter().zip(y_true.iter()).filter(|&(x,y)| x == y).count();
        Ok(hits as f64 / y_true.len() as f64)
    }

    fn save(&self, path:String) -> PyResult<()> {
        let serialized = serde_json::to_string(self).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let serialized = std::fs::read_to_string(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let model: Self = serde_json::from_str(&serialized).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(model)
    }

}

















