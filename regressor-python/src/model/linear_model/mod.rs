pub mod regression;

use ndarray::prelude::*;
use polars::frame::DataFrame;
use polars::series::Series;
use pyo3::{pyclass, pymethods, PyResult};
use pyo3_polars::{PyDataFrame, PySeries};
use crate::model::{Model, RegressorResult};
use crate::utils::{data_frame_to_ndarray, series_to_ndarray};
use crate::utils::error::RegressorError;

#[pyclass]
pub struct LinearRegression {
    betas: Option <Array1<f64>>,
}
impl LinearRegression {
    fn betas(&self) -> Option<Array1<f64>>{
        self.betas.clone()
    }

}
impl Model for LinearRegression {
    type Input = Array2<f64>;
    type Target = Array1<f64>;
    type Output = Array1<f64>;




    fn fit(&mut self, x: Self::Input, y: Self::Target) -> RegressorResult<()> {
        let betas = regression::linear_regression_betas(x, y)?;
        self.betas = Some(betas);
        Ok(())
        }

    fn predict(&self, x: Self::Input) -> RegressorResult<Self::Output> {
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
        let x:DataFrame = x.into();
        let y:Series = y.into();

        let x = data_frame_to_ndarray(x)?;
        let y = series_to_ndarray(y)?;

        Model::fit(self, x, y).map_err(Into::into)
        
    }
    fn predict(&self, x: PyDataFrame) -> PyResult<Vec<f64>> {
        let x:DataFrame = x.into();
        let x = data_frame_to_ndarray(x)?;
        let output = Model::predict(self, x)?;
        Ok(output.to_vec())
    }
    
}
















