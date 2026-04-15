use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
pub mod linear_model;



pub trait Model {
    fn fit(&mut self, x: PyDataFrame, target: PySeries ) -> PyResult<()>;
    fn predict(&self, x: PyDataFrame)-> PyResult<Vec<f64>>;
    fn score(&self, x:PyDataFrame, y: PySeries)-> PyResult<f64>;

}