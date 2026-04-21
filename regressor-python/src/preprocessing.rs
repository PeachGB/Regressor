use ndarray::Array1;
use polars::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use pyo3::exceptions::PyValueError;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct StandardScaler{
    means:Option<Array1<f64>>,
    standard_deviations:Option<Array1<f64>>
}

#[pymethods]
impl StandardScaler{

    #[new]
    fn new() -> StandardScaler{
        StandardScaler{ means:None, standard_deviations:None}
    }
    fn fit(&mut self, df:PyDataFrame) -> PyResult<()>{
        let df:DataFrame = df.into();
        let n_cols = df.width();

        let mut mean = Array1::<f64>::zeros(n_cols);
        let mut std = Array1::<f64>::zeros(n_cols);

        for (i,col) in df.columns().iter().enumerate() {
            let Ok(column) = col.cast(&DataType::Float64) else {
                return Err(PyErr::new::<PyValueError, _>(format!("Failed to cast column {} to Float64", col.name())));
            };
            let Ok(column) = column.f64() else {
            return Err(PyErr::new::<PyValueError, _>(format!("Failed to cast column {} to Float64", col.name())));
        };
            mean[i] = column.mean().unwrap_or(0.0);
            std[i] = column.std(1).unwrap_or(1.0);
        }

        self.means = Some(mean);
        self.standard_deviations = Some(std);
        Ok(())
    }
    fn transform(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = df.into();
        let height = df.height();
        let means = self.means.as_ref().ok_or_else(||
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Scaler not trained (call fit first)")
        )?;
        let stds = self.standard_deviations.as_ref().ok_or_else(||
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Scaler not trained (call fit first)")
        )?;

        let mut transformed_cols = Vec::with_capacity(df.width());

        for (i, col) in df.columns().iter().enumerate() {
            let mean = means[i];
            let std = if stds[i] == 0.0 { 1.0 } else { stds[i] };
            let s = col
                .cast(&DataType::Float64)
                .map_err(
                |err| PyErr::new::<PyValueError, _>(err.to_string()))?
                .f64().map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?
                .to_owned();
            let res = ((s - mean) / std).into_column();
            transformed_cols.push(res);

        }


        let new_df = DataFrame::new(height,transformed_cols).map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(PyDataFrame(new_df))
    }

    fn fit_transform(&mut self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        self.fit(df.clone())?;
        self.transform(df)
    }
}


#[pyclass]
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct LabelEncoder;

#[pymethods]
impl LabelEncoder{

    #[new]
    fn new() -> LabelEncoder{
        LabelEncoder{}
    }
    pub fn fit(&self) -> PyResult<Series>{
        todo!();
    }

}