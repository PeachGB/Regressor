use ndarray::prelude::*;
use polars::frame::DataFrame;
use polars::series::Series;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use pyo3::exceptions::PyException;
use pyo3_polars::{PyDataFrame, PySeries};
use crate::model::Model;

#[pyclass]
#[derive(Clone,Copy)]
pub enum Solver {
    Simple,
    Multiple,
    GradientDescent,
    Auto,
}

#[pyclass]
pub struct LinearRegression {
    betas: Option <Array1<f64>>,
    n_features: usize,
    solver: Option<Solver>,


}
impl LinearRegression {
    fn set_betas_and_features(&mut self, x: &DataFrame){
        let n_features = x.width();
        self.n_features = n_features;
        let betas = Array1::<f64>::zeros(n_features);
        self.betas = Some(betas);
    }
    fn solver(&self) -> Solver{
        match self.solver{
            Some(solver) => solver.clone(),
            _=> panic!("Solver not set yet"),

        }
    }
    fn betas(&self) -> Option<Array1<f64>>{
        self.betas.clone()
    }

}
impl Model for LinearRegression {
    fn fit(&mut self, x: PyDataFrame, y: PySeries) -> PyResult<()> {
        let x:DataFrame = x.into();
        let y:Series = y.into();
        self.set_betas_and_features(&x);
        let n_samples = x.height();

        if self.solver.is_none() {
            match self.n_features {
                1 => self.solver = Some(Solver::Simple),
                _ => self.solver = Some(Solver::Multiple),

            }
        }

        select_fit_for_solver(self)(self,&x,&y)
        }




    fn predict(&self, x: PyDataFrame) -> PyResult<Vec<f64>> {
        if self.solver.is_none() || self.betas.is_none() {
           return Err(PyErr::new::<PyException, &str>("Model not fitted yet"))
        }

        let x:DataFrame = x.into();

        select_predict_for_solver(self)(self,&x)








    }

    fn score(&self, x: PyDataFrame, y: PySeries) -> PyResult<f64> {
        todo!()
    }
}
#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> LinearRegression {
        LinearRegression {
            betas: None,
            n_features: 0,
            solver: None,
        }
    }
    fn params(&self) -> PyResult<(f64, Vec<f64>)> {

        let Some(betas) = self.betas() else {return err_model_not_fitted()};

        let mut betas = betas.to_vec();

        let intercept = betas.remove(0);

        Ok((intercept,betas))
    }

    fn fit(&mut self,x: PyDataFrame, target: PySeries) -> PyResult<()> {
        Model::fit(self, x, target)
    }
    fn predict(&self, x: PyDataFrame) -> PyResult<Vec<f64>> {
        Model::predict(self, x)
    }
    fn score(&self, x: PyDataFrame, y: PySeries) -> PyResult<f64> {
        Model::score(self, x, y)
    }
}

fn simple_lin_reg_df_to_vec(x:&DataFrame) -> Vec<f64>{

        if let Some(x_1) = x.select_at_idx(0){
            let x_1:Vec<f64> = x_1.f64().unwrap().into_iter().map(|n|n.unwrap()).collect();
            x_1
        }
        else {
            unreachable!();
        }}

fn simple_lin_reg_fit(reg: &mut LinearRegression, x: &DataFrame, y: &Series)  -> PyResult<()>{
    if x.height() != y.len(){
        return Err(PyErr::new::<PyException, &str>("Size of Features and Target dont match"));
    }

    let x :Vec<f64>= simple_lin_reg_df_to_vec(x);
    let y:Vec<f64> = y.f64().unwrap().into_iter().map(|n| n.unwrap()).collect();


    let x = Array1::from(x);
    let y = Array1::from(y);

    let x_hat = x.mean().unwrap();
    let y_hat = y.mean().unwrap();

    let beta1 = {
        let mut sum_dividend:f64 = 0.0;
        let mut sum_divisor: f64 = 0.0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let mean_deviationx = xi - x_hat;
            sum_dividend += mean_deviationx * (yi - y_hat);
            sum_divisor += mean_deviationx * mean_deviationx;
        }
        sum_divisor / sum_dividend
    };

    let beta0 = y_hat - beta1*x_hat;

    reg.betas = Some(array![beta0,beta1]);
    Ok(())
}








fn multiple_lin_reg(reg:&mut LinearRegression,x: &DataFrame, y: &Series) -> PyResult<()>{todo!()}
fn gradient_descent_fit(reg: &mut LinearRegression ,x: &DataFrame, y: &Series) -> PyResult<()> {todo!()}

fn select_fit_for_solver(reg: &LinearRegression) -> fn(&mut LinearRegression,&DataFrame,&Series) -> PyResult<()>{

    let solver = reg.solver();
    match solver{
        Solver::Simple => simple_lin_reg_fit,

        Solver::Multiple => multiple_lin_reg,
        Solver::GradientDescent => gradient_descent_fit ,
        Solver::Auto => {
            if reg.n_features == 1{
                simple_lin_reg_fit
            }
            else {
                multiple_lin_reg
            }
        }
    }
}

fn err_model_not_fitted<T>() -> PyResult<T>{
    Err(PyErr::new::<PyException, &str>("Model not fitted yet"))
}
fn simple_lin_reg_predict(reg:&LinearRegression, x: &DataFrame) -> PyResult<Vec<f64>>{

    let Some(ref betas)= reg.betas else {return err_model_not_fitted()};

    let Some(beta0) = betas.get(0usize) else {return err_model_not_fitted()};
    let Some(beta1) = betas.get(1usize) else {return err_model_not_fitted()};

    let x :Vec<f64>= simple_lin_reg_df_to_vec(x);

    let mut output:Vec<f64> = vec![];

    for xn in x{
        output.push((xn*beta1)+beta0)
    }

    Ok(output)

}

fn multiple_lin_reg_predict(reg: &LinearRegression, df: &DataFrame) -> PyResult<Vec<f64>>{todo!()}
fn gradient_descent_predict(reg:&LinearRegression, df: &DataFrame) -> PyResult<Vec<f64>>{todo!()}

fn select_predict_for_solver(reg: &LinearRegression) -> fn(&LinearRegression,&DataFrame) -> PyResult<Vec<f64>>{

    match reg.solver(){
        Solver::Simple => simple_lin_reg_predict,
        Solver::Multiple => multiple_lin_reg_predict,
        Solver::GradientDescent => gradient_descent_predict,
        Solver::Auto => unreachable!()
    }
}

#[cfg(test)]
mod tests{

    use super::*;


}
