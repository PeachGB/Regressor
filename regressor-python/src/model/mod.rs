use pyo3::prelude::*;
use crate::utils::error::RegressorResult;

pub mod linear_model;



pub trait Model {
    type Input;
    type Target;
    type Output;
    fn fit(&mut self, x: Self::Input, target: Self::Target ) -> RegressorResult<()>;
    fn predict(&self, x: Self::Input)-> RegressorResult<Self::Output>;

}