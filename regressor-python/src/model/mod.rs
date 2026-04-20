use pyo3::prelude::*;
use crate::utils::error::RegressorResult;

pub mod linear_model;
pub mod functions;

pub type ModelOutput<T> = Vec<T>;
pub trait Model {
    type Input;
    type Target;
    type Output;
    fn fit(&mut self, x: &Self::Input, target: &Self::Target ) -> RegressorResult<()>;
    fn predict(&self, x: &Self::Input)-> RegressorResult<Self::Output>;

}

pub trait Differentiable{
    type Input;
    type Target;
    type Betas;
    fn compute_gradient(&self, x: &Self::Input, target: &Self::Target, betas: &Self::Betas ) -> RegressorResult<Self::Betas>;
}