use crate::model::Differentiable;
use crate::utils::error::RegressorResult;

pub fn gradient_descent<T>(
    model: &T,
    x: &T::Input,
    y: &T::Target,
    mut betas: T::Betas,
    learning_rate: f64,
    epochs: usize,
) -> RegressorResult<T::Betas>
where
    T: Differentiable,
    T::Betas: std::ops::Sub<Output = T::Betas> + Clone,
    T::Betas: std::ops::Mul<f64, Output = T::Betas>,
{
    for _ in 0..epochs {
        let gradient = model.compute_gradient(&x, &y, &betas)?;
        betas = betas - (gradient * learning_rate);
    }
    Ok(betas)
}
