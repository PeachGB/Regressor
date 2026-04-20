use crate::model::Differentiable;
use crate::utils::error::RegressorResult;

pub fn gradient_descent<T>(
    model: &T,
    x: &T::Input,
    y: &T::Target,
    mut betas: T::Betas,
    learning_rate: f64,
    epochs:usize

) -> RegressorResult<T::Betas>
where
    T: Differentiable, T::Betas: std::ops::Sub<Output = T::Betas> + Clone,
    f64: std::ops::Mul<T::Betas, Output = T::Betas>
{
    for _ in 0..epochs{
        let gradient = model.compute_gradient(&x, &y, &betas)?;
        betas = betas - (learning_rate * gradient);
    }
    Ok(betas)

}
