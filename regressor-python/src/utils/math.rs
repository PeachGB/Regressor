use ndarray::linalg::general_mat_vec_mul;
use ndarray::prelude::*;



pub fn derivative(f:fn(f64) -> f64) -> impl Fn(f64) -> f64{
    let h:f64 = 1e-8;
     move |x| -> f64 {
        (f(x+h) - f(x))/(h)
    }
}


pub fn linear_equation(x:f64,intercept:f64,slope:f64) -> f64{
    (slope*x) + intercept
}
