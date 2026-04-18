use ndarray::{Array, Array1, Axis, Ix1, Ix2};

pub fn linear_regression_betas(x:Array<f64,Ix2>, y:Array<f64,Ix1>) -> Array<f64,Ix1> {
    if x.ncols() == 1{
        let Some(xhat) = x.mean() else {panic!()};
        let Some(yhat) = y.mean() else {panic!()};
        let beta1 ={
            let mut sum = 0f64;
            let mut sum_divisor = 0f64;
            for (xi,yi) in x.iter().zip(y.iter()){
                sum += (xi-&xhat)*(yi-&yhat);
                sum_divisor += (xi-&xhat)*(xi-&xhat);
            }
            sum/sum_divisor
        };
        let beta0 = &yhat - &beta1*&xhat;

        Array1::from_vec(vec![beta0,beta1])
    } else {
        todo!("multi lin reg fit")
    }
}
pub fn linear_regression(x:&Array<f64,Ix2>,betas:&Array<f64,Ix1>) -> Array<f64,Ix1> {x.dot(betas)}
