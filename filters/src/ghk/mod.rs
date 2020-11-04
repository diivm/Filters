use ndarray::{Array1, ScalarOperand};
use num::traits::Float;
use std::ops::{AddAssign, Mul};

#[derive(Default, Debug)]
struct State<T: Float + ScalarOperand> {
    x: Array1<T>,
    dx: Array1<T>,
    ddx: Array1<T>,
}

#[derive(Default, Debug)]
struct GhkFilterConfig<T: Float + ScalarOperand> {
    g: T,
    h: T,
    k: T,
}

fn update<T>(state: &mut State<T>, config: &GhkFilterConfig<T>, z: &Array1<T>, dt: &T) -> ()
where
    T: Float + ScalarOperand + AddAssign + Mul<f32, Output = T>,
{
    // check dimensions
    assert_eq!(state.x.raw_dim(), state.dx.raw_dim());
    assert_eq!(state.x.raw_dim(), state.ddx.raw_dim());
    assert_eq!(state.x.raw_dim(), z.raw_dim());

    let dt_sq = (*dt) * (*dt);

    // prediction step
    let ddx_prediction = state.ddx.clone();
    let dx_prediction = &state.dx + &(&state.ddx * (*dt));
    let x_prediction = &state.x + &(&state.dx * (*dt)) + &((&state.ddx * dt_sq) * 0.5);

    // update step
    let y = z - &x_prediction;
    state.x = &x_prediction + &(&y * config.g);
    state.dx = &dx_prediction + &(&y * config.h / (*dt));
    state.ddx = &ddx_prediction + &(&y * config.k / dt_sq);
}

#[cfg(test)]
mod UnitTests {
    use ndarray::{arr1, Array1};
    #[test]
    fn test_update_function_random_values() {
        // Values taken from:
        // https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/01-g-h-filter.ipynb
        let x = arr1::<f32>(&[1.0, 10.0, 100.0]);
        let x_dim = x.raw_dim();
        let dx = arr1::<f32>(&[10.0, 12.0, 0.2]);
        let ddx = Array1::<f32>::zeros(x_dim);
        let mut state = super::State::<f32> {
            x: x,
            dx: dx,
            ddx: ddx,
        };
        let config = super::GhkFilterConfig::<f32> {
            g: 0.8,
            h: 0.2,
            ..Default::default()
        };
        let z = arr1::<f32>(&[2.0, 11.0, 102.0]);
        let dt = 1.0;

        println!("Intital State: {:?}", state);
        super::update::<f32>(&mut state, &config, &z, &dt);
        println!("State after Updation: {:?}", state);

        assert_eq!(state.x, arr1::<f32>(&[3.7999997, 13.2, 101.64]));
        assert_eq!(state.dx, arr1::<f32>(&[8.2, 9.8, 0.5600006]));
    }
}
