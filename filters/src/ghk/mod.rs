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
