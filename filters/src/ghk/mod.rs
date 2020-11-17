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
struct GHKFilterConfig<T: Float + ScalarOperand> {
    g: T,
    h: T,
    k: T,
}

fn update<T>(state: &mut State<T>, config: &GHKFilterConfig<T>, z: &Array1<T>, dt: &T) -> ()
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
    use gnuplot::{Caption, Color, Figure};
    use ndarray::{arr1, Array1};
    use rand::random;
    use std::env;

    #[test]
    fn test_update_function_random_values() {
        /*
        **********
        Values taken from:
        https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/01-g-h-filter.ipynb
        **********
        */
        let x = arr1::<f32>(&[1.0, 10.0, 100.0]);
        let x_dim = x.raw_dim();
        let dx = arr1::<f32>(&[10.0, 12.0, 0.2]);
        let ddx = Array1::<f32>::zeros(x_dim);
        let mut state = super::State::<f32> {
            x: x,
            dx: dx,
            ddx: ddx,
        };
        let config = super::GHKFilterConfig::<f32> {
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

    /*
    **********
    Tests from https://github.com/rlabbe/filterpy/blob/master/filterpy/gh/tests/test_gh.py
    **********
    */
    #[test]
    fn test_least_squares() {
        /*
        There is an alternative form for computing h for the least squares.
        It works for all but the very first term (n=0). Use it to partially test
        the output of least_squares_parameters(). This test does not test that
        g is correct.
        */
        for n in 1..100 {
            let config = super::least_squares_parameters(n);
            let h2 = 4.0
                - 2.0 * config.g
                - (4.0 * (config.g - 2.0).powf(2.0) - 3.0 * config.g.powf(2.0)).powf(0.5);
            assert!((h2 - config.h).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(GRAPH_TESTS)]
    fn test_optimal_smoothing() {
        fn fx(x: f32) -> Array1<f32> {
            arr1::<f32>(&[0.1 * x.powf(2.0) + 3.0 * x - 4.0])
        }

        let config = super::optimal_noise_smoothing(0.2);

        let x = arr1::<f32>(&[4.0]);
        let x_dim = x.raw_dim();
        let dx = Array1::<f32>::zeros(x_dim);
        let ddx = Array1::<f32>::zeros(x_dim);
        let mut state = super::State::<f32> {
            x: x,
            dx: dx,
            ddx: ddx,
        };
        let dt = 1.0;

        let mut ys = Vec::new();
        let mut zs = Vec::new();

        for i in 0..100 {
            let z = fx(i as f32) + rand::random::<f32>() * 10.0;
            super::update::<f32>(&mut state, &config, &z, &dt);
            ys.push(state.x[0]);
            zs.push(z[0]);
        }

        let time: Vec<u32> = (0..100).collect();
        let mut fg = Figure::new();
        fg.axes2d()
            .lines(&ys, &time, &[Caption("State x"), Color("blue")]);
        fg.axes2d()
            .lines(&zs, &time, &[Caption("Observation"), Color("red")]);
        fg.show();
    }
}
