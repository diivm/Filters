use ndarray::{ArrayD, ScalarOperand};
use std::ops::AddAssign;
use num::traits::{Float, Zero};

#[derive(Default)]
struct State<T: Float + ScalarOperand> {
    x: Array1<T>,
    dx: Array1<T>,
    ddx: Array1<T>,
}

#[derive(Default)]
struct GhFilterConfig<T: Float + ScalarOperand> {
    order: usize,
    g: T,
    h: T,
    k: T,
}

impl<T: Float + ScalarOperand> GhFilterConfig<T> {
    fn checks(&self) -> Result<usize, &'static str> {
        match self.order {
            0 => {
                if Zero::is_zero(&self.h) && Zero::is_zero(&self.k) {
                    Ok(self.order)
                } else {
                    Err("h and k non-zero for order 0")
                }
            }
            1 => {
                if Zero::is_zero(&self.k) {
                    Ok(self.order)
                } else {
                    Err("k non-zero for order 1")
                }
            }
            2 => Ok(self.order),
            _ => Err("Order should be in between 0 and 2"),
        }
    }
}

fn update<T>(state: &mut State<T>, config: &GhFilterConfig<T>, dt: &T) -> ()
where
    T: Float + ScalarOperand + AddAssign,
{
    match GhFilterConfig::<T>::checks(config).unwrap() {
        0 => {
            state.y = &state.z - &state.x;
            state.x += &(&state.y * config.g);
        }

//     match GhFilterOrder::<T>::check_valid_order(&self.order).unwrap() {
//         0 => {
//             self.y = z - self.x;
//             self.x += &self.y * g_param;
//         }
//         1 => {
//             let x = self.x.column(0);
//             let dx = self.x.column(1);
//             let dxdt = dx * self.dt;

//             self.y = z.clone() - (&x + &dxdt);

//             let new_x = x.clone() + &dxdt + &self.y * g_param;
//             // // FIXME
//             // self.x.set_column(0, &new_x);

//             let new_dx = dx + &self.y * h_param / self.dt;
//             self.x.set_column(1, &new_dx);

//             self.z = z;
//         }
//         2 => {
//             let x = self.x.column(0);
        _ => panic!("Not implemented for order not in {0, 1, 2}"),
    }
}
//             let T2 = self.dt.powi(2);

//             // // FIXME
//             // self.y = z - x + dxdt + (ddx * T2) * 0.5;

//             // // FIXME
//             // let new_x = x + dxdt + ddx.mul_to(T2, ddx) + 0.5 * ddx * T2 + self.y * g_param;
//             // self.x.set_column(0, &new_x);

//             // let new_dx = dx + ddx * self.dt + self.y * h_param / self.dt;
//             // self.x.set_column(1, &new_dx);

//             // // FIXME
//             // let new_ddx = ddx + 2 * k_param * self.y / self.dt.powi(2);
//             // self.x.set_column(2, &new_ddx);
//         }
//         _ => panic!("Not implemented for order not in {0, 1, 2}"),
//     }
// }
