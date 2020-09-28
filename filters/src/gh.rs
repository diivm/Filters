use ndarray::ArrayD;
use num::traits::{Float, Zero};

#[derive(Default)]
struct GainParameters<T: Float> {
    g: T,
    h: T,
    k: T,
}

#[derive(Default)]
struct State<T: Float> {
    x: ArrayD<T>,
    y: ArrayD<T>,
    z: ArrayD<T>,
}

#[derive(Default)]
struct GhFilterConfig<T: Float> {
    order: usize,
    gain_parameters: GainParameters<T>,
    dt: T,
}

impl<T: Float> GhFilterConfig<T> {
    fn checks(order: &usize, g: &T, h: &T, k: &T) -> Result<usize, &'static str> {
        match order {
            0 => {
                if Zero::is_zero(h) && Zero::is_zero(k) {
                    Ok(*order)
                } else {
                    Err("h and k non-zero for order 0")
                }
            }
            1 => {
                if Zero::is_zero(k) {
                    Ok(*order)
                } else {
                    Err("k non-zero for order 1")
                }
            }
            2 => Ok(*order),
            _ => Err("Order should be in between 0 and 2"),
        }
    }
}

// fn update(&, z: DVector<T>) {
//     let g_param = (GhFilterOrder::<T>::check_valid_g(&self.order, &self.g).unwrap()).unwrap();
//     let h_param = (GhFilterOrder::<T>::check_valid_h(&self.order, &self.h).unwrap()).unwrap();
//     let k_param = (GhFilterOrder::<T>::check_valid_k(&self.order, &self.k).unwrap()).unwrap();

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
//             let dx = self.x.column(1);
//             let ddx = self.x.column(2);
//             let dxdt = dx * self.dt;
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
