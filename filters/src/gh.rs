use alga::general::Real;
use nalgebra::{DMatrix, DVector};

#[derive(Debug)]
struct GhFilterOrder<T: Real> {
    order: usize,
    g: Option<T>,
    h: Option<T>,
    k: Option<T>,
    x: DMatrix<T>,
    y: DVector<T>,
    z: DVector<T>,
    dt: T,
}

impl<T: Real> GhFilterOrder<T> {
    fn check_valid_order(order: &usize) -> Result<usize, &'static str> {
        match order {
            0 | 1 | 2 => Ok(*order),
            _ => Err("Order should be in between 0 and 2"),
        }
    }

    fn check_valid_g(order: &usize, g: &Option<T>) -> Result<Option<T>, &'static str> {
        match GhFilterOrder::<T>::check_valid_order(order)? {
            _ => match g {
                None => Err("Cannot use None value"),
                Some(_) => Ok(*g),
            },
        }
    }

    fn check_valid_h(order: &usize, h: &Option<T>) -> Result<Option<T>, &'static str> {
        match GhFilterOrder::<T>::check_valid_order(order)? {
            0 => Err("Order should be >= 1"),
            _ => match h {
                None => Err("Cannot use None value"),
                Some(_) => Ok(*h),
            },
        }
    }

    fn check_valid_k(order: &usize, k: &Option<T>) -> Result<Option<T>, &'static str> {
        match GhFilterOrder::<T>::check_valid_order(order)? {
            0 | 1 => Err("Order should be >= 2"),
            _ => match k {
                None => Err("Cannot use None value"),
                Some(_) => Ok(*k),
            },
        }
    }

    fn new(order: &usize, g: &Option<T>, x: &DMatrix<T>, dt: &T) -> GhFilterOrder<T> {
        GhFilterOrder {
            order: GhFilterOrder::<T>::check_valid_order(order).unwrap(),
            g: GhFilterOrder::<T>::check_valid_g(order, g).unwrap(),
            h: None,
            k: None,
            x: x.clone(),
            y: DVector::zeros(x.nrows()),
            z: DVector::zeros(x.nrows()),
            dt: *dt,
        }
    }

    fn set_g(&mut self, g: &Option<T>) {
        self.g = GhFilterOrder::<T>::check_valid_g(&self.order, g).unwrap();
    }

    fn set_h(&mut self, h: &Option<T>) {
        self.h = GhFilterOrder::<T>::check_valid_h(&self.order, h).unwrap();
    }

    fn set_k(&mut self, k: &Option<T>) {
        self.h = GhFilterOrder::<T>::check_valid_k(&self.order, k).unwrap();
    }

    fn update(&mut self, z: DVector<T>) {
        let g_param = (GhFilterOrder::<T>::check_valid_g(&self.order, &self.g).unwrap()).unwrap();
        let h_param = (GhFilterOrder::<T>::check_valid_h(&self.order, &self.h).unwrap()).unwrap();
        let k_param = (GhFilterOrder::<T>::check_valid_k(&self.order, &self.k).unwrap()).unwrap();

        match GhFilterOrder::<T>::check_valid_order(&self.order).unwrap() {
            0 => {
                self.y = z - &self.x;
                self.x += &self.y * g_param;
            }
            1 => {
                let x = self.x.column(0);
                let dx = self.x.column(1);
                let dxdt = dx * self.dt;

                self.y = z.clone() - (&x + &dxdt);

                let new_x = x.clone() + &dxdt + &self.y * g_param;
                // // FIXME
                // self.x.set_column(0, &new_x);

                let new_dx = dx + &self.y * h_param / self.dt;
                self.x.set_column(1, &new_dx);

                self.z = z;
            }
            2 => {
                let x = self.x.column(0);
                let dx = self.x.column(1);
                let ddx = self.x.column(2);
                let dxdt = dx * self.dt;
                let T2 = self.dt.powi(2);

                // // FIXME
                // self.y = z - x + dxdt + (ddx * T2) * 0.5;

                // // FIXME
                // let new_x = x + dxdt + ddx.mul_to(T2, ddx) + 0.5 * ddx * T2 + self.y * g_param;
                // self.x.set_column(0, &new_x);

                // let new_dx = dx + ddx * self.dt + self.y * h_param / self.dt;
                // self.x.set_column(1, &new_dx);

                // // FIXME
                // let new_ddx = ddx + 2 * k_param * self.y / self.dt.powi(2);
                // self.x.set_column(2, &new_ddx);
            }
            _ => panic!("Not implemented for order not in {0, 1, 2}"),
        }
    }
}
