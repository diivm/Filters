use ndarray::Array1;

#[derive(Default, Builder)]
struct GhFilterOrder<T> {
    order: u32,
    g: T,
    h: T,
    k: T,
    x: Array1<T>,
    y: Array1<T>,
    z: Array1<T>,
    dt: T,
}

impl<T> GhFilterOrder<T> {
    fn check_order(order: u32) -> Result<u32, &'static str> {
        match order {
            0 | 1 | 2 => Ok(order),
            _ => Err("Order should be in between 0 and 2"),
        }
    }
}
