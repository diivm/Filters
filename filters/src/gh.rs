use ndarray::Array1;

#[derive(Default)]
struct GhFilterOrder<T> {
    order: u32,
    g: T,
    h: Option<T>,
    k: Option<T>,
    x: Array1<T>,
    y: Array1<T>,
    z: Array1<T>,
    dt: T,
}
