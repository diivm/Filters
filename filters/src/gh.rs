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
