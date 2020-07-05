use ndarray::prelude::*;

#[allow(dead_code)]
struct GhFilterOrder<T> {
  order: u32,
  g: T,
  h: T,
  k: T,
  x: Array3::<T>,
  dt: T,
}
